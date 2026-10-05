"""vec2text-style inverter built on Qwen3-0.6B-Base.

The embedding (1024-d, from Qwen3-Embedding-0.6B) is projected into `n_prefix` virtual tokens that are
prepended to the decoder input. The corrector variant also receives the embedding of the current
hypothesis, the difference target - hypothesis (each as `n_prefix` virtual tokens), and the hypothesis text:

    [prefix(e_target)] [prefix(e_hyp)] [prefix(e_target - e_hyp)] hyp_tokens <SEP> target_tokens <EOS>

Loss is applied on target_tokens + <EOS> only.
"""
import json, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-0.6B-Base"
EOS_ID = 151643   # <|endoftext|>
SEP_ID = 151644   # <|im_start|>, unused by the base model, serves as hypothesis/target separator


class EmbProj(nn.Module):
    def __init__(self, emb_dim, hid, n_prefix, tok_rms):
        super().__init__()
        self.n_prefix, self.hid = n_prefix, hid
        self.net = nn.Sequential(nn.Linear(emb_dim, hid), nn.GELU(), nn.Linear(hid, hid * n_prefix))
        self.norm = nn.LayerNorm(hid)
        with torch.no_grad():
            self.norm.weight.fill_(tok_rms)  # match the scale of the token embeddings

    def forward(self, e):
        x = self.net(e.to(self.net[0].weight.dtype)).view(e.shape[0], self.n_prefix, self.hid)
        return self.norm(x)


class Vec2TextQwen(nn.Module):
    def __init__(self, base=BASE, emb_dim=1024, n_prefix=8, corrector=False, dtype=torch.float32, max_tokens=32):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        self.cfg = dict(base=base, emb_dim=emb_dim, n_prefix=n_prefix, corrector=corrector, max_tokens=max_tokens)
        self.max_tokens = max_tokens  # longest text (in tokens) the model was trained to reconstruct
        hid = self.lm.config.hidden_size
        tok_rms = self.lm.get_input_embeddings().weight.detach().float().pow(2).mean(-1).sqrt().mean().item()
        self.proj_target = EmbProj(emb_dim, hid, n_prefix, tok_rms)
        self.corrector = corrector
        if corrector:
            self.proj_hyp = EmbProj(emb_dim, hid, n_prefix, tok_rms)
            self.proj_diff = EmbProj(emb_dim, hid, n_prefix, tok_rms)
        self.n_prefix = n_prefix

    @property
    def device(self):
        return self.lm.get_input_embeddings().weight.device

    def n_prefix_total(self):
        return self.n_prefix * (3 if self.corrector else 1)

    def prefix(self, e_target, e_hyp=None):
        """(B, P, hid) virtual tokens."""
        p = self.proj_target(e_target)
        if self.corrector:
            assert e_hyp is not None
            p = torch.cat([p, self.proj_hyp(e_hyp), self.proj_diff(e_target - e_hyp)], dim=1)
        return p

    def build_inputs(self, e_target, target_ids, e_hyp=None, hyp_ids=None, pad_to=1):
        """Right-padded training batch. target_ids / hyp_ids: lists of lists of token ids (no specials).
        pad_to: round the padded length up to a multiple (fewer distinct shapes for torch.compile)."""
        B = len(target_ids)
        P = self.n_prefix_total()
        seqs, labels = [], []
        for i in range(B):
            body = []
            if self.corrector:
                body += list(hyp_ids[i]) + [SEP_ID]
            tgt = list(target_ids[i]) + [EOS_ID]
            seqs.append(body + tgt)
            labels.append([-100] * (P + len(body)) + tgt)
        L = max(len(s) for s in seqs)
        L = ((L + pad_to - 1) // pad_to) * pad_to
        ids = torch.full((B, L), EOS_ID, dtype=torch.long)
        lab = torch.full((B, P + L), -100, dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labels)):
            ids[i, :len(s)] = torch.tensor(s)
            lab[i, :len(l)] = torch.tensor(l)
        dev = self.device
        ids, lab = ids.to(dev), lab.to(dev)
        tok_emb = self.lm.get_input_embeddings()(ids)
        pre = self.prefix(e_target.to(dev), None if e_hyp is None else e_hyp.to(dev)).to(tok_emb.dtype)
        return torch.cat([pre, tok_emb], dim=1), lab

    def forward(self, e_target, target_ids, e_hyp=None, hyp_ids=None, pad_to=1):
        """Returns (loss, n_tokens). Pads are at the right end only, so no attention mask is needed."""
        inputs_embeds, labels = self.build_inputs(e_target, target_ids, e_hyp, hyp_ids, pad_to=pad_to)
        hidden = self.lm.model(inputs_embeds=inputs_embeds).last_hidden_state
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        h = hidden[:, :-1][mask]
        logits = self.lm.lm_head(h).float()
        loss = F.cross_entropy(logits, shift_labels[mask])
        return loss, int(mask.sum())

    @torch.no_grad()
    def gen_inputs(self, e_target, e_hyp=None, hyp_ids=None):
        """Left-padded (inputs_embeds, attention_mask) for generation."""
        B = e_target.shape[0]
        dev = self.device
        pre = self.prefix(e_target.to(dev), None if e_hyp is None else e_hyp.to(dev))
        pre = pre.to(self.lm.get_input_embeddings().weight.dtype)
        if not self.corrector:
            return pre, torch.ones(B, pre.shape[1], dtype=torch.long, device=dev)
        bodies = [list(h) + [SEP_ID] for h in hyp_ids]
        L = max(len(b) for b in bodies)
        ids = torch.full((B, L), EOS_ID, dtype=torch.long)
        att = torch.zeros((B, pre.shape[1] + L), dtype=torch.long)
        for i, b in enumerate(bodies):
            ids[i, L - len(b):] = torch.tensor(b)
            att[i, L - len(b):] = 1  # left pad of the body; prefix comes after the pads
        ids, att = ids.to(dev), att.to(dev)
        tok_emb = self.lm.get_input_embeddings()(ids)
        # layout: [pads][prefix][body]; place prefix right before the body for each sample
        inputs = torch.zeros((B, pre.shape[1] + L, pre.shape[2]), dtype=pre.dtype, device=dev)
        for i, b in enumerate(bodies):
            npad = L - len(b)
            inputs[i, npad:npad + pre.shape[1]] = pre[i]
            inputs[i, npad + pre.shape[1]:] = tok_emb[i, npad:]
            att[i, npad:] = 1
        return inputs, att

    @torch.no_grad()
    def generate(self, e_target, e_hyp=None, hyp_ids=None, max_new_tokens=None, **kw):
        if max_new_tokens is None:
            max_new_tokens = self.max_tokens + 8
        inputs, att = self.gen_inputs(e_target, e_hyp, hyp_ids)
        out = self.lm.generate(inputs_embeds=inputs, attention_mask=att, max_new_tokens=max_new_tokens,
                               eos_token_id=EOS_ID, pad_token_id=EOS_ID, **kw)
        res = []
        for row in out.tolist():
            if EOS_ID in row:
                row = row[:row.index(EOS_ID)]
            res.append(row)
        return res

    # ---- checkpointing (projections + LM weights) ----
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        inner = self.lm.model
        if hasattr(inner, "_orig_mod"):  # torch.compile wrapper: save the plain module so keys stay standard
            self.lm.model = inner._orig_mod
        try:
            self.lm.save_pretrained(path, safe_serialization=True)
        finally:
            self.lm.model = inner
        extra = {k: v for k, v in self.state_dict().items() if k.startswith("proj_")}
        torch.save(extra, os.path.join(path, "proj.pt"))
        json.dump(self.cfg, open(os.path.join(path, "v2t_config.json"), "w"))

    @classmethod
    def load(cls, path, dtype=torch.bfloat16, device="cuda"):
        cfg = json.load(open(os.path.join(path, "v2t_config.json")))
        m = cls(base=path, emb_dim=cfg["emb_dim"], n_prefix=cfg["n_prefix"], corrector=cfg["corrector"], dtype=dtype,
                max_tokens=cfg.get("max_tokens", 32))
        extra = torch.load(os.path.join(path, "proj.pt"), map_location="cpu")
        missing, unexpected = m.load_state_dict(extra, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("lm.") for k in missing), [k for k in missing if not k.startswith("lm.")]
        m.cfg["base"] = cfg["base"]
        return m.to(device).eval()

    @classmethod
    def from_inverter(cls, path, dtype=torch.float32):
        """Initialize a corrector from a trained inversion model (LM weights + proj_target)."""
        cfg = json.load(open(os.path.join(path, "v2t_config.json")))
        m = cls(base=path, emb_dim=cfg["emb_dim"], n_prefix=cfg["n_prefix"], corrector=True, dtype=dtype,
                max_tokens=cfg.get("max_tokens", 32))
        extra = torch.load(os.path.join(path, "proj.pt"), map_location="cpu")
        m.load_state_dict(extra, strict=False)
        m.cfg["base"] = cfg["base"]
        return m


def get_tokenizer():
    return AutoTokenizer.from_pretrained(BASE)
