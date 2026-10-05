"""Inference: embedding -> text, with vec2text-style iterative correction and beam re-ranking by cosine."""
import numpy as np
import torch
from .model import Vec2TextQwen, get_tokenizer

EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class Inverter:
    def __init__(self, inverter_path, corrector_path=None, device="cuda", load_encoder=True):
        self.tok = get_tokenizer()
        self.inv = Vec2TextQwen.load(inverter_path, device=device)
        self.corr = Vec2TextQwen.load(corrector_path, device=device) if corrector_path else None
        self.device = device
        self.enc = None
        if load_encoder:
            from sentence_transformers import SentenceTransformer
            self.enc = SentenceTransformer(EMB_MODEL, device=device, model_kwargs={"torch_dtype": torch.bfloat16})

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def embed_texts(self, texts, bs=256):
        e = self.enc.encode(texts, batch_size=bs, convert_to_tensor=True, normalize_embeddings=True)
        return e.float().to(self.device)

    @torch.no_grad()
    def invert(self, E, steps=0, beam=1, max_new_tokens=None, bs=None, verbose=False, log=None):
        """E: (N, 1024) array/tensor of (normalized) embeddings. Returns (texts, cosines).

        steps=0: inversion model only (beam search if beam > 1).
        steps>0: vec2text correction loop; keeps the `beam` best hypotheses (by cosine with E) per sample.
        bs=None: batch size chosen from text length and beam width (the corrector step runs bs*beam*beam sequences).
        """
        if bs is None:
            bs = max(2, min(128, 1024 // (beam * beam) // max(1, self.inv.max_tokens // 32)))
        E = torch.as_tensor(np.asarray(E, dtype=np.float32) if not torch.is_tensor(E) else E).float().to(self.device)
        E = torch.nn.functional.normalize(E, dim=-1)
        if max_new_tokens is None:
            max_new_tokens = self.inv.max_tokens + 8
        out_texts, out_cos = [], []
        for i in range(0, E.shape[0], bs):
            t, c = self._invert_batch(E[i:i + bs], steps, beam, max_new_tokens, log=log)
            out_texts += t
            out_cos += c
            if verbose:
                print(f"{min(i+bs, E.shape[0])}/{E.shape[0]}", flush=True)
        return out_texts, out_cos

    no_repeat_ngram_size = 0  # >0 blocks repeated n-grams at decode time (helps against loops on long texts)
    min_new_tokens = 0        # >0 forbids the end-of-text token before this length (forces longer outputs)

    def _gen_kwargs(self, beam):
        kw = dict(do_sample=False)
        if self.no_repeat_ngram_size:
            kw["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        if self.min_new_tokens:
            kw["min_new_tokens"] = self.min_new_tokens
        if beam > 1:
            kw.update(num_beams=beam, num_return_sequences=beam, early_stopping=True)
        return kw

    def _invert_batch(self, E, steps, beam, max_new_tokens, log=None):
        B = E.shape[0]
        # --- step 0: inversion model
        cands = self.inv.generate(E, max_new_tokens=max_new_tokens, **self._gen_kwargs(beam))
        texts = [self.decode(c) for c in cands]  # B*beam
        # hypotheses per sample: list of dict(text, cos)
        hyps = [dict() for _ in range(B)]
        self._score_into(hyps, texts, E, beam)
        if log is not None:
            log.append([max(h.values()) for h in hyps])
        if steps == 0 or self.corr is None:
            return self._best(hyps)
        for _ in range(steps):
            # expand: for each sample, each kept hypothesis -> beam new candidates from the corrector
            flat_idx, flat_txt = [], []
            for b in range(B):
                keep = sorted(hyps[b].items(), key=lambda kv: -kv[1])[:beam]
                for txt, _ in keep:
                    flat_idx.append(b)
                    flat_txt.append(txt)
            e_t = E[flat_idx]
            e_h = self.embed_texts(flat_txt)
            max_hyp = max(self.inv.max_tokens + 32, max_new_tokens)
            hyp_ids = [self.tok(t, add_special_tokens=False)["input_ids"][:max_hyp] for t in flat_txt]
            cands = self.corr.generate(e_t, e_h, hyp_ids, max_new_tokens=max_new_tokens, **self._gen_kwargs(beam))
            new_texts = [self.decode(c) for c in cands]
            owner = [flat_idx[j // beam] for j in range(len(new_texts))]
            self._score_into(hyps, new_texts, E, beam, owner=owner)
            if log is not None:
                log.append([max(h.values()) for h in hyps])
            if all(max(h.values()) > 0.9995 for h in hyps):
                break
        return self._best(hyps)

    def _score_into(self, hyps, texts, E, beam, owner=None):
        """Embed candidate texts, compute cosine with their target, keep the best `beam` per sample."""
        B = E.shape[0]
        if owner is None:
            owner = [j // beam for j in range(len(texts))]
        # only embed texts not already scored
        todo = [(j, t) for j, t in enumerate(texts) if t not in hyps[owner[j]]]
        if todo:
            e = self.embed_texts([t for _, t in todo])
            cos = (e * E[[owner[j] for j, _ in todo]]).sum(-1).clamp(max=1.0).tolist()
            for (j, t), c in zip(todo, cos):
                hyps[owner[j]][t] = c
        for b in range(B):
            if len(hyps[b]) > beam:
                hyps[b] = dict(sorted(hyps[b].items(), key=lambda kv: -kv[1])[:beam])

    @staticmethod
    def _best(hyps):
        texts, cos = [], []
        for h in hyps:
            t, c = max(h.items(), key=lambda kv: kv[1])
            texts.append(t)
            cos.append(c)
        return texts, cos
