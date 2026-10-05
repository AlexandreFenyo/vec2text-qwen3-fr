"""Corpus loading: texts (jsonl), token ids, precomputed embeddings (fp16 .npy)."""
import json, os
import numpy as np
import torch


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f]


def merge_sources(dir_, split, sources=("wiki", "web")):
    texts = []
    for s in sources:
        p = os.path.join(dir_, f"{split}.{s}.jsonl")
        if os.path.exists(p):
            texts += read_jsonl(p)
    return texts


class Corpus:
    """texts + embeddings (+ optional hypotheses & their embeddings) for one split."""

    def __init__(self, dir_, split, tok, with_hyp=False, limit=None, max_hyp=40):
        self.texts = read_jsonl(os.path.join(dir_, f"{split}.jsonl"))
        self.emb = np.load(os.path.join(dir_, f"{split}.emb.npy"), mmap_mode="r")
        assert len(self.texts) == self.emb.shape[0], (len(self.texts), self.emb.shape)
        if limit:
            self.texts, self.emb = self.texts[:limit], self.emb[:limit]
        self.ids = tok(self.texts, add_special_tokens=False)["input_ids"]
        self.hyp_ids = self.hyp_emb = None
        if with_hyp:
            self.hyp_texts = read_jsonl(os.path.join(dir_, f"{split}.hyp.jsonl"))
            self.hyp_emb = np.load(os.path.join(dir_, f"{split}.hyp.emb.npy"), mmap_mode="r")
            if limit:
                self.hyp_texts, self.hyp_emb = self.hyp_texts[:limit], self.hyp_emb[:limit]
            assert len(self.hyp_texts) == len(self.texts)
            self.hyp_ids = tok(self.hyp_texts, add_special_tokens=False)["input_ids"]
            self.hyp_ids = [h[:max_hyp] for h in self.hyp_ids]

    def __len__(self):
        return len(self.texts)

    def batch(self, idx):
        idx = np.asarray(idx)
        order = np.sort(idx)  # sorted reads are faster on memmaps
        e = torch.from_numpy(np.asarray(self.emb[order], dtype=np.float32))
        out = dict(e_target=e, target_ids=[self.ids[i] for i in order], idx=order)
        if self.hyp_ids is not None:
            out["e_hyp"] = torch.from_numpy(np.asarray(self.hyp_emb[order], dtype=np.float32))
            out["hyp_ids"] = [self.hyp_ids[i] for i in order]
        return out
