"""Assemble data/corpus_corr3: corrector training passages never seen by inverter_v2.
  = general train[2.6M:] (fresh, embeddings reused) + data/fresh/*.jsonl (new general passages)
    + data/fresh_domain/*.jsonl (new domain passages); val = data/corpus_v2/val (fresh by construction)."""
import json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.embed_corpus import load_encoder, encode

N_GEN = 2600000


def read(path):
    return [l for l in open(path, encoding="utf-8").read().split("\n") if l] if os.path.exists(path) else []


def main():
    rng = random.Random(0)
    gen = read("data/corpus/train.jsonl")
    gen_emb = np.load("data/corpus/train.emb.npy", mmap_mode="r")
    fresh_lines = []
    for d in ("data/fresh", "data/fresh_domain"):
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".jsonl"):
                    fresh_lines += read(os.path.join(d, f))
    seen = set(json.loads(l)["text"] for l in gen[:N_GEN])  # exclude accidental overlaps with inverter data
    seen |= set(json.loads(l)["text"] for l in read("data/domain/train.wiki.jsonl") + read("data/domain/train.web.jsonl"))
    fresh_lines = [l for l in dict.fromkeys(fresh_lines) if json.loads(l)["text"] not in seen]
    print("fresh new passages:", len(fresh_lines), flush=True)
    enc = load_encoder()
    fresh_emb = encode(enc, [json.loads(l)["text"] for l in fresh_lines], verbose=True)

    lines = gen[N_GEN:] + fresh_lines
    emb = np.concatenate([np.asarray(gen_emb[N_GEN:]), fresh_emb])
    perm = np.array(rng.sample(range(len(lines)), len(lines)))
    os.makedirs("data/corpus_corr3", exist_ok=True)
    with open("data/corpus_corr3/train.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(lines[p] for p in perm) + "\n")
    np.save("data/corpus_corr3/train.emb.npy", emb[perm])
    with open("data/corpus_corr3/val.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(read("data/corpus_v2/val.jsonl")) + "\n")
    np.save("data/corpus_corr3/val.emb.npy", np.load("data/corpus_v2/val.emb.npy"))
    print("corpus_corr3 train", len(lines), "val", len(read("data/corpus_v2/val.jsonl")), flush=True)


if __name__ == "__main__":
    main()
