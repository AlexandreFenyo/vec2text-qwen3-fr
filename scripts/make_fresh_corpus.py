"""Assemble a corrector-training corpus from freshly built passage files (never seen by the inverter):
embeds the passages, dedups against the inverter's training texts, writes {out}/train.*, copies a val split,
and optionally embeds a domain test split.

  python scripts/make_fresh_corpus.py --dirs data/fresh128 data/fresh_domain128 --exclude data/corpus128/train.jsonl \
      --val data/corpus128 --test_domain data/fresh_domain128/test.web.jsonl --out data/corpus_corr128
"""
import argparse, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.embed_corpus import load_encoder, encode


def read(path):
    return [l for l in open(path, encoding="utf-8").read().split("\n") if l] if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="directories with train*.jsonl passage files")
    ap.add_argument("--exclude", nargs="*", default=[], help="jsonl files whose texts must not appear (inverter data)")
    ap.add_argument("--val", required=True, help="directory providing val.jsonl / val.emb.npy")
    ap.add_argument("--test_domain", default=None, help="jsonl of domain test passages to embed as test_domain")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(0)
    lines = []
    for d in args.dirs:
        for f in sorted(os.listdir(d)):
            if f.startswith("train") and f.endswith(".jsonl"):
                lines += read(os.path.join(d, f))
    seen = set()
    for p in args.exclude:
        seen |= {json.loads(l)["text"] for l in read(p)}
    lines = [l for l in dict.fromkeys(lines) if json.loads(l)["text"] not in seen]
    rng.shuffle(lines)
    print("fresh passages:", len(lines), flush=True)
    enc = load_encoder()
    os.makedirs(args.out, exist_ok=True)
    emb = encode(enc, [json.loads(l)["text"] for l in lines], bs=256, verbose=True)
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    np.save(os.path.join(args.out, "train.emb.npy"), emb)
    with open(os.path.join(args.out, "val.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(read(os.path.join(args.val, "val.jsonl"))) + "\n")
    np.save(os.path.join(args.out, "val.emb.npy"), np.load(os.path.join(args.val, "val.emb.npy")))
    if args.test_domain:
        t = read(args.test_domain)
        with open(os.path.join(args.out, "test_domain.jsonl"), "w", encoding="utf-8") as f:
            f.write("\n".join(t) + "\n")
        np.save(os.path.join(args.out, "test_domain.emb.npy"), encode(enc, [json.loads(l)["text"] for l in t], bs=256, verbose=False))
        print("test_domain:", len(t), flush=True)
    print("done:", args.out, len(lines), flush=True)


if __name__ == "__main__":
    main()
