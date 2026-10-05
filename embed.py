#!/usr/bin/env python
"""Embed texts with Qwen3-Embedding-0.6B and write them in the TensorFlow Embedding Projector format
(vectors TSV: tab-separated floats, one line per text; metadata TSV: one line per text).

  python embed.py --texts phrases.txt --out vecs.tsv --metadata meta.tsv
  python embed.py --texts phrases.txt --out vecs.bytes --bytes      # raw float32 like the Projector demo data
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2t.env import quiet_offline
quiet_offline()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="one text per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metadata", default=None, help="write the texts as a Projector metadata TSV")
    ap.add_argument("--bytes", action="store_true", help="write raw float32 instead of TSV")
    args = ap.parse_args()
    texts = [l.rstrip("\n") for l in open(args.texts, encoding="utf-8") if l.strip()]
    from scripts.embed_corpus import load_encoder, encode
    E = encode(load_encoder(), texts, verbose=False).astype(np.float32)
    if args.bytes:
        E.tofile(args.out)
    else:
        np.savetxt(args.out, E, delimiter="\t", fmt="%.6f")
    if args.metadata:
        with open(args.metadata, "w", encoding="utf-8") as f:
            f.write("\n".join(t.replace("\t", " ") for t in texts) + "\n")
    print(f"wrote {len(texts)} vectors -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
