"""Calibrate the key-rotation period X: simulate an attacker who knows p (text, rotated vector) pairs under one key,
estimates the rotation by orthogonal Procrustes, de-rotates other vectors and runs the inverter on them.

  python scripts/attack_pairs.py --pairs 0,50,100,200,300,500,1000 --n_test 60 --profile short --steps 5 --beam 4

Reading: the largest p for which BLEU/F1 stay at the no-key baseline is the number of *known* pairs a key may be
exposed to; X = that number / (fraction of texts the attacker can know), with a safety factor of 2.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2t.env import quiet_offline
quiet_offline()
from make_rotation import rotation_matrix
from invert import resolve_models
from v2t.infer import Inverter
from scripts.eval import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/corpus_v2", help="dir with train.emb.npy (pairs) and test.jsonl/test.emb.npy (victims)")
    ap.add_argument("--pairs", default="0,25,50,100,150,200,300,500,1000")
    ap.add_argument("--n_test", type=int, default=60)
    ap.add_argument("--profile", choices=["short", "long"], default="short")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--key_seed", default="a3f1c9", help="hex seed of the (secret) 90-degree key")
    args = ap.parse_args()
    K, _ = rotation_matrix(1024, 0, args.key_seed, "random", fixed_angle=90)
    E = np.load(os.path.join(args.data, "train.emb.npy"), mmap_mode="r")
    texts = [json.loads(l)["text"] for l in open(os.path.join(args.data, "test.jsonl"), encoding="utf-8")][:args.n_test]
    T = np.load(os.path.join(args.data, "test.emb.npy"))[:args.n_test].astype(np.float64)
    inv_p, cor_p, _ = resolve_models(args.profile, None, None)
    inv = Inverter(inv_p, cor_p)
    inv.no_repeat_ngram_size = 6 if inv.inv.max_tokens > 64 else 0
    print(f"models {inv_p} + {cor_p}; {args.n_test} victim texts; steps={args.steps} beam={args.beam}")
    print(" p pairs | cos(de-rotated, original) |  BLEU |   F1  | exact")
    for p in [int(x) for x in args.pairs.split(",")]:
        if p == 0:
            Mh = np.eye(1024)
        else:
            X = np.asarray(E[:p], dtype=np.float64); Y = X @ K.T
            U, S, Vt = np.linalg.svd(Y.T @ X); Mh = U @ Vt
        rec = (T @ K.T) @ Mh
        cos = float(((rec * T).sum(1) / np.linalg.norm(T, axis=1) ** 2).mean())
        preds, c = inv.invert(rec.astype(np.float32), steps=args.steps, beam=args.beam)
        m = metrics(preds, texts, c)
        print(f"  {p:6d}  |           {cos:.3f}           | {m['bleu']:5.1f} | {m['token_f1']:5.1f} | {m['exact']:4.1f} %", flush=True)
    preds, c = inv.invert(T.astype(np.float32), steps=args.steps, beam=args.beam)
    m = metrics(preds, texts, c)
    print(f"  (no rotation at all: BLEU {m['bleu']:.1f} | F1 {m['token_f1']:.1f} | exact {m['exact']:.1f} %)")


if __name__ == "__main__":
    main()
