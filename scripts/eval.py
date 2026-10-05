"""Evaluate inversion quality on the test split (BLEU, token F1, exact match, cosine)."""
import argparse, json, os, sys, time, re
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2t.infer import Inverter


def words(s):
    return re.findall(r"\w+|[^\w\s]", s.lower())


def token_f1(pred, ref):
    p, r = Counter(words(pred)), Counter(words(ref))
    common = sum((p & r).values())
    if common == 0:
        return 0.0
    prec, rec = common / max(1, sum(p.values())), common / max(1, sum(r.values()))
    return 2 * prec * rec / (prec + rec)


def metrics(preds, refs, cos):
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    f1 = float(np.mean([token_f1(p, r) for p, r in zip(preds, refs)]))
    exact = float(np.mean([p.strip() == r.strip() for p, r in zip(preds, refs)]))
    return dict(bleu=round(bleu, 2), token_f1=round(100 * f1, 2), exact=round(100 * exact, 2), cos=round(float(np.mean(cos)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inverter", default="ckpt/inverter")
    ap.add_argument("--corrector", default=None)
    ap.add_argument("--data", default="data/corpus")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--configs", default="0x1,0x4,1x4,5x4", help="comma list of stepsxbeam")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--no_repeat", type=int, default=0, help="no_repeat_ngram_size at decode time (0 = off)")
    args = ap.parse_args()

    texts = [json.loads(l)["text"] for l in open(os.path.join(args.data, f"{args.split}.jsonl"), encoding="utf-8")][:args.n]
    E = np.load(os.path.join(args.data, f"{args.split}.emb.npy"), mmap_mode="r")[:args.n]
    inv = Inverter(args.inverter, args.corrector)
    inv.no_repeat_ngram_size = args.no_repeat
    results = {}
    for cfg in args.configs.split(","):
        steps, beam = map(int, cfg.split("x"))
        if steps > 0 and inv.corr is None:
            continue
        t0 = time.time()
        preds, cos = inv.invert(E, steps=steps, beam=beam)
        m = metrics(preds, texts, cos)
        m["sec_per_sample"] = round((time.time() - t0) / len(texts), 3)
        results[cfg] = m
        print(f"[{cfg}] {m}", flush=True)
        for p, r in list(zip(preds, texts))[:args.show]:
            print(f"   ref: {r}\n   out: {p}")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
