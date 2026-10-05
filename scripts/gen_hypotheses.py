"""Generate step-0 hypotheses (greedy) with a trained inverter for the train/val splits, then embed them.
Produces data/corpus/{split}.hyp.jsonl and {split}.hyp.emb.npy (training data for the corrector)."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2t.model import Vec2TextQwen, get_tokenizer
from scripts.embed_corpus import load_encoder, encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inverter", required=True)
    ap.add_argument("--data", default="data/corpus")
    ap.add_argument("--splits", default="val,train")
    ap.add_argument("--limit", type=int, default=None, help="only the first N train samples")
    ap.add_argument("--start", type=int, default=0, help="train: start at this row and APPEND to an existing hyp file")
    ap.add_argument("--bs", type=int, default=1024)
    args = ap.parse_args()

    tok = get_tokenizer()
    model = Vec2TextQwen.load(args.inverter)
    for split in args.splits.split(","):
        emb = np.load(os.path.join(args.data, f"{split}.emb.npy"), mmap_mode="r")
        n = emb.shape[0] if (split != "train" or not args.limit) else min(args.limit, emb.shape[0])
        out_path = os.path.join(args.data, f"{split}.hyp.jsonl")
        t0 = time.time()
        texts = []
        start = args.start if split == "train" else 0
        if start:
            assert sum(1 for _ in open(out_path, encoding="utf-8")) == start, "existing hyp file must have exactly --start rows"
        with open(out_path, "a" if start else "w", encoding="utf-8") as f:
            for i in range(start, n, args.bs):
                e = torch.from_numpy(np.asarray(emb[i:i + args.bs], dtype=np.float32))
                ids = model.generate(e, do_sample=False)
                for row in ids:
                    t = tok.decode(row, skip_special_tokens=True).strip()
                    texts.append(t)
                    f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                if (i // args.bs) % 50 == 0:
                    print(f"{split}: {min(i+args.bs, n)}/{n} {time.time()-t0:.0f}s", flush=True)
        print(f"{split}: generated {len(texts)} in {time.time()-t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    enc = load_encoder()
    for split in args.splits.split(","):
        texts = [json.loads(l)["text"] for l in open(os.path.join(args.data, f"{split}.hyp.jsonl"), encoding="utf-8")]
        np.save(os.path.join(args.data, f"{split}.hyp.emb.npy"), encode(enc, texts))
        print(f"{split}: embedded", flush=True)


if __name__ == "__main__":
    main()
