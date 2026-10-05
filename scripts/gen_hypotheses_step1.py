"""Second-round corrector data: apply a trained corrector once (greedy) to the step-0 hypotheses of a corpus,
then write a mixture corpus (step-0 and step-1 hypotheses for every text) usable by v2t.train --stage corrector.
  in : {data}/{split}.jsonl, .emb.npy, .hyp.jsonl, .hyp.emb.npy   (step-0 hypotheses)
  out: {out}/{split}.jsonl (texts x2), .emb.npy (x2), .hyp.jsonl (hyp0 + hyp1), .hyp.emb.npy"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2t.model import Vec2TextQwen, get_tokenizer
from scripts.embed_corpus import load_encoder, encode


def read_texts(p):
    return [json.loads(l)["text"] for l in open(p, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrector", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="val,train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bs", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = get_tokenizer()
    model = Vec2TextQwen.load(args.corrector)
    max_hyp = model.max_tokens + 32
    for split in args.splits.split(","):
        texts = read_texts(os.path.join(args.data, f"{split}.jsonl"))
        hyp0 = read_texts(os.path.join(args.data, f"{split}.hyp.jsonl"))
        emb = np.load(os.path.join(args.data, f"{split}.emb.npy"), mmap_mode="r")
        hemb = np.load(os.path.join(args.data, f"{split}.hyp.emb.npy"), mmap_mode="r")
        n = len(texts) if (split != "train" or not args.limit) else min(args.limit, len(texts))
        t0 = time.time()
        hyp1 = []
        for i in range(0, n, args.bs):
            e = torch.from_numpy(np.asarray(emb[i:i+args.bs], dtype=np.float32))
            eh = torch.from_numpy(np.asarray(hemb[i:i+args.bs], dtype=np.float32))
            ids = [tok(h, add_special_tokens=False)["input_ids"][:max_hyp] for h in hyp0[i:i+args.bs]]
            out = model.generate(e, eh, ids, do_sample=False)
            hyp1 += [tok.decode(o, skip_special_tokens=True).strip() for o in out]
            if (i // args.bs) % 50 == 0:
                print(f"{split}: {min(i+args.bs, n)}/{n} {time.time()-t0:.0f}s", flush=True)
        with open(os.path.join(args.out, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for t in texts[:n] + texts[:n]:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        with open(os.path.join(args.out, f"{split}.hyp.jsonl"), "w", encoding="utf-8") as f:
            for h in hyp0[:n] + hyp1:
                f.write(json.dumps({"text": h}, ensure_ascii=False) + "\n")
        e2 = np.asarray(emb[:n]); np.save(os.path.join(args.out, f"{split}.emb.npy"), np.concatenate([e2, e2]))
        print(f"{split}: step-1 hypotheses generated in {time.time()-t0:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()
    enc = load_encoder()
    for split in args.splits.split(","):
        hyps = read_texts(os.path.join(args.out, f"{split}.hyp.jsonl"))
        n2 = len(hyps) // 2
        h0e = np.asarray(np.load(os.path.join(args.data, f"{split}.hyp.emb.npy"), mmap_mode="r")[:n2])
        h1e = encode(enc, hyps[n2:], verbose=False)
        np.save(os.path.join(args.out, f"{split}.hyp.emb.npy"), np.concatenate([h0e, h1e]))
        print(f"{split}: embedded", flush=True)


if __name__ == "__main__":
    main()
