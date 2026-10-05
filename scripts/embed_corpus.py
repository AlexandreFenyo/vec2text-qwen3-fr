"""Encode a jsonl of texts with Qwen3-Embedding-0.6B (document mode: no instruction) -> fp16 .npy"""
import argparse, json, time
import numpy as np, torch
from sentence_transformers import SentenceTransformer

EMB = "Qwen/Qwen3-Embedding-0.6B"


def load_encoder():
    return SentenceTransformer(EMB, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})


def encode(model, texts, bs=512, chunk=100_000, verbose=True):
    out = np.zeros((len(texts), 1024), dtype=np.float16)
    t0 = time.time()
    for i in range(0, len(texts), chunk):
        e = model.encode(texts[i:i + chunk], batch_size=bs, convert_to_numpy=True, normalize_embeddings=True)
        out[i:i + chunk] = e.astype(np.float16)
        if verbose:
            print(f"{min(i+chunk, len(texts))}/{len(texts)} {time.time()-t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--bs", type=int, default=512)
    args = ap.parse_args()
    texts = [json.loads(l)["text"] for l in open(args.inp, encoding="utf-8")]
    m = load_encoder()
    np.save(args.out, encode(m, texts, bs=args.bs))
    print("saved", args.out)
