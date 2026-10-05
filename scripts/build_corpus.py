"""Build a corpus of short French passages (<= MAX_TOK Qwen tokens) from Wikipedia FR and FineWeb-2 FR.

Passages are random windows cut at word boundaries, so lengths are varied (4..MAX_TOK tokens),
with a bias toward full-length (MAX_TOK) windows. Output: JSONL lines {"text": ..., "src": ...}.
Train/val/test are split at the document level.
"""
import argparse, json, os, random, re, time
from datasets import load_dataset
from transformers import AutoTokenizer

MAX_TOK = 32
MIN_TOK = 4
P_FULL = 0.4  # probability of a full MAX_TOK window

BAD_RE = re.compile(r"[\u0000-\u0008\u000b-\u001f�]|https?://|www\.")


def iter_docs(source, data_files=None):
    """data_files: optional parquet path(s) inside the repo, e.g. 'data/fra_Latn/train/000_00010.parquet' (fresh shard)."""
    if source == "wiki":
        ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
    elif data_files:
        ds = load_dataset("parquet", data_files={"train": "hf://datasets/HuggingFaceFW/fineweb-2/" + data_files}, split="train", streaming=True)
    else:
        ds = load_dataset("HuggingFaceFW/fineweb-2", name="fra_Latn", split="train", streaming=True)
    for x in ds:
        yield x["text"]


def passages_from_doc(text, tok, rng, max_per_doc, MAX_TOK=MAX_TOK, MIN_TOK=MIN_TOK):
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) >= 60 and not BAD_RE.search(p)]
    if not paras:
        return []
    rng.shuffle(paras)
    out = []
    for p in paras:
        if len(out) >= max_per_doc:
            break
        ids = tok(p, add_special_tokens=False)["input_ids"]
        n = len(ids)
        if n < MIN_TOK:
            continue
        toks = tok.convert_ids_to_tokens(ids)
        starts = [0] + [i for i, t in enumerate(toks) if t.startswith("Ġ") and i > 0]
        L = MAX_TOK if rng.random() < P_FULL else rng.randint(MIN_TOK, MAX_TOK - 1)
        L = min(L, n)
        cand = [s for s in starts if s + L <= n] or [0]
        s = rng.choice(cand)
        e = min(s + L, n)
        # snap the end to a word boundary (a token that starts a word), looking back up to 3 tokens
        snap = [b for b in starts if e - 3 <= b <= e and b > s + MIN_TOK - 1]
        if snap and e < n:
            e = max(snap)
        txt = tok.decode(ids[s:e]).strip()
        if not txt or any(c in txt for c in "\n\x0b\x0c\x1c\x1d\x1e\x85  "):
            continue
        letters = sum(c.isalpha() for c in txt)
        if letters < 0.5 * len(txt):
            continue
        n2 = len(tok(txt, add_special_tokens=False)["input_ids"])
        if MIN_TOK <= n2 <= MAX_TOK:
            out.append(txt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wiki", "web"], required=True)
    ap.add_argument("--n_train", type=int, required=True)
    ap.add_argument("--n_eval", type=int, default=1500, help="passages for val and for test (each)")
    ap.add_argument("--max_per_doc", type=int, default=6)
    ap.add_argument("--out_dir", default="data/corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_docs", type=int, default=0, help="skip the first N documents of the stream (documents already used)")
    ap.add_argument("--data_files", default=None, help="web only: parquet path(s) inside the fineweb-2 repo to stream instead of the default order")
    ap.add_argument("--max_tok", type=int, default=MAX_TOK)
    ap.add_argument("--min_tok", type=int, default=MIN_TOK)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    rng = random.Random(args.seed)
    seen = set()
    files = {s: open(os.path.join(args.out_dir, f"{s}.{args.source}.jsonl"), "w") for s in ("train", "val", "test")}
    counts = {"train": 0, "val": 0, "test": 0}
    targets = {"train": args.n_train, "val": args.n_eval, "test": args.n_eval}
    t0 = time.time()
    ndocs = 0
    for doc in iter_docs(args.source, args.data_files):
        ndocs += 1
        if ndocs <= args.skip_docs:
            continue
        # document-level split: eval docs come first (small), then everything else is train
        split = "val" if counts["val"] < targets["val"] else "test" if counts["test"] < targets["test"] else "train"
        if counts[split] >= targets[split]:
            break
        for txt in passages_from_doc(doc, tok, rng, args.max_per_doc, args.max_tok, args.min_tok):
            if txt in seen:
                continue
            seen.add(txt)
            files[split].write(json.dumps({"text": txt, "src": args.source}, ensure_ascii=False) + "\n")
            counts[split] += 1
        if ndocs % 5000 == 0:
            print(f"{args.source}: docs={ndocs} {counts} {time.time()-t0:.0f}s", flush=True)
    for f in files.values():
        f.close()
    print(f"{args.source}: done docs={ndocs} {counts} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
