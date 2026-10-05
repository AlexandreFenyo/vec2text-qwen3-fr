"""Assemble the v2 training corpora (32 tokens):
  data/corpus_v2  : inverter training = general train[:N_GEN] + domain train; val = general val + domain val;
                    test = general test; test_domain = domain test.
  data/corpus_corr: corrector training = general train[N_GEN:] (never seen by the inverter: realistic hypotheses)
                    + the first N_CORR_SEEN of corpus_v2 train; val = corpus_v2 val.
Embeddings of general passages are reused from data/corpus/*.emb.npy; domain passages are embedded here."""
import argparse, json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.embed_corpus import load_encoder, encode


def read(path):
    return [l for l in open(path, encoding="utf-8").read().split("\n") if l]


def write(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_gen", type=int, default=2600000, help="general train passages used by the inverter")
    ap.add_argument("--n_corr_seen", type=int, default=1600000, help="inverter-seen passages added to the corrector set")
    args = ap.parse_args()
    rng = random.Random(0)

    gen_train = read("data/corpus/train.jsonl")
    gen_emb = np.load("data/corpus/train.emb.npy", mmap_mode="r")
    assert len(gen_train) == gen_emb.shape[0]
    dom = {}
    for split in ("train", "val", "test"):
        lines = []
        for src in ("wiki", "web"):
            p = f"data/domain/{split}.{src}.jsonl"
            if os.path.exists(p):
                lines += read(p)
        rng.shuffle(lines)
        dom[split] = lines
        print("domain", split, len(lines), flush=True)
    enc = load_encoder()
    dom_emb = {s: encode(enc, [json.loads(l)["text"] for l in dom[s]], verbose=False) for s in dom}

    os.makedirs("data/corpus_v2", exist_ok=True)
    os.makedirs("data/corpus_corr", exist_ok=True)
    # --- corpus_v2 (inverter): general[:n_gen] + domain train, shuffled together
    idx_gen = list(range(args.n_gen))
    order = [("g", i) for i in idx_gen] + [("d", i) for i in range(len(dom["train"]))]
    rng.shuffle(order)
    lines = [gen_train[i] if k == "g" else dom["train"][i] for k, i in order]
    write("data/corpus_v2/train.jsonl", lines)
    emb = np.lib.format.open_memmap("data/corpus_v2/train.emb.npy", mode="w+", dtype=np.float16, shape=(len(order), 1024))
    g_rows = np.array([i for k, i in order if k == "g"]); g_pos = np.array([p for p, (k, i) in enumerate(order) if k == "g"])
    d_rows = np.array([i for k, i in order if k == "d"]); d_pos = np.array([p for p, (k, i) in enumerate(order) if k == "d"])
    for s in range(0, len(g_pos), 200000):  # chunked gather from the memmap
        emb[g_pos[s:s+200000]] = gen_emb[np.sort(g_rows[s:s+200000])][np.argsort(np.argsort(g_rows[s:s+200000]))]
    emb[d_pos] = dom_emb["train"][d_rows]
    emb.flush(); del emb
    print("corpus_v2 train", len(order), flush=True)
    val_lines = read("data/corpus/val.jsonl") + dom["val"]
    write("data/corpus_v2/val.jsonl", val_lines)
    np.save("data/corpus_v2/val.emb.npy", np.concatenate([np.load("data/corpus/val.emb.npy"), dom_emb["val"]]))
    write("data/corpus_v2/test.jsonl", read("data/corpus/test.jsonl"))
    np.save("data/corpus_v2/test.emb.npy", np.load("data/corpus/test.emb.npy"))
    write("data/corpus_v2/test_domain.jsonl", dom["test"])
    np.save("data/corpus_v2/test_domain.emb.npy", dom_emb["test"])

    # --- corpus_corr (corrector): fresh general[n_gen:] + first n_corr_seen rows of corpus_v2 train
    fresh = list(range(args.n_gen, len(gen_train)))
    lines = [gen_train[i] for i in fresh] + read("data/corpus_v2/train.jsonl")[:args.n_corr_seen]
    perm = list(range(len(lines))); rng.shuffle(perm)
    write("data/corpus_corr/train.jsonl", [lines[p] for p in perm])
    src_emb = np.concatenate([np.asarray(gen_emb[args.n_gen:]), np.load("data/corpus_v2/train.emb.npy", mmap_mode="r")[:args.n_corr_seen]])
    np.save("data/corpus_corr/train.emb.npy", src_emb[np.array(perm)])
    write("data/corpus_corr/val.jsonl", val_lines)
    np.save("data/corpus_corr/val.emb.npy", np.load("data/corpus_v2/val.emb.npy"))
    print("corpus_corr train", len(lines), "(fresh", len(fresh), ")", flush=True)


if __name__ == "__main__":
    main()
