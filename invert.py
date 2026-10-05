#!/usr/bin/env python
"""Reconstruct texts from Qwen3-Embedding-0.6B embeddings.

Input: a vectors file in the TensorFlow Embedding Projector format (https://projector.tensorflow.org/):
tab-separated floats, one 1024-d vector per line, no header. Optionally a metadata TSV (one line per vector,
with a header line only when it has several columns), which is copied to the output.
A raw float32 ".bytes" file (the Projector's binary tensor format) is also accepted with --bytes.

Examples:
  python invert.py --vectors vecs.tsv                        # inversion model only (fast)
  python invert.py --vectors vecs.tsv --steps 5 --beam 4     # + iterative correction (best quality)
  python invert.py --vectors t.bytes --bytes --metadata labels.tsv --out result.tsv
"""
import argparse, csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v2t.env import quiet_offline
quiet_offline()


PROFILES = {  # best measured pairs (the inverter and the corrector are independent models)
    "short": (["inverter_v2", "inverter"], ["corrector_v3", "corrector"]),
    "long": (["inverter128"], ["corrector128_v3", "corrector128"]),
}
HUB_MODELS = {  # published copies of the best pairs (bfloat16) on the Hugging Face Hub
    "short": ("fenyo/vec2text-qwen3-fr-inverter-32", "fenyo/vec2text-qwen3-fr-corrector-32"),
    "long": ("fenyo/vec2text-qwen3-fr-inverter-128", "fenyo/vec2text-qwen3-fr-corrector-128"),
}


def hub_download(repo_id):
    """Download (or reuse from the HF cache) a published model; returns its local directory."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download
    print(f"downloading {repo_id} from the Hugging Face Hub (first use only)...", file=sys.stderr)
    return snapshot_download(repo_id, allow_patterns=["*.json", "*.safetensors", "proj.pt"])


def first_existing(names, hub_fallback):
    for n in names:
        p = os.path.join(HERE, "ckpt", n)
        if os.path.exists(os.path.join(p, "v2t_config.json")):
            return p
    return hub_download(hub_fallback)


def resolve_path(p):
    """A local directory, or a Hub repo id such as 'fenyo/vec2text-qwen3-fr-inverter-32'."""
    if p and not os.path.exists(os.path.join(p, "v2t_config.json")) and "/" in p and not os.path.isabs(p):
        return hub_download(p)
    return p


def resolve_models(profile, inverter, corrector, need_corrector=True):
    if profile is None:
        profile = "long" if os.path.exists(os.path.join(HERE, "ckpt", "inverter128", "v2t_config.json")) else "short"
    inv_names, cor_names = PROFILES[profile]
    hub_inv, hub_cor = HUB_MODELS[profile]
    inverter = resolve_path(inverter) if inverter else first_existing(inv_names, hub_inv)
    if need_corrector:
        corrector = resolve_path(corrector) if corrector else first_existing(cor_names, hub_cor)
    else:
        corrector = corrector or ""
    return inverter, corrector, profile


def read_vectors(path, as_bytes=False, dim=1024):
    if as_bytes:
        return np.fromfile(path, dtype=np.float32).reshape(-1, dim)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.replace(",", "\t").split("\t") if x != ""])
    return np.asarray(rows, dtype=np.float32)


def read_metadata(path):
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    if lines and "\t" in lines[0]:
        return lines[0].split("\t"), [l.split("\t") for l in lines[1:]]
    return ["label"], [[l] for l in lines]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vectors", required=True, help="TSV of tab-separated floats (Projector format) or raw float32 with --bytes")
    ap.add_argument("--bytes", action="store_true", help="vectors file is raw float32 (Projector .bytes)")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--metadata", default=None, help="optional Projector metadata TSV, copied to the output")
    ap.add_argument("--out", default=None, help="output TSV (default: stdout only)")
    ap.add_argument("--steps", type=int, default=3, help="correction steps (0 = inversion model only)")
    ap.add_argument("--beam", type=int, default=4, help="beam width / hypotheses kept per step")
    ap.add_argument("--profile", choices=["short", "long"], default=None,
                    help="preset model pair: 'short' = texts up to 32 tokens (ckpt/inverter_v2 + ckpt/corrector_v3), "
                         "'long' = up to 128 tokens (ckpt/inverter128 + ckpt/corrector128_v3). Default: long if present, else short.")
    ap.add_argument("--inverter", default=None, help="inversion model directory (overrides --profile)")
    ap.add_argument("--corrector", default=None, help="corrector model directory (overrides --profile)")
    ap.add_argument("--bs", type=int, default=None, help="batch size (default: chosen from text length and beam)")
    ap.add_argument("--no_repeat", type=int, default=None, help="block repeated n-grams of this size when decoding (default: 6 for 128-token models, off otherwise)")
    ap.add_argument("--max_new_tokens", type=int, default=None, help="max generated tokens (default: model's training length + 8)")
    ap.add_argument("--min_new_tokens", type=int, default=0, help="forbid the end-of-text token before this many tokens: forces longer outputs than the model was trained for (fidelity drops beyond its training length)")
    args = ap.parse_args()

    E = read_vectors(args.vectors, args.bytes, args.dim)
    if E.ndim != 2 or E.shape[1] != 1024:
        sys.exit(f"expected 1024-d vectors (Qwen3-Embedding-0.6B), got shape {E.shape}")
    norms = np.linalg.norm(E, axis=1)
    if (np.abs(norms - 1) > 0.05).any():
        print(f"note: vectors are not unit-norm (mean norm {norms.mean():.3f}); normalizing", file=sys.stderr)
    meta_cols, meta_rows = (None, None)
    if args.metadata:
        meta_cols, meta_rows = read_metadata(args.metadata)
        if len(meta_rows) != len(E):
            sys.exit(f"metadata has {len(meta_rows)} rows but there are {len(E)} vectors")

    from v2t.infer import Inverter
    args.inverter, args.corrector, profile = resolve_models(args.profile, args.inverter, args.corrector, need_corrector=args.steps > 0)
    print(f"models: {args.inverter} + {args.corrector} (profile {profile})", file=sys.stderr)
    corr = args.corrector if (args.steps > 0 and os.path.exists(args.corrector)) else None
    if args.steps > 0 and corr is None:
        print(f"note: corrector not found at {args.corrector}, using the inversion model only", file=sys.stderr)
    inv = Inverter(args.inverter, corr)
    inv.no_repeat_ngram_size = args.no_repeat if args.no_repeat is not None else (6 if inv.inv.max_tokens > 64 else 0)
    inv.min_new_tokens = args.min_new_tokens
    if args.min_new_tokens and (args.max_new_tokens or 0) <= args.min_new_tokens:
        args.max_new_tokens = args.min_new_tokens + 64
    print(f"inverting {len(E)} vectors (steps={args.steps if corr else 0}, beam={args.beam})...", file=sys.stderr)
    texts, cos = inv.invert(E, steps=args.steps if corr else 0, beam=args.beam, bs=args.bs, verbose=len(E) > 16,
                            max_new_tokens=args.max_new_tokens)

    header = ["index", "cosine", "text"] + (meta_cols or [])
    rows = [[i, f"{c:.4f}", t] + (meta_rows[i] if meta_rows else []) for i, (t, c) in enumerate(zip(texts, cos))]
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)
        print(f"wrote {args.out}", file=sys.stderr)
    for r in rows:
        print(f"[{r[0]}] cos={r[1]}  {r[2]}")


if __name__ == "__main__":
    main()
