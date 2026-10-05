#!/usr/bin/env python
"""Sweep the rotation angle: for each max angle D in a range, build the rotation (same seed, so the per-plane
angles are D x the same uniform draws: the sweep is continuous), rotate the input vectors, and invert them.
Models are loaded once and all rotated vectors are inverted in batches.

  python sweep_rotation.py --vectors /tmp/vecs.tsv --metadata /tmp/meta.tsv --start 0 --stop 180 --step 1 \
      --profile long --steps 5 --beam 4 --out /tmp/sweep.tsv

Output TSV columns: degrees, index, cos_rotation (original vs rotated vector), cosine (reconstruction vs rotated
input), text, then the metadata columns. Add --per_angle_dir DIR to also write one TSV per angle.
"""
import argparse, csv, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v2t.env import quiet_offline
quiet_offline()
from make_rotation import rotation_matrix
from invert import read_vectors, read_metadata, resolve_models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--stop", type=float, default=180)
    ap.add_argument("--step", type=float, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--basis", choices=["random", "canonical"], default="random")
    ap.add_argument("--profile", choices=["short", "long"], default=None)
    ap.add_argument("--inverter", default=None)
    ap.add_argument("--corrector", default=None)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--no_repeat", type=int, default=None)
    ap.add_argument("--out", required=True, help="single TSV with all angles")
    ap.add_argument("--per_angle_dir", default=None, help="also write resultat-<deg>.tsv files here")
    args = ap.parse_args()

    V = read_vectors(args.vectors)
    meta_cols, meta_rows = read_metadata(args.metadata) if args.metadata else (None, None)
    degrees = np.arange(args.start, args.stop + 1e-9, args.step)
    print(f"{len(V)} vector(s) x {len(degrees)} angles", file=sys.stderr)

    # rotated vectors for every angle (the random frame Q is the same for all angles: same seed)
    rows, allvecs = [], []
    t0 = time.time()
    for d in degrees:
        M, _ = rotation_matrix(V.shape[1], float(d), args.seed, args.basis)
        W = V @ M.T
        cos_rot = (V * W).sum(1) / (np.linalg.norm(V, axis=1) * np.linalg.norm(W, axis=1))
        for i in range(len(V)):
            rows.append((float(d), i, float(cos_rot[i])))
            allvecs.append(W[i])
    print(f"rotations built in {time.time()-t0:.1f}s", file=sys.stderr)

    from v2t.infer import Inverter
    inv_path, cor_path, profile = resolve_models(args.profile, args.inverter, args.corrector)
    print(f"models: {inv_path} + {cor_path} (profile {profile})", file=sys.stderr)
    inv = Inverter(inv_path, cor_path if args.steps > 0 else None)
    inv.no_repeat_ngram_size = args.no_repeat if args.no_repeat is not None else (6 if inv.inv.max_tokens > 64 else 0)
    t0 = time.time()
    texts, cos = inv.invert(np.asarray(allvecs, dtype=np.float32), steps=args.steps, beam=args.beam, verbose=True)
    print(f"inverted {len(texts)} vectors in {time.time()-t0:.0f}s", file=sys.stderr)

    header = ["degrees", "index", "cos_rotation", "cosine", "text"] + (meta_cols or [])
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        for (d, i, cr), t, c in zip(rows, texts, cos):
            w.writerow([f"{d:g}", i, f"{cr:.4f}", f"{c:.4f}", t] + (meta_rows[i] if meta_rows else []))
    print(f"wrote {args.out}", file=sys.stderr)
    if args.per_angle_dir:
        os.makedirs(args.per_angle_dir, exist_ok=True)
        by_deg = {}
        for (d, i, cr), t, c in zip(rows, texts, cos):
            by_deg.setdefault(d, []).append([i, f"{c:.4f}", t] + (meta_rows[i] if meta_rows else []))
        for d, rs in by_deg.items():
            with open(os.path.join(args.per_angle_dir, f"resultat-{d:g}.tsv"), "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter="\t", lineterminator="\n")
                w.writerow(["index", "cosine", "text"] + (meta_cols or []))
                w.writerows(rs)
    for (d, i, cr), t, c in zip(rows, texts, cos):
        if i == 0:
            print(f"{d:6g}°  cos_rot={cr:+.3f}  cos={c:.3f}  {t[:90]}")


if __name__ == "__main__":
    main()
