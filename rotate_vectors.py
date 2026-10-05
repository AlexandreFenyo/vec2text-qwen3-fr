#!/usr/bin/env python
"""Apply a rotation matrix (from make_rotation.py) to vectors stored in the Projector TSV format
(one vector per line, tab-separated floats). Output has the same format.

  python rotate_vectors.py --matrix rot30.npy --vectors vecs.tsv --out vecs_rot.tsv
  python rotate_vectors.py --matrix rot30.npy --vectors vecs_rot.tsv --out back.tsv --inverse
"""
import argparse, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", required=True, help=".npy or .tsv rotation matrix")
    ap.add_argument("--vectors", required=True, help="input TSV (one vector per line)")
    ap.add_argument("--out", required=True, help="output TSV")
    ap.add_argument("--inverse", action="store_true", help="apply the inverse rotation (M^T)")
    args = ap.parse_args()
    M = np.load(args.matrix) if args.matrix.endswith(".npy") else np.loadtxt(args.matrix, delimiter="\t")
    V = np.loadtxt(args.vectors, delimiter="\t", ndmin=2, dtype=np.float64)
    if V.shape[1] != M.shape[0]:
        sys.exit(f"dimension mismatch: vectors are {V.shape[1]}-d, matrix is {M.shape[0]}x{M.shape[1]}")
    if args.inverse:
        M = M.T
    W = V @ M.T  # row vectors: w = M v
    np.savetxt(args.out, W, delimiter="\t", fmt="%.6f")
    cos = (V * W).sum(1) / (np.linalg.norm(V, axis=1) * np.linalg.norm(W, axis=1))
    print(f"{len(V)} vector(s) rotated -> {args.out}; cosine(original, rotated): mean {cos.mean():.4f}, min {cos.min():.4f}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
