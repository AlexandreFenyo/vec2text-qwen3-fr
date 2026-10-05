#!/usr/bin/env python
"""Build a random rotation matrix (special orthogonal, det = +1) of the embedding dimension.

Any rotation of R^n decomposes into independent planar rotations in n/2 mutually orthogonal 2-D planes.
This script draws such a rotation: for each of the dim/2 planes of an orthogonal frame, a rotation angle is
drawn uniformly in [0, max_degrees] (up to 360: angles beyond 180 are rotations in the opposite direction of the
plane, i.e. distinct matrices). With --basis random (default) the frame itself is a random orthonormal basis
(Haar-distributed); with --basis canonical the planes are the coordinate planes (axes 0-1, 2-3, ...).
Expected cosine between a vector and its image: sin(D)/D with D in radians (1 at 0, 0 at 180 and 360, -0.21 at 270).

  python make_rotation.py --degrees 30 --out rot30.npy            # random frame, angles in [0, 30°]
  python make_rotation.py --degrees 180 --seed 7 --out rot.npy --tsv rot.tsv
  python make_rotation.py --fixed_angle 90 --seed_hex $(python -c "import hmac,hashlib;print(hmac.new(b'master-key', b'user-123', hashlib.sha256).hexdigest())") --out user123.npy
"""
import argparse, sys
import numpy as np


def rotation_matrix(dim, max_degrees, seed=0, basis="random", fixed_angle=None):
    """seed: int, or a 256-bit hex string (e.g. an HMAC of a master key and a user id) for key derivation.
    fixed_angle: use this angle (degrees) in every plane instead of random angles in [0, max_degrees]."""
    rng = np.random.default_rng(int(seed, 16) if isinstance(seed, str) else seed)
    n_planes = dim // 2
    if fixed_angle is not None:
        angles = np.full(n_planes, np.deg2rad(fixed_angle))
    else:
        angles = np.deg2rad(rng.uniform(0.0, max_degrees, size=n_planes))
    # block-diagonal matrix of 2x2 rotations (plus an untouched axis if dim is odd)
    R = np.eye(dim)
    c, s = np.cos(angles), np.sin(angles)
    for k in range(n_planes):
        i, j = 2 * k, 2 * k + 1
        R[i, i], R[i, j], R[j, i], R[j, j] = c[k], -s[k], s[k], c[k]
    if basis == "canonical":
        return R, np.rad2deg(angles)
    # random orthonormal frame Q (Haar measure via QR of a Gaussian matrix); rotation = Q R Q^T
    A = rng.standard_normal((dim, dim))
    Q, Rq = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(Rq))
    return Q @ R @ Q.T, np.rad2deg(angles)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--degrees", type=float, default=None, help="maximum rotation angle per plane, in degrees (0..360); each plane gets a uniform angle in [0, max]")
    ap.add_argument("--fixed_angle", type=float, default=None, help="same angle in every plane (90 recommended; 0, 180 and 360 are refused: they give +I or -I, i.e. no key)")
    ap.add_argument("--seed_hex", default=None, help="256-bit hex seed for key derivation, e.g. HMAC-SHA256(master key, user id); overrides --seed")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--basis", choices=["random", "canonical"], default="random")
    ap.add_argument("--out", required=True, help="output .npy (float64 matrix dim x dim)")
    ap.add_argument("--tsv", default=None, help="also write the matrix as TSV")
    args = ap.parse_args()
    if args.fixed_angle is None and args.degrees is None:
        sys.exit("give --degrees (random angles) or --fixed_angle")
    if args.fixed_angle is not None and (abs(np.sin(np.deg2rad(args.fixed_angle))) < 1e-9):
        sys.exit("--fixed_angle 0/180/360 gives +I or -I: no secret at all; use 90")
    if args.degrees is not None and not 0 <= args.degrees <= 360:
        sys.exit("--degrees must be between 0 and 360")
    seed = args.seed_hex if args.seed_hex else args.seed
    M, angles = rotation_matrix(args.dim, args.degrees or 0.0, seed, args.basis, args.fixed_angle)
    np.save(args.out, M)
    if args.tsv:
        np.savetxt(args.tsv, M, delimiter="\t", fmt="%.10f")
    orth = np.abs(M @ M.T - np.eye(args.dim)).max()
    rng_desc = f"fixed angle {args.fixed_angle}°" if args.fixed_angle is not None else f"angles in [0, {args.degrees}°]"
    print(f"rotation {args.dim}x{args.dim} saved to {args.out}: {len(angles)} planes, {rng_desc} "
          f"(mean {angles.mean():.2f}°), basis={args.basis}, det={np.linalg.det(M):+.6f}, max|MM^T-I|={orth:.1e}")
    # effect on a typical unit vector: expected cosine between v and Mv
    v = np.random.default_rng(1).standard_normal(args.dim); v /= np.linalg.norm(v)
    print(f"cosine between a random unit vector and its image: {float(v @ (M @ v)):.4f}")


if __name__ == "__main__":
    main()
