"""Harmonic-extension skinning weights from the elastic Hessian (no fitted weights).

Idea (see docs/mech_qem.tex, Schur section): a skinning map is a prolongation
S that reconstructs the fine mesh from a coarse set of "handle" vertices,
d_fine = S d_coarse. The energy-OPTIMAL S is the harmonic extension of the
stiffness matrix K (the elastic-energy Hessian): partition DOFs into retained
(handles, r) and eliminated (e), then the fine DOFs that minimize 1/2 d^T K d
given the handle motion are

        d_e = -K_ee^{-1} K_er d_r          =>   S = [ I ; -K_ee^{-1} K_er ].

The rows of -K_ee^{-1}K_er ARE the skinning weights, read off K in closed form
-- no pre-simulated frames, no post-hoc weight optimization (cf. Zheng et al.
2024, who fit LBS weights to PBD frames with ARAP losses). Because K annihilates
the rigid modes, S reproduces rigid motion exactly (partition of unity).

This script: assemble the (corrected) fine K, pick coarse handles, emit S, and
compare reconstruction error against a geometric inverse-distance kNN baseline
(the naive "bind to nearby handles" skin), on smooth test deformations.

Run:
  python prototype/harmonic_skinning.py                 # skirt, 128 handles
  python prototype/harmonic_skinning.py --handles 200
  python prototype/harmonic_skinning.py --smoke         # headless checks
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech_qem as mq  # noqa: E402

from kms.mesh import load_obj  # noqa: E402


DEFAULT_MESH = "data/9423122485_cleaned.obj"


# --------------------------------------------------------------------------- #
#  Assemble the fine elastic Hessian K (sparse), using the CORRECTED elements
# --------------------------------------------------------------------------- #
def assemble_K_sparse(model: mq.MechModel) -> sparse.csc_matrix:
    """Global K = sum of membrane (9x9) + bending hinge (12x12) element blocks."""
    mesh = model.mesh
    v, f = mesh.vertices, mesh.faces
    ndof = 3 * mesh.n_verts
    rows, cols, vals = [], [], []

    def scatter(Ke, idx):
        dofs = np.array([3 * i + c for i in idx for c in range(3)])
        nb = len(dofs)
        rows.append(np.repeat(dofs, nb))
        cols.append(np.tile(dofs, nb))
        vals.append(Ke.ravel())

    for fi, tri in enumerate(f):
        Ke = model.face_Ke[fi]
        if Ke is not None:
            scatter(Ke, [int(tri[0]), int(tri[1]), int(tri[2])])
    for h in model.hinges:
        a, b, c, d = int(h[0]), int(h[1]), int(h[2]), int(h[3])
        Ke = mq.hinge_bend_Ke(v[a], v[b], v[c], v[d], model.kb)
        if Ke is not None:
            scatter(Ke, [a, b, c, d])

    K = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ndof, ndof),
    ).tocsc()
    return 0.5 * (K + K.T)


# --------------------------------------------------------------------------- #
#  Handles, harmonic prolongation, geometric baseline
# --------------------------------------------------------------------------- #
def farthest_point_sampling(points: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Well-spread subset of vertex indices (the coarse handles)."""
    idx = [seed % len(points)]
    d = np.linalg.norm(points - points[idx[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    return np.array(sorted(set(idx)), dtype=np.int64)


def harmonic_prolongation(K, R, E, reg=1e-8):
    """X = -K_ee^{-1} K_er  (dense, 3|E| x 3|R|): the energy-minimizing skinning
    weights mapping handle displacements to eliminated-vertex displacements."""
    r_dofs = (R[:, None] * 3 + np.arange(3)).ravel()
    e_dofs = (E[:, None] * 3 + np.arange(3)).ravel()
    Kcsr = K.tocsr()
    Kee = Kcsr[e_dofs][:, e_dofs].tocsc()
    Ker = Kcsr[e_dofs][:, r_dofs].tocsc()
    Kee = Kee + reg * sparse.eye(Kee.shape[0], format="csc")  # guard floppy modes
    lu = splu(Kee)
    X = lu.solve(-Ker.toarray())
    return X, r_dofs, e_dofs


def harmonic_prolongation_matrix(K, survivors, n_verts, reg=1e-8):
    """Assemble the FULL coarse->fine prolongation S (sparse, 3 n_verts x 3 n_coarse)
    from the GLOBAL harmonic solve: handle rows are identity, eliminated rows are
    the -K_ee^{-1} K_er block.  d_fine = S d_coarse.

    survivors[k] is the fine index of coarse handle k, so column k of S is handle k
    -- the SAME indexing simplify_mechanics' accumulated prolongation uses, so the
    global and step-by-step maps are directly comparable / interchangeable."""
    R = np.asarray(survivors, dtype=np.int64)
    E = np.array(sorted(set(range(n_verts)) - set(R.tolist())), dtype=np.int64)
    X, r_dofs, e_dofs = harmonic_prolongation(K, R, E, reg=reg)   # X: 3|E| x 3|R|
    nc3 = 3 * len(R)

    # eliminated rows: scatter dense X (its columns are already in coarse/handle order)
    ei = np.repeat(e_dofs, nc3)
    ej = np.tile(np.arange(nc3), len(e_dofs))
    ev = X.ravel()
    # handle rows: identity block, fine dof 3*R[k]+c <-> coarse dof 3*k+c
    hi = r_dofs
    hj = (np.arange(len(R))[:, None] * 3 + np.arange(3)).ravel()
    hv = np.ones(len(hi))

    return sparse.csr_matrix(
        (np.concatenate([ev, hv]), (np.concatenate([ei, hi]), np.concatenate([ej, hj]))),
        shape=(3 * n_verts, nc3))


def geometric_weights(verts, R, E, k=4, eps=1e-9):
    """Inverse-distance kNN skinning to the handles (scalar weights, the naive
    geometric LBS baseline): W is (|E| x |R|), rows nonneg, summing to 1."""
    tree = cKDTree(verts[R])
    dist, nn = tree.query(verts[E], k=k)
    dist = np.atleast_2d(dist)
    nn = np.atleast_2d(nn)
    w = 1.0 / (dist + eps)
    w /= w.sum(axis=1, keepdims=True)
    rows = np.repeat(np.arange(len(E)), k)
    W = sparse.coo_matrix((w.ravel(), (rows, nn.ravel())), shape=(len(E), len(R))).tocsr()
    return W


# --------------------------------------------------------------------------- #
#  Test deformations + reconstruction error
# --------------------------------------------------------------------------- #
def _rand_unit(rng):
    x = rng.standard_normal(3)
    return x / np.linalg.norm(x)


def rigid_basis(verts):
    """Orthonormal basis (3n x 6) of the rigid modes: 3 translations + 3 rotations."""
    n = len(verts)
    c = verts.mean(0)
    cols = []
    for k in range(3):
        t = np.zeros((n, 3)); t[:, k] = 1.0; cols.append(t.ravel())
    for k in range(3):
        w = np.zeros(3); w[k] = 1.0
        cols.append(np.cross(w[None, :], verts - c).ravel())
    Q, _ = np.linalg.qr(np.column_stack(cols))
    return Q


def elastic_fields(K, verts, ntrials=15, seed=1):
    """ELASTIC test deformations: d = (K + eps I)^{-1} f for random f, with the
    rigid part projected out.  These are soft-mode-dominated (bend easily, resist
    stretch) -- the deformations the structure actually undergoes, and the regime
    skinning targets (the proxy is elastically simulated)."""
    n = K.shape[0]
    eps = 1e-6 * K.diagonal().mean()
    lu = splu((K + eps * sparse.eye(n, format="csc")).tocsc())
    Q = rigid_basis(verts)
    rng = np.random.default_rng(seed)
    fields = []
    for _ in range(ntrials):
        d = lu.solve(rng.standard_normal(n))
        d = d - Q @ (Q.T @ d)                       # remove rigid component
        d = d.reshape(-1, 3)
        fields.append(d / (np.linalg.norm(d) + 1e-30))
    return fields


def sinusoidal_fields(verts, diag, seed=0):
    """ARBITRARY smooth fields d_i = dir * sin(omega . X_i + phi) -- NOT elastic
    deformations; included as an honest stress test (a generic interpolator, not
    energy-minimization, is what these reward)."""
    rng = np.random.default_rng(seed)
    fields = []
    for f in (1.0, 2.0, 4.0):
        for _ in range(5):
            omega = (2 * np.pi * f / diag) * _rand_unit(rng)
            d = np.sin(verts @ omega + rng.uniform(0, 2 * np.pi))[:, None] * _rand_unit(rng)[None, :]
            fields.append((f, d))
    return fields


def rigid_field(verts, seed=0):
    """A rigid motion (translation + small rotation) -- harmonic must reproduce
    it exactly; geometric LBS reproduces translation but only approximates rotation."""
    rng = np.random.default_rng(seed)
    c = verts.mean(0)
    w = 0.3 * _rand_unit(rng)                  # small rotation vector
    t = _rand_unit(rng)
    return np.cross(w[None, :], verts - c) + t[None, :]


def reconstruct_errors(d_true, R, E, X, r_dofs, e_dofs, W):
    """Relative L2 reconstruction error over the eliminated vertices, for the
    harmonic and geometric skins (retained vertices are exact for both)."""
    d_true_E = d_true[E]
    denom = np.linalg.norm(d_true_E) + 1e-30

    d_r_flat = d_true.ravel()[r_dofs]
    d_e_harm = (X @ d_r_flat).reshape(len(E), 3)
    err_h = np.linalg.norm(d_e_harm - d_true_E) / denom

    d_e_geom = W @ d_true[R]
    err_g = np.linalg.norm(d_e_geom - d_true_E) / denom
    return err_h, err_g


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=DEFAULT_MESH)
    p.add_argument("--handles", type=int, default=128, help="number of coarse handles")
    p.add_argument("--thickness", type=float, default=0.05,
                   help="shell thickness (bending conditions the out-of-plane solve)")
    p.add_argument("--knn", type=int, default=4, help="kNN for the geometric baseline")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    print("=== Harmonic-extension skinning weights ===")
    mesh = load_obj(args.mesh)
    verts = mesh.vertices
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    print(f"  mesh: {mesh.n_verts} verts, {mesh.n_faces} faces; handles: {args.handles}")

    t0 = time.time()
    model = mq.build_model(mesh, thickness=args.thickness)
    K = assemble_K_sparse(model)
    print(f"  assembled K ({K.shape[0]} dof, nnz={K.nnz}) in {time.time()-t0:.2f}s")

    R = farthest_point_sampling(verts, args.handles)
    E = np.array(sorted(set(range(mesh.n_verts)) - set(R.tolist())), dtype=np.int64)

    t0 = time.time()
    X, r_dofs, e_dofs = harmonic_prolongation(K, R, E)
    print(f"  harmonic prolongation S = -K_ee^-1 K_er  ({X.shape}) in {time.time()-t0:.2f}s")
    W = geometric_weights(verts, R, E, k=args.knn)

    # --- partition of unity: harmonic must reproduce rigid translation exactly ---
    t = np.array([1.0, -0.5, 0.3])
    d_r_trans = np.tile(t, len(R))
    d_e_trans = (X @ d_r_trans).reshape(len(E), 3)
    pou_err = np.abs(d_e_trans - t).max()
    print(f"\n  [check] harmonic translation reproduction (partition of unity): "
          f"max err {pou_err:.2e}")

    # --- rigid motion: harmonic exact (it reproduces rotation; translation-only
    #     geometric LBS cannot) ---
    d_rig = rigid_field(verts)
    eh_rig, eg_rig = reconstruct_errors(d_rig, R, E, X, r_dofs, e_dofs, W)
    print(f"  [rigid motion]      harmonic={eh_rig:.2e}   geometric={eg_rig:.2e}")

    def report(name, dfields):
        hs = [reconstruct_errors(d, R, E, X, r_dofs, e_dofs, W) for d in dfields]
        hs = np.array(hs)
        mh, mg = hs[:, 0].mean(), hs[:, 1].mean()
        print(f"  [{name}]  harmonic={mh:.3e}   geometric={mg:.3e}   "
              f"(geom/harm = {mg/mh:.2f}x)")
        return mh, mg

    print("\n  mean relative L2 reconstruction error over eliminated vertices:")
    mh_el, mg_el = report("elastic deformations ", elastic_fields(K, verts))
    mh_si, mg_si = report("arbitrary sinusoids  ", [d for _, d in sinusoidal_fields(verts, diag)])

    print("\n  Reading it: on ELASTIC deformations (the skinning regime -- the proxy")
    print("  is elastically simulated) the energy-minimizing harmonic skin wins; on")
    print("  ARBITRARY smooth fields a generic interpolator (geometric IDW) can match")
    print("  or beat it, because harmonic is optimal in the ELASTIC-ENERGY norm, not")
    print("  as a general-purpose smoother. Harmonic also reproduces rigid rotation")
    print("  exactly (full-matrix S), which translation-only LBS structurally cannot.")

    if args.smoke:
        print("\n--- Smoke checks ---")
        assert pou_err < 1e-4, f"harmonic not partition-of-unity: {pou_err:.2e}"
        assert eh_rig < 1e-4, f"harmonic did not reproduce rigid motion: {eh_rig:.2e}"
        assert eh_rig < eg_rig, "harmonic should beat geometric on rigid motion"
        assert mh_el < mg_el, f"harmonic ({mh_el:.2e}) should beat geometric ({mg_el:.2e}) on elastic"
        print("  partition-of-unity, rigid exactness, harmonic<geometric (elastic): all hold")
        print("\nSMOKE: PASS")


if __name__ == "__main__":
    main()
