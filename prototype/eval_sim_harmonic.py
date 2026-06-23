"""Design-B end to end: PBD-simulate the coarse proxy, then drive the fine "visual"
mesh through the GLOBAL harmonic skinning map  S = -K_ee^{-1} K_er.

This is the prototype-path analogue of scripts/eval_sim.py.  scripts/eval_sim.py
uses the src/kms simplifier + its fitted/scalar LBS weights; here we instead use the
prototype skinning stack:

  1. get_lbs (viz_lbs)            -- decimate the fine mesh into a coarse proxy and
                                     return survivors + the optimized coarse rest.
  2. harmonic_prolongation_matrix -- the ONE-SHOT global harmonic solve on the fine
                                     elastic Hessian K with the survivors as handles,
                                     S = [ I ; -K_ee^{-1} K_er ]   (3n_fine x 3n_coarse).
  3. run_sim (kms.eval)           -- PBD-simulate the coarse proxy (dancing_sway).
  4. d_fine = S d_coarse          -- propagate the proxy's per-frame displacement to
                                     the fine mesh.  --deform selects HOW:
       full   -- the FULL 3x3-BLOCK map  d_fine[i] = sum_k S_block[i,k] d_coarse[k].
                 The off-diagonal block entries couple x/y/z, so harmonic reproduces
                 ROTATION (the operator viz_lbs propagates).
       scalar -- classic LINEAR BLEND SKINNING: collapse each 3x3 block to one scalar
                 weight  w[i,k] = (1/3) tr(S_block[i,k])  (prolongation_scalar_weights;
                 rows sum to 1), then  d_fine[i] = sum_k w[i,k] d_coarse[k]  applied
                 IDENTICALLY to x/y/z.  Translation-only -- no rotation coupling.  This
                 is the form proxy-asset-gen uses (sparse kNN weights it OPTIMIZES on
                 sim frames; here the weights are read off the operator instead).
       corotational -- KEEP the operator weights w but add a per-handle finite rotation
                 R_k (estimated from the coarse 1-ring, ARAP polar decomposition):
                   x_fine[i] = sum_k w[i,k] [ X_k + R_k (F_i - C_k) ].
                 Reproduces finite rotation EXACTLY and carries the drift offset rigidly,
                 so it fixes BOTH the linear block's rotation-as-stretch inflation AND the
                 id-handle drift scramble -- operator-faithful weights, visually faithful
                 motion (they are NOT mutually exclusive).

The fine reconstruction is  V_fine(t) = V_fine_rest + (S or W)(X_coarse(t) - V_coarse_rest)
for full/scalar; co-rotational uses the position form above.  A finite-rotation fidelity
diagnostic (synthetic rigid proxy tilt) is printed so the full/scalar/corot gap is explicit.

NOTE on rest mismatch (drift): the optimized coarse rest V_coarse_rest drifts from the fine
anchors mesh.vertices[survivors].  full/scalar's handle rows are identity (assume coarse
handle k sits ON fine vertex survivors[k]) -> translation transfers exactly but rotation
errs ~ drift x angle.  corotational carries the offset (F_i - C_k) under R_k, removing it.

NOTE on handle KINKS: the exact harmonic map pins each handle fine vertex to its proxy
vertex (identity rows) -> a CUSP at the handle when the proxy is faceted/drifted (a hard
point constraint pokes off the smooth trend).  Two levers, with a printed handle-roughness
ratio (Laplacian at handles / mesh avg, 1.0 = no kink) to tune them:
  --relax R  (THE fix): soft handles -- APPROXIMATE the proxy at handles instead of pinning
             it (regularized energy 1/2 d^T K d + lambda/2|d_h-d_coarse|^2, lambda=med(diagK)/R).
             Relaxes the cusp into a smooth bump.  Measured: corot 1.77->1.17 at R=20, no
             overshoot (unlike pure biharmonic), small handle slip.  Try 5-50.
  --smooth B (interior only): blend more bending into the operator (thickness*sqrt(B)).
             Smooths BETWEEN handles but barely touches the cusp (can't soften a hard
             constraint) and can make pinned handles poke more -- use --relax for kinks.

Run:
  python prototype/eval_sim_harmonic.py                          # full | scalar | corotational
  python prototype/eval_sim_harmonic.py --deform corotational    # only the corrected skin
  python prototype/eval_sim_harmonic.py --relax 20               # soft handles: smooth the kinks
  python prototype/eval_sim_harmonic.py --handles 200 --frames 240
  python prototype/eval_sim_harmonic.py --screenshot out/eval_harmonic.png
  python prototype/eval_sim_harmonic.py --smoke --frames 12 --settle 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harmonic_skinning as hs  # noqa: E402
import mech_qem as mq  # noqa: E402
import simplify_mechanics as sm  # noqa: E402
from viz_lbs import get_lbs, DEFAULT_MESH, PROLONG_THICKNESS  # noqa: E402

from kms.mesh import load_obj  # noqa: E402
from kms.eval.sim_runner import run_sim  # noqa: E402
from kms.eval.lbs_reconstruct import lbs_reconstruct  # noqa: E402
from kms.eval.scenarios import DancingSway  # noqa: E402


def find_pinned_vertices(V: np.ndarray, top_fraction: float = 0.1) -> np.ndarray:
    """Vertices in the top `top_fraction` by y (pin the top of the garment)."""
    y = V[:, 1]
    threshold = y.max() - top_fraction * (y.max() - y.min())
    return np.where(y >= threshold)[0].astype(np.int64)


def build_global_S(mesh, survivors, stem, thickness, bending_weight, reg=1e-10,
                   cache_dir="out"):
    """The one-shot global harmonic prolongation  S = [I; -K_ee^{-1} K_er]
    (3 n_fine x 3 n_coarse), cached.  Depends only on the fine K and the survivors
    (NOT on the coarse rest)."""
    base = os.path.join(cache_dir,
                        f"_Sglobal_{stem}_{len(survivors)}_pt{thickness:g}_b{bending_weight:g}")
    npz = base + ".npz"
    if os.path.exists(npz):
        print(f"  global S: loaded cached {npz}")
        return sparse.load_npz(npz)
    t0 = time.time()
    K = hs.assemble_K_sparse(mq.build_model(mesh, thickness=thickness))
    S = hs.harmonic_prolongation_matrix(K, survivors, mesh.n_verts, reg=reg)
    os.makedirs(cache_dir, exist_ok=True)
    sparse.save_npz(npz, S.tocsr())
    print(f"  global S = -K_ee^-1 K_er  {S.shape} ({S.nnz} nnz) in {time.time()-t0:.1f}s")
    return S.tocsr()


def build_soft_S(mesh, survivors, thickness, relax, reg=1e-9):
    """Soft-handle (approximating) prolongation.  The exact harmonic map pins each handle
    fine vertex to its proxy vertex (identity rows) -> a CUSP at the handle when the proxy
    is faceted/drifted/off-trend; no interior operator can remove a hard point constraint.
    Instead minimize over ALL fine dofs
        1/2 d^T K d  +  (lambda/2) |d_handles - d_coarse|^2
    ->  S = lambda (K + lambda D_h)^{-1} P_h^T   (3n x 3nc),  D_h = handle-dof selector.
    Handles are now APPROXIMATED (the surface no longer interpolates the proxy exactly), so
    the cusp relaxes into a smooth bump.  `relax` is the dimensionless softness (lambda =
    median(diag K)/relax): larger -> softer -> smoother (and more handle slip).  Still
    reproduces translation/rotation (rigid is the 0-energy, 0-penalty minimizer)."""
    from scipy.sparse.linalg import splu
    n = mesh.n_verts
    K = hs.assemble_K_sparse(mq.build_model(mesh, thickness=thickness)).tocsc()
    kscale = float(np.median(K.diagonal()))
    lam = kscale / relax
    r = (np.asarray(survivors, np.int64)[:, None] * 3 + np.arange(3)).ravel()   # handle dofs
    nc = len(survivors)
    dvec = np.zeros(3 * n); dvec[r] = lam                       # lambda * D_h (diagonal)
    A = (K + sparse.diags(dvec) + reg * kscale * sparse.eye(3 * n)).tocsc()
    Pht = sparse.csc_matrix((np.full(3 * nc, lam), (r, np.arange(3 * nc))),
                            shape=(3 * n, 3 * nc))              # lambda * P_h^T
    return sparse.csr_matrix(splu(A).solve(Pht.toarray()))


def reconstruct_full(S, V_fine_rest, V_coarse_rest, X_coarse):
    """Drive the fine mesh from the proxy sim through the FULL block map S:
        V_fine(t) = V_fine_rest + S (X_coarse(t) - V_coarse_rest).
    S is 3n_fine x 3n_coarse, applied to the flattened 3n_coarse displacement."""
    T, n_coarse, _ = X_coarse.shape
    n_fine = V_fine_rest.shape[0]
    V_fine = np.zeros((T, n_fine, 3), dtype=np.float64)
    for t in range(T):
        disp = (X_coarse[t] - V_coarse_rest).reshape(3 * n_coarse)
        V_fine[t] = V_fine_rest + (S @ disp).reshape(n_fine, 3)
    return V_fine


# --------------------------------------------------------------------------- #
#  Co-rotational LBS: keep the operator weights, add per-handle finite rotation
# --------------------------------------------------------------------------- #
def ring_edges(faces, n_coarse):
    """Directed 1-ring edges (src=k, dst=neighbor) of the coarse mesh, for per-handle
    rotation estimation."""
    nb = [set() for _ in range(n_coarse)]
    for a, b, c in faces:
        nb[a].update((b, c)); nb[b].update((a, c)); nb[c].update((a, b))
    src, dst = [], []
    for k, s in enumerate(nb):
        for j in s:
            src.append(k); dst.append(j)
    return np.asarray(src, np.int64), np.asarray(dst, np.int64)


def _rotation_angle(R):
    """Rotation angle (degrees) of matrices R (..., 3, 3), from the trace."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def estimate_handle_rotations(C_rest, src, dst, X_coarse):
    """Per-handle finite rotation R_k(t) from the coarse 1-ring deformation gradient
    (ARAP / polar decomposition):
        R_k = argmin_R  sum_{j in ring(k)} || (X_j - X_k) - R (C_j - C_k) ||^2 ,
    solved by SVD of the cross-covariance M_k = sum_j (X_j-X_k)(C_j-C_k)^T (det-corrected
    to a proper rotation).  Returns R_seq (T, n_coarse, 3, 3); rank-deficient handles
    (colinear / isolated 1-ring) fall back to identity.

    A curved shell's 1-ring is non-coplanar, so M_k is full rank and the out-of-plane
    rotation is well posed; for a perfectly flat patch one would augment M_k with the
    vertex normal pair (n0 -> n) -- not needed here."""
    T = X_coarse.shape[0]
    nc = C_rest.shape[0]
    e0 = C_rest[dst] - C_rest[src]                       # (E,3) rest edges (time-constant)
    R_seq = np.tile(np.eye(3), (T, nc, 1, 1))
    for t in range(T):
        e = X_coarse[t][dst] - X_coarse[t][src]          # (E,3) current edges
        outer = e[:, :, None] * e0[:, None, :]           # (E,3,3)  (X_j-X_k)(C_j-C_k)^T
        M = np.zeros((nc, 3, 3))
        np.add.at(M, src, outer)                         # cross-covariance per handle
        U, sigma, Vt = np.linalg.svd(M)
        d = np.sign(np.linalg.det(U @ Vt))               # det-correct -> proper rotation
        U[:, :, 2] *= d[:, None]
        R = U @ Vt
        bad = sigma[:, 1] < 1e-9 * (sigma[:, 0] + 1e-30)  # rank-deficient -> identity
        R[bad] = np.eye(3)
        R_seq[t] = R
    return R_seq


def reconstruct_corotational(W, C_rest, F_rest, X_coarse, R_seq):
    """Co-rotational LBS with the SAME operator weights W (scalar):
        x_fine[i](t) = sum_k w[i,k] [ X_k(t) + R_k(t) (F_i - C_k) ] .
    Reproduces finite rotation EXACTLY (a global rotation -> X_k=R C_k, R_k=R gives
    x_i = R F_i) and carries the drift offset (F_i - C_k) rigidly, so neither the
    rotation-as-stretch artifact of the linear block nor the drift scramble appears.
    Returns V_seq (T, n_fine, 3)."""
    Wd = np.asarray(W)                                   # (n_fine, n_coarse) dense scalar
    T, nc, _, _ = R_seq.shape
    n_fine = F_rest.shape[0]
    V = np.zeros((T, n_fine, 3))
    for t in range(T):
        R = R_seq[t]                                     # (nc,3,3)
        Rbar = (Wd @ R.reshape(nc, 9)).reshape(n_fine, 3, 3)          # blended rotation / fine vertex
        RC = np.einsum("kij,kj->ki", R, C_rest)          # R_k C_k  (nc,3)
        anchor = Wd @ (X_coarse[t] - RC)                 # sum_k w (X_k - R_k C_k)
        V[t] = anchor + np.einsum("nij,nj->ni", Rbar, F_rest)
    return V


def finite_rotation_fidelity(S, W, coarse, F_rest, angle_deg=30.0):
    """Diagnostic: rotate the proxy RIGIDLY by angle_deg about z (a swing-like tilt) and
    measure how well each drive reproduces the corresponding finite fine-mesh rotation
    (relative L2).  Co-rotational ~ exact; the linear block inflates (rotation-as-strain);
    scalar translation-only LBS cannot rotate at all.  Returns (errors, recovered_deg)."""
    th = np.radians(angle_deg)
    Rg = np.array([[np.cos(th), -np.sin(th), 0.0],
                   [np.sin(th), np.cos(th), 0.0],
                   [0.0, 0.0, 1.0]])
    c = coarse.vertices.mean(0)
    Xrot = (coarse.vertices - c) @ Rg.T + c              # rigidly rotated proxy (1 frame)
    gt = (F_rest - c) @ Rg.T + c                         # ground-truth fine rotation
    denom = np.linalg.norm(gt - F_rest) + 1e-30
    Xseq = Xrot[None]
    out = {}
    Vf = (S @ (Xrot - coarse.vertices).reshape(-1)).reshape(-1, 3) + F_rest
    out["full 3x3"] = float(np.linalg.norm(Vf - gt) / denom)
    out["scalar LBS"] = float(np.linalg.norm((F_rest + W @ (Xrot - coarse.vertices)) - gt) / denom)
    src, dst = ring_edges(coarse.faces, coarse.vertices.shape[0])
    R_seq = estimate_handle_rotations(coarse.vertices, src, dst, Xseq)
    Vc = reconstruct_corotational(W, coarse.vertices, F_rest, Xseq, R_seq)
    out["corotational"] = float(np.linalg.norm(Vc[0] - gt) / denom)
    return out, float(_rotation_angle(R_seq).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=DEFAULT_MESH)
    p.add_argument("--handles", type=int, default=128, help="target coarse vertex count")
    p.add_argument("--bending-weight", type=float, default=1.0,
                   help="decimation bending weight (must match the cached proxy)")
    p.add_argument("--thickness", type=float, default=PROLONG_THICKNESS,
                   help="shell thickness for the harmonic map's fine K")
    p.add_argument("--smooth", type=float, default=1.0,
                   help="bending multiplier for the PROLONGATION operator (built at "
                        "thickness*sqrt(smooth)). Smooths the INTERIOR between handles; "
                        "weak on the handle cusp itself (that is a hard constraint) -- use "
                        "--relax for that.")
    p.add_argument("--relax", type=float, default=0.0,
                   help="soft-handle relaxation (0 = exact interpolation, the cusp). >0 "
                        "approximates the proxy at handles instead of pinning it (lambda = "
                        "median(diag K)/relax), relaxing the kink into a smooth bump; larger "
                        "= smoother (and more handle slip). Try 5-50.")
    p.add_argument("--deform", choices=["full", "scalar", "corotational", "both", "all"],
                   default="all",
                   help="how to drive the fine mesh: 'full' 3x3-block S; 'scalar' LBS "
                        "weights w=(1/3)tr(S_block) (proxy-asset-gen style); 'corotational' "
                        "LBS keeping those weights but adding per-handle finite rotation; "
                        "'both' (full+scalar) or 'all' (default: + corotational)")
    p.add_argument("--rot-angle", type=float, default=30.0,
                   help="angle for the finite-rotation fidelity diagnostic")
    # scenario / PBD (matched to scripts/eval_sim.py defaults)
    p.add_argument("--frames", type=int, default=240, help="logged simulation frames")
    p.add_argument("--settle", type=int, default=120, help="un-logged settle frames")
    p.add_argument("--cycles", type=float, default=2.0)
    p.add_argument("--amplitude", type=float, default=0.3, help="sway amplitude (frac of bbox)")
    p.add_argument("--sharpness", type=float, default=1.5)
    p.add_argument("--pin-fraction", type=float, default=0.1)
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--k-stretch", type=float, default=0.99)
    p.add_argument("--k-bend", type=float, default=0.1)
    p.add_argument("--k-damp", type=float, default=0.05)
    p.add_argument("--friction", type=float, default=0.4)
    p.add_argument("--restitution", type=float, default=0.0)
    p.add_argument("--contact-skin", type=float, default=0.025)
    p.add_argument("--screenshot", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    print("=== PBD proxy sim -> fine mesh via global harmonic skin  S = -K_ee^-1 K_er ===")
    mesh = load_obj(args.mesh)
    stem = Path(args.mesh).stem
    print(f"  fine mesh: {mesh.n_verts}v, {mesh.n_faces}f; target {args.handles} handles, "
          f"frames {args.frames}, scenario dancing_sway")

    # --- coarse proxy + the global harmonic map ---
    survivors, coarse, _maps = get_lbs(mesh, args.handles, stem,
                                       bending_weight=args.bending_weight)
    prolong_thickness = args.thickness * np.sqrt(args.smooth)   # bending/membrane ~ smooth
    if args.relax > 0:
        t0 = time.time()
        S = build_soft_S(mesh, survivors, prolong_thickness, args.relax)
        print(f"  soft-handle prolongation: relax={args.relax:g} "
              f"(approximating, not interpolating) in {time.time()-t0:.1f}s")
    else:
        S = build_global_S(mesh, survivors, stem, prolong_thickness, args.bending_weight)
    n_fine, n_coarse = mesh.n_verts, len(survivors)
    if args.smooth != 1.0:
        print(f"  prolongation bending x{args.smooth:g} (effective thickness "
              f"{prolong_thickness:g}) -> smoother interior")

    drift = np.linalg.norm(coarse.vertices - mesh.vertices[survivors], axis=1)
    edge = _mean_edge(mesh)
    print(f"  proxy {n_coarse}v; rest-to-anchor drift {drift.mean()/edge:.2f}x/"
          f"{drift.max()/edge:.2f}x edge (absorbed by co-rotational LBS)")

    # --- PBD-simulate the coarse proxy (dancing sway) ---
    pinned = find_pinned_vertices(coarse.vertices, args.pin_fraction)
    assert len(pinned) > 0, "no pinned vertices on the proxy top"
    scenario = DancingSway(n_frames=args.frames, cycles=args.cycles,
                           amplitude_x=args.amplitude, sharpness=args.sharpness)
    scenario.setup(coarse.vertices, pinned)
    print(f"\n  simulating proxy ({args.settle} settle + {args.frames} logged, "
          f"{len(pinned)} pinned)...")
    t0 = time.time()
    X_coarse = run_sim(coarse.vertices, coarse.faces, pinned,
                       per_frame=scenario.per_frame, n_frames=args.frames,
                       n_settle=args.settle, dt=args.dt, iters=args.iters,
                       k_stretch=args.k_stretch, k_bend=args.k_bend, k_damp=args.k_damp,
                       friction=args.friction, restitution=args.restitution,
                       contact_skin=args.contact_skin)
    print(f"    proxy sim {time.time()-t0:.1f}s; "
          f"proxy y=[{X_coarse[...,1].min():.3f}, {X_coarse[...,1].max():.3f}]")

    # --- drive the fine mesh: full 3x3-block / scalar LBS / co-rotational LBS ---
    sel = {"both": ["full", "scalar"],
           "all": ["full", "scalar", "corotational"]}.get(args.deform, [args.deform])
    # scalar weights drive both 'scalar' and 'corotational' (and the diagnostic), so
    # build them whenever the operator is needed.
    W = sm.prolongation_scalar_weights(S, n_fine, n_coarse)
    recons = {}                                                 # label -> (T, n_fine, 3)

    if "full" in sel:
        t0 = time.time()
        Vf = reconstruct_full(S, mesh.vertices, coarse.vertices, X_coarse)
        recons["full 3x3"] = Vf
        print(f"    full 3x3-block drive   {time.time()-t0:4.1f}s; "
              f"max |d_fine|={np.linalg.norm(Vf - mesh.vertices[None], axis=2).max():.3f}")
    if "scalar" in sel:
        t0 = time.time()
        Vs = lbs_reconstruct(W, mesh.vertices, coarse.vertices, X_coarse)
        recons["scalar LBS"] = Vs
        nnz = int((np.abs(W) > 1e-9).sum())
        pou = float(np.abs(W.sum(1) - 1.0).max())
        eff = float(np.mean(1.0 / np.sum(W**2, axis=1).clip(1e-30)))
        print(f"    scalar LBS drive       {time.time()-t0:4.1f}s; w=(1/3)tr(S_block): "
              f"{nnz/(n_fine*n_coarse):.0%} dense, PoU {pou:.0e}, ~{eff:.1f} eff.handles/v; "
              f"max |d_fine|={np.linalg.norm(Vs - mesh.vertices[None], axis=2).max():.3f}")
    if "corotational" in sel:
        t0 = time.time()
        src, dst = ring_edges(coarse.faces, n_coarse)
        R_seq = estimate_handle_rotations(coarse.vertices, src, dst, X_coarse)
        V_corot = reconstruct_corotational(W, coarse.vertices, mesh.vertices,
                                           X_coarse, R_seq)
        recons["corotational"] = V_corot
        ang = _rotation_angle(R_seq)
        print(f"    co-rotational LBS drive {time.time()-t0:4.1f}s; per-handle |R| "
              f"mean {ang.mean():.1f}deg / max {ang.max():.1f}deg; "
              f"max |d_fine|={np.linalg.norm(V_corot - mesh.vertices[None], axis=2).max():.3f}")
    if "full 3x3" in recons and "scalar LBS" in recons:
        diff = np.linalg.norm(recons["full 3x3"] - recons["scalar LBS"], axis=2)
        print(f"  full-vs-scalar divergence: mean {diff.mean()/edge:.2f}x / "
              f"max {diff.max()/edge:.2f}x edge (where scalar LBS drops the rotation coupling)")

    # handle-kink diagnostic: Laplacian roughness at handles vs the mesh average, at the
    # strongest-sway frame (lower = smoother handles; tune with --smooth)
    fsrc, fdst = ring_edges(mesh.faces, n_fine)
    tmax = int(np.argmax(np.abs(X_coarse[..., 0] - coarse.vertices[None, :, 0]).mean(1)))
    print(f"\n  handle-kink (Laplacian roughness at handles / mesh avg, frame {tmax}; "
          f"1.0 = no kink, --smooth lowers it):")
    for lab, V in recons.items():
        print(f"    {lab:<18s} {handle_roughness(V[tmax] - mesh.vertices, fsrc, fdst, n_fine, survivors):.2f}")

    # finite-rotation fidelity: the empirical payoff (corot reproduces finite rotation)
    rotfid, rec_deg = finite_rotation_fidelity(S, W, coarse, mesh.vertices, args.rot_angle)
    print(f"\n  finite-rotation fidelity (synthetic {args.rot_angle:g}deg rigid proxy tilt; "
          f"corot recovered ~{rec_deg:.1f}deg/handle):")
    for k in ("full 3x3", "scalar LBS", "corotational"):
        print(f"    {k:<14s} rel.err {rotfid[k]:.3f}")
    print("    (lower = better; corot ~0 reproduces finite rotation, the linear block "
          "inflates, scalar can't rotate)")

    if args.smoke:
        print("\n--- Smoke checks ---")
        assert X_coarse.shape == (args.frames, n_coarse, 3), X_coarse.shape
        for lab, V in recons.items():
            assert V.shape == (args.frames, n_fine, 3), (lab, V.shape)
            assert np.isfinite(V).all(), f"{lab}: non-finite"
        tvec = np.array([0.13, -0.37, 0.21])
        Xt = coarse.vertices[None] + tvec                       # pure proxy translation
        if "full" in sel:
            assert S.shape == (3 * n_fine, 3 * n_coarse), S.shape
            tr = np.abs((reconstruct_full(S, mesh.vertices, coarse.vertices, Xt)[0]
                         - mesh.vertices) - tvec).max()
            assert tr < 1e-4, f"full: translation not reproduced: {tr:.2e}"
            if args.relax == 0:                                 # exact map: handle rows are identity
                hd = np.abs((recons["full 3x3"][:, survivors] - mesh.vertices[survivors][None])
                            - (X_coarse - coarse.vertices[None])).max()
                assert hd < 1e-8, f"full: handle rows not identity: {hd:.2e}"
                print(f"  full   : handle-id err {hd:.1e}; translation {tr:.1e}")
            else:
                print(f"  full   : soft handles (relax={args.relax:g}); translation {tr:.1e}")
        pou = float(np.abs(W.sum(1) - 1.0).max())
        assert pou < 1e-4, f"scalar weights not partition-of-unity: {pou:.2e}"
        if "corotational" in sel:
            # co-rotational reproduces rest and finite rotation
            R_id = np.tile(np.eye(3), (1, n_coarse, 1, 1))
            Vr = reconstruct_corotational(W, coarse.vertices, mesh.vertices,
                                          coarse.vertices[None], R_id)
            rest = np.abs(Vr[0] - mesh.vertices).max()
            assert rest < 1e-4, f"corot: rest not reproduced: {rest:.2e}"
            assert rotfid["corotational"] < 1e-3, \
                f"corot: finite rotation not reproduced: {rotfid['corotational']:.2e}"
            assert rotfid["full 3x3"] > rotfid["corotational"], "full should trail corot on rotation"
            assert rotfid["scalar LBS"] > rotfid["corotational"], "scalar should trail corot on rotation"
            print(f"  corot  : rest {rest:.1e}; finite-rot err {rotfid['corotational']:.1e} "
                  f"(<< full {rotfid['full 3x3']:.2f}, scalar {rotfid['scalar LBS']:.2f})")
        print("\nSMOKE: PASS")
        return

    _render(mesh, coarse, X_coarse, recons, args.screenshot)


def _mean_edge(mesh):
    V, F = mesh.vertices, mesh.faces
    e = [np.linalg.norm(V[F[:, a]] - V[F[:, b]], axis=1) for a, b in ((0, 1), (1, 2), (0, 2))]
    return float(np.concatenate(e).mean())


def handle_roughness(d, src, dst, n, handles):
    """Graph-Laplacian roughness  r_i = |d_i - mean_{j~i} d_j|  of a displacement field d,
    returned as the ratio handle-mean / mesh-mean.  >1 => kinks concentrate at handles
    (the membrane interpolant's cusp); ~1 => the handles are as smooth as the surface."""
    deg = np.maximum(np.bincount(src, minlength=n), 1)
    sums = np.zeros((n, 3)); np.add.at(sums, src, d[dst])
    r = np.linalg.norm(d - sums / deg[:, None], axis=1)
    return float(r[handles].mean() / (r.mean() + 1e-30))


def _render(mesh, coarse, X_coarse, recons, screenshot):
    import polyscope as ps
    import polyscope.imgui as psim
    verts, faces = mesh.vertices, mesh.faces
    n_frames = X_coarse.shape[0]

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_front_dir("neg_z_front")
    ps.set_ground_plane_mode("none")
    bbox = verts.max(0) - verts.min(0)
    dx = bbox[0] * 1.4
    # columns are proxy then each reconstruction; neg_z_front mirrors x, so assign
    # DESCENDING x to read left->right on screen as [proxy, <recons in order>].
    ncols = 1 + len(recons)

    def off(c):
        return np.array([(ncols - 1 - c) * dx, 0.0, 0.0])

    palette = {"full 3x3": [0.55, 0.75, 0.95], "scalar LBS": [0.55, 0.9, 0.6],
               "corotational": [0.80, 0.62, 0.95]}
    proxy = ps.register_surface_mesh("coarse proxy (PBD)", coarse.vertices + off(0),
                                     coarse.faces, edge_width=1.0, color=[0.95, 0.6, 0.25])
    hp = ps.register_point_cloud("proxy verts", coarse.vertices + off(0)); hp.set_radius(0.004)
    fine_meshes = []
    for c, (lab, V) in enumerate(recons.items(), start=1):
        m = ps.register_surface_mesh(f"fine: {lab}", verts + off(c), faces, edge_width=0.0,
                                     color=palette.get(lab, [0.7, 0.7, 0.72]))
        fine_meshes.append((m, V, off(c)))

    def show_frame(t):
        proxy.update_vertex_positions(X_coarse[t] + off(0))
        hp.update_point_positions(X_coarse[t] + off(0))
        for m, V, o in fine_meshes:
            m.update_vertex_positions(V[t] + o)

    # screenshot: the strongest-sway frame (max mean |x| displacement of the proxy)
    if screenshot:
        sway = np.abs(X_coarse[..., 0] - coarse.vertices[None, :, 0]).mean(1)
        t = int(np.argmax(sway))
        show_frame(t)
        pts = [X_coarse[t] + off(0)] + [V[t] + o for _, V, o in fine_meshes]
        ext = np.vstack(pts); ext = ext.max(0) - ext.min(0)
        ps.set_window_size(int(760 * max(ext[0] / (ext[1] + 1e-30), 1.0)), 760)
        ps.set_view_projection_mode("orthographic")
        ps.reset_camera_to_home_view()
        ps.screenshot(screenshot, transparent_bg=False)
        print(f"  saved {screenshot} (strongest-sway frame {t})")
        return

    state = {"frame": 0, "playing": True, "fps": 60.0, "last": 0.0}

    def cb():
        now = time.perf_counter()
        if psim.Button("pause" if state["playing"] else "play"):
            state["playing"] = not state["playing"]; state["last"] = now
        psim.SameLine(); _, state["fps"] = psim.SliderFloat("fps", state["fps"], 1.0, 120.0)
        advanced = False
        if state["playing"] and now - state["last"] >= 1.0 / max(state["fps"], 1e-3):
            state["frame"] = (state["frame"] + 1) % n_frames; state["last"] = now; advanced = True
        changed, state["frame"] = psim.SliderInt("frame", state["frame"], 0, n_frames - 1)
        if changed or advanced:
            show_frame(state["frame"])

    show_frame(0)
    ps.set_user_callback(cb)
    ps.reset_camera_to_home_view()
    print("\n  Polyscope open — left: PBD proxy sim; then fine mesh per drive: "
          + ", ".join(recons.keys()) + ".")
    print("  play/pause + frame slider in the panel.")
    ps.show()


if __name__ == "__main__":
    main()
