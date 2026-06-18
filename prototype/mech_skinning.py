"""Connect mechanics-aware simplification to skinning weights.

    simplify_mechanics  ->  surviving fine-vertex indices = coarse handles R
                        ->  harmonic-extend the fine elastic Hessian K on (R, E)
                        ->  S = -K_ee^{-1} K_er   (skinning weights, no fitting)

So the mechanics cost decides WHICH vertices become the proxy/handles, and the
harmonic extension supplies the weights that drive the full mesh from them.

The decimation is bending-aware (curvature probe + hinge term, bending_weight=1.0),
matching `viz_simplify --bending-weight 1.0` and viz_skinning -- all three use the
SAME proxy and share its on-disk cache.

(We use only the decimation's CHOICE of survivors, at their original rest
positions, so the fine K stays consistent. The decimation thickness is decoupled
from the skinning-K thickness; the tighter "emit weights during the collapse"
variant is left for later.)

Compares mechanics-chosen handles vs farthest-point-sampled (FPS) handles of the
same count -- each with the harmonic skin and a geometric kNN baseline -- on
rigid / elastic / sinusoid test deformations, and reports handle uniformity
(the mechanics cost concentrates handles at features; Zheng et al. 2024 warn a
non-uniform proxy breaks PBD).

Run:
  python prototype/mech_skinning.py                       # 128 handles
  python prototype/mech_skinning.py --handles 200
  python prototype/mech_skinning.py --smoke
  python prototype/mech_skinning.py --screenshot out/mech_skinning.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harmonic_skinning as hs  # noqa: E402
import simplify_mechanics as sm  # noqa: E402
import mech_qem as mq  # noqa: E402

from kms.mesh import load_obj, save_obj  # noqa: E402


DEFAULT_MESH = "data/9423122485_cleaned.obj"


def rotation_field(verts, seed=0):
    """Pure rigid rotation (no translation), so the rigid error isn't diluted by
    a large translation term -- isolates rotation handling."""
    rng = np.random.default_rng(seed)
    w = 0.4 * hs._rand_unit(rng)
    return np.cross(w[None, :], verts - verts.mean(0))


def uniformity_cv(verts, R):
    """Coefficient of variation of nearest-handle spacing (lower = more uniform)."""
    d, _ = cKDTree(verts[R]).query(verts[R], k=2)
    nn = d[:, 1]
    return float(nn.std() / (nn.mean() + 1e-30))


def eval_handle_set(K, verts, R, knn):
    E = np.array(sorted(set(range(len(verts))) - set(R.tolist())), dtype=np.int64)
    X, r_dofs, e_dofs = hs.harmonic_prolongation(K, R, E)
    W = hs.geometric_weights(verts, R, E, k=knn)
    return E, X, r_dofs, e_dofs, W


# Decimation thickness is DECOUPLED from the skinning-K thickness, matching
# viz_skinning: greedy collapse is path-dependent, so to reproduce
# `viz_simplify --bending-weight 1.0` we must match its thickness (1e-3) even
# though it cancels in the cost algebraically. The skinning K (built in main with
# --thickness, ~0.05) needs a thicker shell or K_ee goes ill-conditioned.
DECIM_THICKNESS = 1e-3  # matches viz_simplify.py's default --thickness


def mechanics_handles(mesh, n_handles, stem, bending_weight=1.0,
                      decim_thickness=DECIM_THICKNESS, cache_dir="out"):
    """Coarse handles = vertices that survive a mechanics-aware decimation,
    bending-aware to match `viz_simplify --bending-weight 1.0` and viz_skinning
    (curvature probe + hinge term when bending_weight>0). Returns (survivors,
    coarse_mesh) and shares viz_skinning's on-disk proxy cache (identical key), so
    the slow decimation runs at most once across both scripts."""
    probe = "curvature" if bending_weight > 0.0 else "affine"
    base = os.path.join(
        cache_dir,
        f"_mech_proxy_{stem}_{n_handles}_{decim_thickness:g}_b{bending_weight:g}_{probe}")
    npy, obj = base + ".npy", base + ".obj"
    if os.path.exists(npy):
        print(f"  mechanics handles: loaded cached {npy}")
        return np.load(npy), (load_obj(obj) if os.path.exists(obj) else None)
    t0 = time.time()
    coarse, survivors = sm.simplify_mechanics(
        mesh, target_verts=n_handles, thickness=decim_thickness, probe=probe,
        bending_weight=bending_weight, return_survivors=True)
    print(f"  mechanics handles: decimated {mesh.n_verts} -> {len(survivors)} verts "
          f"(probe={probe}, bending_weight={bending_weight:g}, "
          f"thickness={decim_thickness:g}) in {time.time()-t0:.1f}s")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(npy, survivors)
    save_obj(coarse, obj)
    return survivors, coarse


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=DEFAULT_MESH)
    p.add_argument("--handles", type=int, default=128)
    p.add_argument("--thickness", type=float, default=0.05,
                   help="shell thickness for the SKINNING Hessian K (conditions the "
                        "out-of-plane harmonic solve); the decimation uses its own "
                        "thickness (DECIM_THICKNESS=1e-3) to match viz_simplify")
    p.add_argument("--bending-weight", type=float, default=1.0,
                   help="bending weight for the mechanics decimation, mirroring "
                        "viz_simplify.py --bending-weight (>0 uses the curvature probe)")
    p.add_argument("--knn", type=int, default=4)
    p.add_argument("--screenshot", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    from pathlib import Path
    stem = Path(args.mesh).stem

    print("=== Mechanics-aware simplification -> skinning weights ===")
    mesh = load_obj(args.mesh)
    verts = mesh.vertices
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    model = mq.build_model(mesh, thickness=args.thickness)
    K = hs.assemble_K_sparse(model)
    print(f"  {mesh.n_verts} verts; fine K assembled ({K.shape[0]} dof)")

    # --- the connection: decimation survivors as handles, + FPS for comparison ---
    # decimation is bending-aware (matches viz_simplify/viz_skinning) and uses its
    # OWN thickness; the fine K above keeps the thicker, well-conditioned shell.
    R_mech, coarse = mechanics_handles(mesh, args.handles, stem,
                                       bending_weight=args.bending_weight)
    R_fps = hs.farthest_point_sampling(verts, len(R_mech))
    if coarse is not None:
        save_obj(coarse, os.path.join("out", f"mech_proxy_{len(R_mech)}.obj"))

    # --- shared test deformations (identical for both handle sets => fair) ---
    fields = {
        "rigid": [rotation_field(verts, seed=0)],
        "elastic": hs.elastic_fields(K, verts, ntrials=10, seed=3),
        "sinusoid": [d for _, d in hs.sinusoidal_fields(verts, diag, seed=1)],
    }

    results = {}
    for label, R in {"mechanics": R_mech, "FPS": R_fps}.items():
        E, X, r_dofs, e_dofs, W = eval_handle_set(K, verts, R, args.knn)
        cv = uniformity_cv(verts, R)
        per = {}
        for fname, ds in fields.items():
            errs = np.array([hs.reconstruct_errors(d, R, E, X, r_dofs, e_dofs, W) for d in ds])
            per[fname] = (float(errs[:, 0].mean()), float(errs[:, 1].mean()))
        results[label] = {"cv": cv, "per": per, "ctx": (R, E, X, r_dofs, W)}

    # --- report ---
    print("\n  handle uniformity (CV of nearest-handle spacing; lower = more uniform):")
    for label in ("mechanics", "FPS"):
        print(f"    {label:<10s} {results[label]['cv']:.3f}")
    print("\n  mean relative L2 reconstruction error over eliminated verts:")
    print(f"    {'handles':<10s} {'skin':<10s} {'rigid':>10s} {'elastic':>10s} {'sinusoid':>10s}")
    for label in ("mechanics", "FPS"):
        per = results[label]["per"]
        for si, sk in enumerate(("harmonic", "geometric")):
            print(f"    {label:<10s} {sk:<10s}"
                  f" {per['rigid'][si]:10.3e} {per['elastic'][si]:10.3e} {per['sinusoid'][si]:10.3e}")

    me, fe = results["mechanics"]["per"]["elastic"][0], results["FPS"]["per"]["elastic"][0]
    print(f"\n  elastic harmonic: mechanics={me:.3e} vs FPS={fe:.3e} "
          f"({'mechanics better' if me < fe else 'FPS better'} by {max(me,fe)/min(me,fe):.2f}x)")
    print("  (mechanics handles cluster at features -> less uniform; whether that")
    print("   helps the skin is exactly what this comparison shows.)")

    if args.smoke:
        print("\n--- Smoke checks ---")
        for label in ("mechanics", "FPS"):
            per = results[label]["per"]
            assert per["rigid"][0] < per["rigid"][1], f"{label}: harmonic should win on rigid"
            assert per["elastic"][0] < per["elastic"][1], f"{label}: harmonic should win on elastic"
        assert len(R_mech) == args.handles, f"expected {args.handles} handles, got {len(R_mech)}"
        print("  handles wired from decimation; harmonic<geometric (rigid & elastic) both sets")
        print("\nSMOKE: PASS")
        return

    if args.screenshot:
        _render(mesh, fields["elastic"][0], results, args.screenshot)


def _render(mesh, d_true, results, path):
    """Elastic case: truth, mechanics-harmonic, FPS-harmonic, colored by error."""
    import polyscope as ps

    verts, faces = mesh.vertices, mesh.faces
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    scale = 0.2 * diag / (np.linalg.norm(d_true, axis=1).max() + 1e-30)

    def recon(ctx):
        R, E, X, r_dofs, W = ctx
        out = np.zeros_like(d_true); out[R] = d_true[R]
        out[E] = (X @ d_true.ravel()[r_dofs]).reshape(len(E), 3)
        return out

    dm = recon(results["mechanics"]["ctx"])
    df = recon(results["FPS"]["ctx"])
    em = np.linalg.norm(dm - d_true, axis=1)
    ef = np.linalg.norm(df - d_true, axis=1)
    vmax = float(max(em.max(), ef.max()))

    ps.init(); ps.set_up_dir("y_up"); ps.set_ground_plane_mode("none")
    dx = (verts.max(0) - verts.min(0))[0] * 1.4
    panels = [("truth", d_true, None, None),
              ("mechanics->harmonic", dm, em, results["mechanics"]["ctx"][0]),
              ("FPS->harmonic", df, ef, results["FPS"]["ctx"][0])]
    for i, (name, disp, err, R) in enumerate(panels):
        P = verts + scale * disp + np.array([i * dx, 0, 0])
        m = ps.register_surface_mesh(name, P, faces, edge_width=0.0, color=[0.7, 0.7, 0.72])
        if err is not None:
            m.add_scalar_quantity("recon error", err, enabled=True, cmap="viridis", vminmax=(0, vmax))
        if R is not None:
            pc = ps.register_point_cloud(f"{name} handles", P[R]); pc.set_radius(0.006); pc.set_color([0.9, 0.1, 0.1])
    ps.reset_camera_to_home_view()
    ps.screenshot(path, transparent_bg=False)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
