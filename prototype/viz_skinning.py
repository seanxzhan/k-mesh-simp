"""Visualize harmonic vs geometric skinning, driven by a mechanics-decimated proxy.

Builds the fine elastic Hessian K, gets coarse handles, and forms two skins:
  - harmonic   : S = -K_ee^{-1} K_er   (energy-minimizing, from K, no fitting)
  - geometric  : inverse-distance kNN to the handles (naive LBS baseline)

Handle source (--source):
  - "mechanics" (default): the handles are the vertices that SURVIVE a
        mechanics-aware decimation (simplify_mechanics) -- bending-aware, to
        match `viz_simplify.py --bending-weight 1.0` (curvature probe + hinge
        term). The coarse decimated proxy mesh is shown alongside, at rest.
  - "fps": farthest-point-sampled handles (uniform; for comparison).

For three handle motions it reconstructs the full mesh from the handle
displacements with each skin, shown in a grid (rows = motion, cols = truth /
harmonic / geometric), colored by per-vertex error. The coarse decimated mesh
sits to the left so you can see the proxy that drives the skin.

The three motions:
  - rigid    : a pure rotation (harmonic reproduces it exactly; translation-only
               geometric LBS cannot).
  - elastic  : a static elastic response, d = (K+epsI)^{-1} f for random f with
               the rigid part removed -- soft-mode-dominated (the skinning regime).
  - sinusoid : an arbitrary smooth field sin(omega . X) -- NOT elastic; an honest
               stress test (rewards a generic interpolator, not energy-min).

Run:
  python prototype/viz_skinning.py                       # mechanics proxy, interactive
  python prototype/viz_skinning.py --source fps          # uniform handles instead
  python prototype/viz_skinning.py --scale 1.5           # exaggerate displacement
  python prototype/viz_skinning.py --screenshot out/skinning.png
  python prototype/viz_skinning.py --smoke               # headless checks
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harmonic_skinning as hs  # noqa: E402
import mech_qem as mq  # noqa: E402

from kms.mesh import load_obj, save_obj, TriMesh  # noqa: E402


DEFAULT_MESH = "data/9423122485_cleaned.obj"


def rotation_field(verts, seed=0):
    """A pure rigid rotation displacement (no translation, so panels stay put).
    Harmonic reproduces it exactly; translation-only geometric LBS cannot."""
    rng = np.random.default_rng(seed)
    c = verts.mean(0)
    w = 0.4 * hs._rand_unit(rng)
    return np.cross(w[None, :], verts - c)


def full_reconstruct(d_true, ctx, method):
    """Reconstruct the whole-mesh displacement from the handle displacements.
    Retained handles are exact for both skins; eliminated verts use the skin."""
    R, E, X, r_dofs, W = ctx["R"], ctx["E"], ctx["X"], ctx["r_dofs"], ctx["W"]
    out = np.zeros_like(d_true)
    out[R] = d_true[R]
    if method == "harmonic":
        out[E] = (X @ d_true.ravel()[r_dofs]).reshape(len(E), 3)
    else:
        out[E] = W @ d_true[R]
    return out


# The decimation thickness is DECOUPLED from the skinning thickness, on purpose:
#   - the collapse cost is thickness-invariant in exact arithmetic (membrane ~ t^2,
#     bending ~ t^6, both divided by their own medians), BUT greedy edge-collapse is
#     path-dependent -- at different absolute magnitudes, floating-point rounding
#     breaks near-ties differently and the survivor set diverges. So to reproduce
#     `viz_simplify --bending-weight 1.0` EXACTLY we must match its thickness (1e-3).
#   - the skinning Hessian K, by contrast, needs a thicker shell (~0.05): the bending
#     term conditions the out-of-plane harmonic solve, and at 1e-3 K_ee goes
#     ill-conditioned (partition-of-unity / rigid reproduction break).
DECIM_THICKNESS = 1e-3  # matches viz_simplify.py's default --thickness


def get_mechanics_proxy(mesh, n_handles, stem, bending_weight=1.0,
                        decim_thickness=DECIM_THICKNESS, cache_dir="out"):
    """Coarse handles = survivors of a mechanics-aware decimation; also returns
    the coarse decimated mesh. Cached (the decimation is the slow step).

    Mirrors `viz_simplify.py --bending-weight <w>`: bending_weight>0 selects the
    curvature probe + hinge term (the bending-aware collapse), exactly as
    viz_simplify does; bending_weight=0 falls back to pure-membrane/affine.
    decim_thickness matches viz_simplify's default so the survivor set is identical
    (greedy collapse is path-dependent, so the thickness must match even though it
    cancels in exact arithmetic -- see the note above)."""
    probe = "curvature" if bending_weight > 0.0 else "affine"
    base = os.path.join(
        cache_dir,
        f"_mech_proxy_{stem}_{n_handles}_{decim_thickness:g}_b{bending_weight:g}_{probe}")
    npy, obj = base + ".npy", base + ".obj"
    if os.path.exists(npy) and os.path.exists(obj):
        print(f"  mechanics proxy: loaded cached {obj}")
        return np.load(npy), load_obj(obj)
    import time
    import simplify_mechanics as sm
    t0 = time.time()
    coarse, survivors = sm.simplify_mechanics(
        mesh, target_verts=n_handles, thickness=decim_thickness, probe=probe,
        bending_weight=bending_weight, return_survivors=True)
    print(f"  mechanics proxy: decimated {mesh.n_verts} -> {len(survivors)} verts "
          f"(probe={probe}, bending_weight={bending_weight:g}, "
          f"thickness={decim_thickness:g}) in {time.time()-t0:.1f}s")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(npy, survivors)
    save_obj(coarse, obj)
    return survivors, coarse


def build_context(mesh, handles, thickness, knn, source, stem, bending_weight):
    verts = mesh.vertices
    model = mq.build_model(mesh, thickness=thickness)   # skinning K (well-conditioned)
    K = hs.assemble_K_sparse(model)

    coarse_mesh = None
    if source == "mechanics":
        # the decimation uses its OWN thickness (DECIM_THICKNESS, 1e-3) to match
        # viz_simplify; the skinning K above keeps the thicker, well-conditioned shell
        R, coarse_mesh = get_mechanics_proxy(mesh, handles, stem,
                                             bending_weight=bending_weight)
    else:
        R = hs.farthest_point_sampling(verts, handles)

    E = np.array(sorted(set(range(mesh.n_verts)) - set(R.tolist())), dtype=np.int64)
    X, r_dofs, e_dofs = hs.harmonic_prolongation(K, R, E)
    W = hs.geometric_weights(verts, R, E, k=knn)
    return {"verts": verts, "K": K, "R": R, "E": E, "X": X, "r_dofs": r_dofs,
            "W": W, "coarse_mesh": coarse_mesh, "source": source}


def make_rows(mesh, ctx, scale_mult):
    """For each deformation type, return (name, d_true, d_harm, d_geom, scale)."""
    verts = ctx["verts"]
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    deforms = {
        "rigid": rotation_field(verts, seed=0),
        "elastic": hs.elastic_fields(ctx["K"], verts, ntrials=1, seed=3)[0],
        "sinusoid": next(d for f, d in hs.sinusoidal_fields(verts, diag, seed=1) if f == 2.0),
    }
    rows = []
    for name, d in deforms.items():
        mag = np.linalg.norm(d, axis=1).max() + 1e-30
        s = scale_mult * 0.2 * diag / mag
        d_h = full_reconstruct(d, ctx, "harmonic")
        d_g = full_reconstruct(d, ctx, "geometric")
        rows.append((name, s * d, s * d_h, s * d_g, s))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=DEFAULT_MESH)
    p.add_argument("--handles", type=int, default=128)
    p.add_argument("--thickness", type=float, default=0.05,
                   help="shell thickness for the SKINNING Hessian K (conditions the "
                        "out-of-plane harmonic solve); the decimation uses its own "
                        "thickness (DECIM_THICKNESS=1e-3) to match viz_simplify")
    p.add_argument("--knn", type=int, default=4)
    p.add_argument("--scale", type=float, default=1.0, help="displacement exaggeration")
    p.add_argument("--source", choices=["mechanics", "fps"], default="mechanics",
                   help="handle source: mechanics-decimation survivors (default) or FPS")
    p.add_argument("--bending-weight", type=float, default=1.0,
                   help="bending weight for the mechanics decimation, mirroring "
                        "viz_simplify.py --bending-weight (>0 uses the curvature probe)")
    p.add_argument("--screenshot", default=None, help="render to this path and exit")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    print(f"=== Skinning viz (handles: {args.source}) ===")
    mesh = load_obj(args.mesh)
    ctx = build_context(mesh, args.handles, args.thickness, args.knn, args.source,
                        Path(args.mesh).stem, args.bending_weight)
    rows = make_rows(mesh, ctx, args.scale)

    print(f"  {mesh.n_verts} verts, {len(ctx['R'])} handles\n")
    print(f"  {'deformation':<10s} {'harmonic err':>14s} {'geometric err':>15s}")
    row_err = {}
    for name, dt, dh, dg, s in rows:
        eh = np.linalg.norm(dh - dt, axis=1)
        eg = np.linalg.norm(dg - dt, axis=1)
        row_err[name] = (eh, eg)
        print(f"  {name:<10s} {eh.mean():14.3e} {eg.mean():15.3e}   "
              f"(geom/harm = {eg.mean()/(eh.mean()+1e-30):.1f}x)")

    if args.smoke:
        print("\n--- Smoke checks ---")
        for key in ("rigid", "elastic"):
            eh, eg = row_err[key]
            assert eh.mean() < eg.mean(), f"harmonic should beat geometric on {key}"
        if args.source == "mechanics":
            assert ctx["coarse_mesh"] is not None, "mechanics source must yield a coarse mesh"
        print("  rows built; harmonic < geometric on rigid & elastic")
        print("\nSMOKE: PASS")
        return

    import polyscope as ps

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")

    verts, faces = mesh.vertices, mesh.faces
    bbox = verts.max(0) - verts.min(0)
    dx = bbox[0] * 1.4          # columns: truth | harmonic | geometric
    dy = bbox[1] * 1.6          # rows:    rigid / elastic / sinusoid (top -> bottom)

    # the coarse decimated proxy mesh, at rest, to the LEFT of the grid
    if ctx["coarse_mesh"] is not None:
        cm = ctx["coarse_mesh"]
        Pc = cm.vertices + np.array([-1.5 * dx, -dy, 0.0])
        ps.register_surface_mesh("coarse decimated proxy", Pc, cm.faces,
                                 edge_width=1.5, color=[0.95, 0.6, 0.25])
        pc = ps.register_point_cloud("proxy vertices (handles)", Pc)
        pc.set_radius(0.008); pc.set_color([0.9, 0.1, 0.1])
        print(f"  coarse decimated proxy: {cm.n_verts} verts, {cm.n_faces} faces "
              "(shown at far left, rest pose)")

    for ri, (name, dt, dh, dg, s) in enumerate(rows):
        eh, eg = row_err[name]
        vmax = float(max(eh.max(), eg.max()))
        for ci, (col, disp, err) in enumerate(
            [("truth", dt, None), ("harmonic", dh, eh), ("geometric", dg, eg)]
        ):
            P = verts + disp + np.array([ci * dx, -ri * dy, 0.0])
            m = ps.register_surface_mesh(f"{name} | {col}", P, faces, edge_width=0.0,
                                         color=[0.7, 0.7, 0.72])
            if err is not None:
                m.add_scalar_quantity("recon error", err, enabled=True,
                                      cmap="viridis", vminmax=(0.0, vmax))
        hp = (verts + dt)[ctx["R"]] + np.array([0.0, -ri * dy, 0.0])
        pc = ps.register_point_cloud(f"{name} | handles", hp)
        pc.set_radius(0.006); pc.set_color([0.9, 0.1, 0.1])

    ps.reset_camera_to_home_view()

    print("\n  Polyscope open. Far left: the coarse decimated proxy (rest).")
    print("  Grid: rows = rigid / elastic / sinusoid (top->bottom),")
    print("  columns = truth | harmonic | geometric (left->right). Red dots = handles.")
    print("  Reconstructions colored by per-vertex error (shared scale per row).")
    if args.screenshot:
        ps.set_view_projection_mode("orthographic")
        ps.reset_camera_to_home_view()
        ps.screenshot(args.screenshot, transparent_bg=False)
        print(f"  saved {args.screenshot}")
    else:
        ps.show()


if __name__ == "__main__":
    main()
