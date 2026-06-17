"""Mechanical-QEM decimation --- visualize and compare against QEM.

Runs three decimations to the same target and shows them side-by-side:
  1. QEM                  (kms.simplify_qem, the geometric baseline)
  2. Mechanical (membrane)(simplify_mechanics, affine probe)
  3. Mechanical + geom    (simplify_mechanics, geom_weight = --geom-weight), OR
     Mechanical + bending (curvature probe, bending_weight) when --bending-weight > 0

Saves the results as .obj and prints a quality table (area ratio, triangle aspect).

Run:
  python prototype/viz_simplify.py                          # skirt mesh, target 600
  python prototype/viz_simplify.py --target 400
  python prototype/viz_simplify.py --mesh data/spot.obj --target 800
  python prototype/viz_simplify.py --geom-weight 2.0        # heavier visual term
  python prototype/viz_simplify.py --crease-demo            # folded grid: bending matters
  python prototype/viz_simplify.py --bending-weight 4       # bending-aware on --mesh
  python prototype/viz_simplify.py --quiet                  # hide per-collapse progress
  python prototype/viz_simplify.py --smoke                  # headless verification

Weights (geom_weight, bending_weight) are dimensionless: each cost term is divided
by the median of its own initial per-edge cost, so 1 == "as costly as a typical
membrane collapse" (useful range ~0-10; 0 = off). The 3rd variant is bending-aware
when --bending-weight > 0, otherwise geom-blended (so only one weight is active).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simplify_mechanics import simplify_mechanics  # noqa: E402

from kms.mesh import load_obj, save_obj, face_areas  # noqa: E402
from kms.simplify_qem import simplify_qem  # noqa: E402
from kms import colors  # noqa: E402


DEFAULT_MESH = "data/9423122485_cleaned.obj"


def quality(mesh):
    """(area_ratio max/min, mean aspect, max aspect) for a mesh."""
    a = face_areas(mesh)
    v, f = mesh.vertices, mesh.faces
    e = np.stack([
        np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1),
        np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1),
        np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1),
    ], axis=1)
    aspect = e.max(axis=1) / (e.min(axis=1) + 1e-30)
    return a.max() / (a.min() + 1e-30), aspect.mean(), aspect.max(), a.min()


def make_creased_grid(n=21, fold=0.35):
    """A flat grid folded into a 'V' along y = 0.5 -- a sharp crease that membrane
    cost ignores (flat in-plane) but bending cost sees."""
    from kms.mesh import make_grid
    m = make_grid(n, n)
    m.vertices[:, 2] = fold * np.abs(m.vertices[:, 1] - 0.5)
    return m


def run_all(mesh, target, thickness, geom_weight, bending_weight, verbose=True):
    """Returns list of (name, mesh, seconds, tag).  The third mechanical variant
    is bending-aware if bending_weight > 0, else geometry-blended."""
    results = []

    def banner(msg):
        if verbose:
            print(f"\n--- {msg} ---")

    banner("[1/3] QEM (geometric baseline)")
    t0 = time.time()
    r_qem = simplify_qem(mesh, target_verts=target, use_line_quadric=True, verbose=verbose)
    results.append(("QEM", r_qem, time.time() - t0, "qem"))

    banner("[2/3] Mechanical (membrane, affine probe)")
    t0 = time.time()
    r_mech = simplify_mechanics(mesh, target_verts=target, thickness=thickness,
                               verbose=verbose)
    results.append(("Mechanical (membrane)", r_mech, time.time() - t0, "mech"))

    t0 = time.time()
    if bending_weight > 0.0:
        banner(f"[3/3] Mechanical + bending (curvature probe, bending_weight={bending_weight:g})")
        r3 = simplify_mechanics(mesh, target_verts=target, thickness=thickness,
                                probe="curvature", bending_weight=bending_weight,
                                verbose=verbose)
        name = f"Mechanical+bending (w={bending_weight:g})"
        tag = "mech_bending"
    else:
        banner(f"[3/3] Mechanical + geom (geom_weight={geom_weight:g})")
        r3 = simplify_mechanics(mesh, target_verts=target, thickness=thickness,
                                geom_weight=geom_weight, verbose=verbose)
        name = f"Mechanical+geom (a={geom_weight:g})"
        tag = "mech_geom"
    results.append((name, r3, time.time() - t0, tag))

    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=DEFAULT_MESH, help="Path to .obj mesh")
    p.add_argument("--target", type=int, default=128, help="Target vertex count")
    # shell thickness is no longer relevant 
    # normalizing bending divides out its ~t³ magnitude, so no longer need a thick plate for bending to register — the weight alone controls it.
    p.add_argument("--thickness", type=float, default=1e-3, help="Shell thickness")
    p.add_argument("--geom-weight", type=float, default=1.0,
                   help="Weight of the geometric QEM term in the blended variant")
    p.add_argument("--bending-weight", type=float, default=0.0,
                   help="Weight of the bending term (curvature probe); >0 replaces "
                        "the geom variant with a bending-aware one")
    p.add_argument("--crease-demo", action="store_true",
                   help="Ignore --mesh: decimate a folded grid so bending matters "
                        "(auto-sets thickness/bending-weight for a visible effect)")
    p.add_argument("--output-dir", default="out", help="Where to save result .obj files")
    p.add_argument("--quiet", action="store_true", help="Suppress per-collapse progress")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.crease_demo:
        mesh = make_creased_grid(21, 0.35)
        stem = "crease"
        thickness = args.thickness
        bending_weight = args.bending_weight or 4.0  # dimensionless; >1 = bending-dominant
        target = args.target if args.target != 600 else 150
    else:
        mesh = load_obj(args.mesh)
        stem = Path(args.mesh).stem
        thickness = args.thickness
        bending_weight = args.bending_weight
        target = args.target

    # the mechanical cost is ~20-130 ms/collapse; in smoke, cap collapses so it stays quick
    if args.smoke:
        target = max(target, int(0.8 * mesh.n_verts))

    # The 3rd variant is bending-aware XOR geom-blended -- so report the inactive
    # weight as 0 to avoid the "both shown" confusion.
    bending_active = bending_weight > 0.0
    eff_geom_weight = 0.0 if bending_active else args.geom_weight

    print("=== Mechanical-QEM Decimation ===")
    print(f"  Input:  {mesh.n_verts} verts, {mesh.n_faces} faces")
    print(f"  Target: {target} verts   thickness={thickness:g}")
    print(f"  3rd variant: {'bending-aware (curvature probe)' if bending_active else 'geom-blended'}"
          f"   ->   geom_weight={eff_geom_weight:g}, bending_weight={bending_weight:g}")
    print("  (weights are dimensionless: 1 == as costly as a typical membrane collapse)")

    results = run_all(mesh, target, thickness, eff_geom_weight, bending_weight,
                      verbose=not args.quiet)

    for name, r, dt, _ in results:
        ar, am, amax, amin = quality(r)
        print(f"  {name:<28s} {r.n_verts}v {r.n_faces}f  time={dt:6.2f}s  "
              f"area_ratio={ar:8.0f}  aspect(mean/max)={am:5.1f}/{amax:6.1f}  "
              f"min_area={amin:.2e}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, r, _, tag in results:
        save_obj(r, str(out_dir / f"{stem}_{tag}_{target}.obj"))
    print(f"\n  Saved {len(results)} meshes to {out_dir}/")

    if args.smoke:
        print("\n--- Smoke checks ---")
        for name, r, _, _ in results:
            f = r.faces
            assert r.n_verts == target, f"{name}: {r.n_verts} != target {target}"
            assert not np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])), \
                f"{name}: repeated vertex in a face"
            assert r.faces.max() < r.n_verts, f"{name}: face index out of range"
            # zero-area (collinear) faces are a WARNING, not a failure: pure-membrane
            # is known to degenerate on creases (it needs an E_triq sliver guard) --
            # which is itself part of why bending / a quality term matters.
            n_zero = int(np.sum(face_areas(r) <= 0))
            if n_zero:
                print(f"  WARN {name}: {n_zero} zero-area face(s) (pure-membrane needs E_triq)")
        print("  all results structurally valid (target hit, no duplicate/out-of-range faces)")
        print("\nSMOKE: PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        ps.set_front_dir("neg_z_front")

        bbox = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        spacing = bbox[0] * 1.3

        panels = [("original", mesh, colors.RENDER_COLORS["gray"])]
        for i, (name, r, _, _) in enumerate(results):
            panels.append((f"{name} ({r.n_verts}v)", r, colors.get_color_by_index(i)))

        for i, (name, m, color) in enumerate(panels):
            verts = m.vertices.copy()
            verts[:, 0] += i * spacing
            ps.register_surface_mesh(name, verts, m.faces, edge_width=1.0, color=color)

        print("\n  Polyscope open. Left to right:")
        for name, _, _ in panels:
            print(f"    {name}")
        ps.show()


if __name__ == "__main__":
    main()
