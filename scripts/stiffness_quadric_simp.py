"""Stiffness Quadric Simplification — Compare against QEM.

Simplifies a mesh using stiffness quadrics (Approach 2) and QEM,
displaying results side-by-side.

Run:
  python scripts/stiffness_quadric_simp.py                            # spot.obj
  python scripts/stiffness_quadric_simp.py --mesh data/spot.obj       # custom mesh
  python scripts/stiffness_quadric_simp.py --target 500               # custom target
  python scripts/stiffness_quadric_simp.py --smoke                    # headless
"""

import argparse
import time
from pathlib import Path

import numpy as np

from kms.mesh import load_obj, save_obj, face_areas
from kms.simplify_qem import simplify_qem
from kms.simplify_stiffness_quadric import simplify_stiffness_quadric
from kms import colors


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mesh", default="data/spot.obj", help="Path to .obj mesh file"
    )
    parser.add_argument(
        "--target", type=int, default=128, help="Target vertex count"
    )
    parser.add_argument(
        "--thickness", type=float, default=0.01, help="Shell thickness"
    )
    parser.add_argument(
        "--output-dir", default="out", help="Directory to save output meshes"
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    mesh = load_obj(args.mesh)
    target = args.target

    print("=== Stiffness Quadric Simplification ===")
    print(f"  Input: {mesh.n_verts} verts, {mesh.n_faces} faces")
    print(f"  Target: {target} verts, thickness: {args.thickness}\n")

    # --- QEM baseline ---
    t0 = time.time()
    result_qem = simplify_qem(mesh, target_verts=target, use_line_quadric=True)
    dt_qem = time.time() - t0
    areas_qem = face_areas(result_qem)
    print(
        f"  QEM:                    {result_qem.n_verts}v, {result_qem.n_faces}f, "
        f"time={dt_qem:.2f}s, min_area={areas_qem.min():.6f}"
    )

    # --- Stiffness only (additive) ---
    t0 = time.time()
    result_add = simplify_stiffness_quadric(
        mesh, target_verts=target, mode="stiffness",
        schur_mode="additive", thickness=args.thickness, verbose=False,
    )
    dt_add = time.time() - t0
    areas_add = face_areas(result_add)
    print(
        f"  Stiffness (additive):   {result_add.n_verts}v, {result_add.n_faces}f, "
        f"time={dt_add:.2f}s, min_area={areas_add.min():.6f}"
    )

    # --- Stiffness only (pure Schur) ---
    t0 = time.time()
    result_schur = simplify_stiffness_quadric(
        mesh, target_verts=target, mode="stiffness",
        schur_mode="schur", thickness=args.thickness, verbose=False,
    )
    dt_schur = time.time() - t0
    areas_schur = face_areas(result_schur)
    print(
        f"  Stiffness (Schur):      {result_schur.n_verts}v, {result_schur.n_faces}f, "
        f"time={dt_schur:.2f}s, min_area={areas_schur.min():.6f}"
    )

    # --- Combined (additive) ---
    t0 = time.time()
    result_combined = simplify_stiffness_quadric(
        mesh, target_verts=target, mode="combined",
        schur_mode="additive", thickness=args.thickness,
        use_line_quadric=True, verbose=False,
    )
    dt_combined = time.time() - t0
    areas_combined = face_areas(result_combined)
    print(
        f"  Combined (additive):    {result_combined.n_verts}v, {result_combined.n_faces}f, "
        f"time={dt_combined:.2f}s, min_area={areas_combined.min():.6f}"
    )

    # --- Save outputs ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.mesh).stem

    save_obj(result_qem, str(out_dir / f"{stem}_qem_{target}.obj"))
    save_obj(result_add, str(out_dir / f"{stem}_stiff_additive_{target}.obj"))
    save_obj(result_schur, str(out_dir / f"{stem}_stiff_schur_{target}.obj"))
    save_obj(result_combined, str(out_dir / f"{stem}_combined_{target}.obj"))

    print(f"\n  Saved to {out_dir}/")

    # --- Quality comparison ---
    print("\n  Quality comparison:")
    print(f"    {'Method':<26s} {'Area ratio':>12s} {'Aspect (mean)':>14s} {'Aspect (max)':>13s}")
    print(f"    {'-'*26} {'-'*12} {'-'*14} {'-'*13}")

    for name, r in [
        ("QEM", result_qem),
        ("Stiffness (additive)", result_add),
        ("Stiffness (Schur)", result_schur),
        ("Combined (additive)", result_combined),
    ]:
        a = face_areas(r)
        v, f = r.vertices, r.faces
        edges = np.stack([
            np.linalg.norm(v[f[:,1]] - v[f[:,0]], axis=1),
            np.linalg.norm(v[f[:,2]] - v[f[:,1]], axis=1),
            np.linalg.norm(v[f[:,0]] - v[f[:,2]], axis=1),
        ], axis=1)
        aspect = edges.max(axis=1) / (edges.min(axis=1) + 1e-30)
        print(f"    {name:<26s} {a.max()/a.min():12.0f} {aspect.mean():14.1f} {aspect.max():13.1f}")

    if args.smoke:
        print("\n--- Smoke checks ---")
        for r in [result_qem, result_add, result_schur, result_combined]:
            assert r.n_verts == target
            assert np.all(face_areas(r) > 0)
            f = r.faces
            assert not np.any((f[:,0]==f[:,1])|(f[:,1]==f[:,2])|(f[:,0]==f[:,2]))
        print("  All checks PASS")
        print("\nSMOKE: PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        ps.set_front_dir("neg_z_front")

        bbox = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        spacing = bbox[0] * 1.3

        meshes = [
            ("original", mesh, colors.RENDER_COLORS["gray"]),
            (f"QEM ({result_qem.n_verts}v)", result_qem, colors.get_color_by_index(0)),
            (f"Stiff Schur ({result_schur.n_verts}v)", result_schur, colors.get_color_by_index(2)),
            (f"Stiff additive ({result_add.n_verts}v)", result_add, colors.get_color_by_index(1)),
            (f"Combined ({result_combined.n_verts}v)", result_combined, colors.get_color_by_index(3)),
        ]

        for i, (name, m, color) in enumerate(meshes):
            verts = m.vertices.copy()
            verts[:, 0] += i * spacing
            ps.register_surface_mesh(name, verts, m.faces, edge_width=1.0, color=color)

        print("\n  Polyscope open. Left to right:")
        print(f"    Original ({mesh.n_verts}v)")
        print(f"    QEM — {dt_qem:.2f}s")
        print(f"    Stiffness (additive) — {dt_add:.2f}s")
        print(f"    Stiffness (Schur) — {dt_schur:.2f}s")
        print(f"    Combined (additive) — {dt_combined:.2f}s")
        ps.show()


if __name__ == "__main__":
    main()
