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

    # --- QEM baseline (with skinning weights via restriction matrix) ---
    t0 = time.time()
    result_qem, P_qem = simplify_qem(mesh, target_verts=target, use_line_quadric=True, compute_restriction=True)
    dt_qem = time.time() - t0
    areas_qem = face_areas(result_qem)
    print(
        f"  QEM:              {result_qem.n_verts}v, {result_qem.n_faces}f, "
        f"time={dt_qem:.2f}s, min_area={areas_qem.min():.6f}"
    )

    # --- Stiffness quadric only (Approach 2) with skinning weights ---
    t0 = time.time()
    result_kq, W_kq = simplify_stiffness_quadric(
        mesh,
        target_verts=target,
        mode="stiffness",
        thickness=args.thickness,
        compute_skinning_weights=True,
        verbose=False,
    )
    dt_kq = time.time() - t0
    areas_kq = face_areas(result_kq)
    print(
        f"  Stiffness only:   {result_kq.n_verts}v, {result_kq.n_faces}f, "
        f"time={dt_kq:.2f}s, min_area={areas_kq.min():.6f}"
    )

    # --- Combined QEM + Stiffness (Approach 3) with skinning weights ---
    t0 = time.time()
    result_combined, W_combined = simplify_stiffness_quadric(
        mesh,
        target_verts=target,
        mode="combined",
        thickness=args.thickness,
        use_line_quadric=True,
        compute_skinning_weights=True,
        verbose=True,
    )
    dt_combined = time.time() - t0
    areas_combined = face_areas(result_combined)
    print(
        f"  Combined (QEM+K): {result_combined.n_verts}v, {result_combined.n_faces}f, "
        f"time={dt_combined:.2f}s, min_area={areas_combined.min():.6f}"
    )

    # --- Save outputs ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.mesh).stem

    save_obj(result_qem, str(out_dir / f"{stem}_qem_{target}.obj"))
    save_obj(result_kq, str(out_dir / f"{stem}_stiffquad_{target}.obj"))
    save_obj(result_combined, str(out_dir / f"{stem}_combined_{target}.obj"))

    print(f"\n  Saved to {out_dir}/:")
    print(f"    {stem}_qem_{target}.obj")
    print(f"    {stem}_stiffquad_{target}.obj")
    print(f"    {stem}_combined_{target}.obj")

    # --- Quality comparison ---
    print("\n  Quality comparison:")
    print(
        f"    {'Method':<22s} {'Area ratio':>12s} {'Aspect (mean)':>14s} {'Aspect (max)':>13s}"
    )
    print(f"    {'-'*22} {'-'*12} {'-'*14} {'-'*13}")

    for name, r in [
        ("QEM", result_qem),
        ("Stiffness only", result_kq),
        ("Combined (QEM+K)", result_combined),
    ]:
        a = face_areas(r)
        v, f = r.vertices, r.faces
        edges = np.stack(
            [
                np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1),
                np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1),
                np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1),
            ],
            axis=1,
        )
        aspect = edges.max(axis=1) / (edges.min(axis=1) + 1e-30)
        print(
            f"    {name:<22s} {a.max()/a.min():12.0f} {aspect.mean():14.1f} {aspect.max():13.1f}"
        )

    if args.smoke:
        print("\n--- Smoke checks ---")
        assert result_qem.n_verts == target
        assert result_kq.n_verts == target
        assert result_combined.n_verts == target
        assert np.all(face_areas(result_qem) > 0)
        assert np.all(face_areas(result_kq) > 0)
        assert np.all(face_areas(result_combined) > 0)
        for r in [result_kq, result_combined]:
            f = r.faces
            assert not np.any(
                (f[:, 0] == f[:, 1])
                | (f[:, 1] == f[:, 2])
                | (f[:, 0] == f[:, 2])
            )
        assert W_kq.shape == (mesh.n_verts, result_kq.n_verts)
        assert W_combined.shape == (mesh.n_verts, result_combined.n_verts)
        print("  All checks PASS")
        print("\nSMOKE: PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        ps.set_front_dir("neg_z_front")

        bbox = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        spacing_x = bbox[0] * 1.3
        spacing_y = bbox[1] * 1.5

        # --- Top row: coarse meshes ---
        meshes = [
            ("original", mesh, colors.RENDER_COLORS["gray"]),
            (f"QEM ({result_qem.n_verts}v)", result_qem, colors.get_color_by_index(0)),
            (f"Stiffness only ({result_kq.n_verts}v)", result_kq, colors.get_color_by_index(1)),
            (f"Combined QEM+K ({result_combined.n_verts}v)", result_combined, colors.get_color_by_index(2)),
        ]

        for i, (name, m, color) in enumerate(meshes):
            verts = m.vertices.copy()
            verts[:, 0] += i * spacing_x
            ps.register_surface_mesh(name, verts, m.faces, edge_width=1.0, color=color)

        # --- Bottom row: fine mesh painted with skinning weights ---
        # For QEM: P is (n_coarse, n_fine), transpose gives (n_fine, n_coarse) weights
        W_qem = P_qem.T.toarray()

        weight_data = [
            ("QEM weights", W_qem),
            ("Stiffness weights", W_kq.toarray()),
            ("Combined weights", W_combined.toarray()),
        ]

        for i, (name, W_dense) in enumerate(weight_data):
            n_coarse = W_dense.shape[1]

            # Paint each fine vertex with the color of its dominant coarse vertex
            dominant = np.argmax(W_dense, axis=1)

            # Assign RGB per fine vertex from Tableau20 (cycling over coarse vert indices)
            vertex_colors = np.zeros((mesh.n_verts, 3))
            for vi in range(mesh.n_verts):
                vertex_colors[vi] = colors.get_color_by_index(dominant[vi])

            verts = mesh.vertices.copy()
            verts[:, 0] += (i + 1) * spacing_x
            verts[:, 1] -= spacing_y

            ps_w = ps.register_surface_mesh(
                f"{name} (fine)", verts, mesh.faces, edge_width=0.3
            )
            ps_w.add_color_quantity(
                "dominant weight color", vertex_colors, defined_on="vertices", enabled=True
            )

            # Max weight scalar: shows how "sharp" the skinning is
            max_weight = W_dense[np.arange(mesh.n_verts), dominant]
            ps_w.add_scalar_quantity(
                "max weight", max_weight, defined_on="vertices", cmap="viridis"
            )

        print("\n  Polyscope open.")
        print("  Top row: Original, QEM, Stiffness only, Combined")
        print("  Bottom row: Fine mesh painted by dominant skinning weight")
        print(f"    QEM ({result_qem.n_verts}v) — {dt_qem:.2f}s")
        print(f"    Stiffness only ({result_kq.n_verts}v) — {dt_kq:.2f}s")
        print(f"    Combined QEM+K ({result_combined.n_verts}v) — {dt_combined:.2f}s")
        ps.show()


if __name__ == "__main__":
    main()
