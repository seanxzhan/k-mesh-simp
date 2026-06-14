"""Schur Complement Simplification — Compare all modes.

Simplifies a mesh using QEM, Schur flow, QEM+flow, and Schur full,
displaying results side-by-side for comparison.

Run:
  python scripts/schur_simp.py                        # spot.obj, interactive
  python scripts/schur_simp.py --mesh data/spot.obj   # custom mesh
  python scripts/schur_simp.py --target 500           # custom target
  python scripts/schur_simp.py --smoke                # headless verification
"""

import argparse
import time
from pathlib import Path

import numpy as np

from kms.mesh import load_obj, save_obj, face_areas
from kms.simplify_qem import simplify_qem
from kms.simplify_schur import simplify_schur
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
        "--output-dir", default="out", help="Directory to save output meshes"
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    mesh = load_obj(args.mesh)
    target = args.target

    print("=== Schur Complement Simplification ===")
    print(f"  Input: {mesh.n_verts} verts, {mesh.n_faces} faces")
    print(f"  Target: {target} verts\n")

    # --- QEM baseline ---
    t0 = time.time()
    result_qem = simplify_qem(mesh, target_verts=target, use_line_quadric=True)
    dt_qem = time.time() - t0
    areas_qem = face_areas(result_qem)
    print(
        f"  QEM:        {result_qem.n_verts}v, {result_qem.n_faces}f, "
        f"time={dt_qem:.2f}s, min_area={areas_qem.min():.6f}"
    )

    # --- QEM + flow ---
    t0 = time.time()
    result_qf = simplify_schur(
        mesh,
        target_verts=target,
        mode="qem+flow",
        use_line_quadric=True,
        thickness=0.1,
        verbose=False,
    )
    dt_qf = time.time() - t0
    areas_qf = face_areas(result_qf)
    print(
        f"  QEM+flow:   {result_qf.n_verts}v, {result_qf.n_faces}f, "
        f"time={dt_qf:.2f}s, min_area={areas_qf.min():.6f}"
    )

    # --- Schur flow ---
    t0 = time.time()
    result_flow = simplify_schur(
        mesh, target_verts=target, mode="flow", thickness=0.1, verbose=False
    )
    dt_flow = time.time() - t0
    areas_flow = face_areas(result_flow)
    print(
        f"  Schur flow: {result_flow.n_verts}v, {result_flow.n_faces}f, "
        f"time={dt_flow:.2f}s, min_area={areas_flow.min():.6f}"
    )

    # --- Schur full ---
    do_full = False
    if do_full:
        t0 = time.time()
        result_full = simplify_schur(
            mesh, target_verts=target, mode="full", thickness=0.1, verbose=True
        )
        dt_full = time.time() - t0
        areas_full = face_areas(result_full)
        print(
            f"  Schur full: {result_full.n_verts}v, {result_full.n_faces}f, "
            f"time={dt_full:.2f}s, min_area={areas_full.min():.6f}"
        )
    else:
        print(f"  Schur full: SKIPPED (mesh too large: {mesh.n_verts} verts)")
        result_full = None

    # --- Save outputs ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.mesh).stem

    save_obj(result_qem, str(out_dir / f"{stem}_qem_{target}.obj"))
    save_obj(result_qf, str(out_dir / f"{stem}_qem_flow_{target}.obj"))
    save_obj(result_flow, str(out_dir / f"{stem}_flow_{target}.obj"))
    if result_full is not None:
        save_obj(result_full, str(out_dir / f"{stem}_full_{target}.obj"))

    print(f"\n  Saved to {out_dir}/:")
    print(f"    {stem}_qem_{target}.obj")
    print(f"    {stem}_qem_flow_{target}.obj")
    print(f"    {stem}_flow_{target}.obj")
    if result_full is not None:
        print(f"    {stem}_full_{target}.obj")

    if args.smoke:
        print("\n--- Smoke checks ---")
        assert result_qem.n_verts == target
        assert result_flow.n_verts == target
        assert result_qf.n_verts == target
        assert np.all(face_areas(result_qem) > 0)
        assert np.all(face_areas(result_flow) > 0)
        assert np.all(face_areas(result_qf) > 0)

        for label, r in [("flow", result_flow), ("qem+flow", result_qf)]:
            f = r.faces
            assert not np.any(
                (f[:, 0] == f[:, 1])
                | (f[:, 1] == f[:, 2])
                | (f[:, 0] == f[:, 2])
            ), f"{label} degenerate"

        if result_full is not None:
            assert result_full.n_verts == target
            assert np.all(face_areas(result_full) > 0)
            f = result_full.faces
            assert not np.any(
                (f[:, 0] == f[:, 1])
                | (f[:, 1] == f[:, 2])
                | (f[:, 0] == f[:, 2])
            )

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

        # Original
        ps.register_surface_mesh(
            "original",
            mesh.vertices,
            mesh.faces,
            edge_width=1.0,
            color=colors.RENDER_COLORS["gray"],
        )

        # QEM
        verts = result_qem.vertices.copy()
        verts[:, 0] += spacing
        ps_qem = ps.register_surface_mesh(
            f"QEM ({result_qem.n_verts}v)",
            verts,
            result_qem.faces,
            edge_width=1.0,
            color=colors.get_color_by_index(0),
        )

        # QEM + flow
        verts = result_qf.vertices.copy()
        verts[:, 0] += 2 * spacing
        ps.register_surface_mesh(
            f"QEM+flow ({result_qf.n_verts}v)",
            verts,
            result_qf.faces,
            edge_width=1.0,
            color=colors.get_color_by_index(1),
        )

        # Schur flow
        verts = result_flow.vertices.copy()
        verts[:, 0] += 3 * spacing
        ps.register_surface_mesh(
            f"Schur flow ({result_flow.n_verts}v)",
            verts,
            result_flow.faces,
            edge_width=1.0,
            color=colors.get_color_by_index(2),
        )

        # Schur full (if computed)
        if result_full is not None:
            verts = result_full.vertices.copy()
            verts[:, 0] += 4 * spacing
            ps.register_surface_mesh(
                f"Schur full ({result_full.n_verts}v)",
                verts,
                result_full.faces,
                edge_width=1.0,
                color=colors.get_color_by_index(3),
            )

        print("\n  Polyscope open. Left to right:")
        print(f"    Original ({mesh.n_verts}v)")
        print(f"    QEM ({result_qem.n_verts}v) — {dt_qem:.2f}s")
        print(f"    QEM+flow ({result_qf.n_verts}v) — {dt_qf:.2f}s")
        print(f"    Schur flow ({result_flow.n_verts}v) — {dt_flow:.2f}s")
        if result_full is not None:
            print(f"    Schur full ({result_full.n_verts}v) — {dt_full:.2f}s")
        ps.show()


if __name__ == "__main__":
    main()
