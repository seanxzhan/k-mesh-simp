"""QEM Simplification — Quadric Error Metrics visualization.

Shows the original mesh alongside progressively simplified versions
using Polyscope for interactive 3D inspection.

Run:
  python scripts/qem_simplification.py                        # interactive (spot.obj)
  python scripts/qem_simplification.py --mesh data/spot.obj   # custom mesh
  python scripts/qem_simplification.py --smoke                # headless verification
"""

import argparse
import time
from pathlib import Path

import numpy as np

from kms.mesh import load_obj, face_areas
from kms.simplify_qem import simplify_qem
from kms import colors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default="data/spot.obj", help="Path to .obj mesh file")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--use-line-quadrics", action="store_true")
    args = parser.parse_args()

    mesh = load_obj(args.mesh)

    print("=== QEM Simplification ===")
    print(f"  Input: {mesh.n_verts} verts, {mesh.n_faces} faces")

    targets = [1000, 500, 200]
    results = []

    for target in targets:
        t0 = time.time()
        simplified = simplify_qem(
            mesh,
            target_verts=target,
            use_optimal_position=True,
            use_line_quadric=args.use_line_quadrics,
        )
        dt = time.time() - t0
        areas = face_areas(simplified)
        min_area = areas.min()
        mean_area = areas.mean()
        results.append((target, simplified, dt, min_area, mean_area))
        print(f"\n  Target {target} verts:")
        print(f"    Result: {simplified.n_verts} verts, {simplified.n_faces} faces")
        print(f"    Time: {dt:.2f}s")
        print(f"    Face areas: min={min_area:.6f}, mean={mean_area:.6f}")

    if args.smoke:
        for target, simplified, dt, min_area, _ in results:
            assert simplified.n_verts == target, f"Expected {target}, got {simplified.n_verts}"
            assert min_area > 0, f"Degenerate face at target={target}"
            f = simplified.faces
            assert not np.any(
                (f[:, 0] == f[:, 1])
                | (f[:, 1] == f[:, 2])
                | (f[:, 0] == f[:, 2])
            )
        assert results[-1][2] < 5.0, "Too slow (>5s for 200 verts target)"
        print("\nSMOKE: PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        ps.set_front_dir("neg_z_front")

        ps.register_surface_mesh(
            "original",
            mesh.vertices,
            mesh.faces,
            edge_width=1.0,
            color=colors.RENDER_COLORS["gray"],
        )

        for i, (target, simplified, _, _, _) in enumerate(results):
            verts = simplified.vertices.copy()
            verts[:, 0] += (i + 1) * 2.1
            ps.register_surface_mesh(
                f"QEM {target}v",
                verts,
                simplified.faces,
                edge_width=1.0,
                color=colors.get_color_by_index(i),
            )

        ps.show()


if __name__ == "__main__":
    main()
