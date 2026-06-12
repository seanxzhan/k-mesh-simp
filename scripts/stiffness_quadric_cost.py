"""Stiffness Quadric Cost Visualization — Approach 2.

Visualizes the stiffness quadric cost metric:
  E(x) = (x - x₀)^T K_vv (x - x₀)

Shows per-vertex and per-edge quantities compared to QEM.

Run:
  python scripts/stiffness_quadric_cost.py                        # spot.obj
  python scripts/stiffness_quadric_cost.py --mesh data/spot.obj   # custom mesh
  python scripts/stiffness_quadric_cost.py --smoke                # headless
"""

import argparse
import time

import numpy as np

from kms.mesh import load_obj, face_areas
from kms.adjacency import MeshAdjacency
from kms.stiffness import shell_stiffness
from kms.stiffness_quadric import (
    build_stiffness_quadrics,
    per_vertex_stiffness_quadric_cost,
    per_edge_stiffness_quadric_cost,
)
from kms.quadrics import Quadric


def per_vertex_diagonal_stiffness(K, n_verts: int) -> np.ndarray:
    diag = K.diagonal()
    return diag.reshape(n_verts, 3).sum(axis=1)


def compute_qem_edge_costs(mesh):
    """Compute QEM edge costs for comparison."""
    adj = MeshAdjacency(mesh)
    areas = face_areas(mesh)

    face_quadrics = {}
    for fi in range(mesh.n_faces):
        a, b, c = adj.get_face_vertices(fi)
        face_quadrics[fi] = Quadric.from_triangle(
            mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
        )

    vertex_quadrics = {}
    for vi in range(mesh.n_verts):
        fq = [face_quadrics[fi] for fi in adj.vert_faces[vi]]
        fa = [areas[fi] for fi in adj.vert_faces[vi]]
        vertex_quadrics[vi] = Quadric.vertex_quadric(fq, fa)

    edges = adj.get_edges()
    costs = np.zeros(len(edges))
    for ei, (u, v) in enumerate(edges):
        eq = Quadric.edge_quadric(vertex_quadrics[u], vertex_quadrics[v])
        pos, success = eq.optimal_position()
        if not success:
            c_u = eq.compute_error(mesh.vertices[u])
            c_v = eq.compute_error(mesh.vertices[v])
            pos = mesh.vertices[u] if c_u <= c_v else mesh.vertices[v]
        costs[ei] = eq.compute_error(pos)
    return edges, costs


def edges_to_vertex_min(edges, costs, n_verts: int) -> np.ndarray:
    vert_min = np.full(n_verts, np.inf)
    for (u, v), c in zip(edges, costs):
        if c < vert_min[u]:
            vert_min[u] = c
        if c < vert_min[v]:
            vert_min[v] = c
    finite_mask = np.isfinite(vert_min)
    if np.any(finite_mask):
        vert_min[~finite_mask] = np.max(vert_min[finite_mask])
    else:
        vert_min[:] = 0
    return vert_min


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", default="data/spot.obj", help="Path to .obj mesh file")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    mesh = load_obj(args.mesh)
    n = mesh.n_verts

    print("=== Stiffness Quadric Cost Visualization ===")
    print(f"  Mesh: {n} verts, {mesh.n_faces} faces")

    # Build stiffness matrix
    t0 = time.time()
    K, _, _ = shell_stiffness(mesh)
    print(f"  Stiffness assembled: {time.time() - t0:.2f}s")

    # 1. Diagonal stiffness (K_vv trace per vertex)
    diag_stiff = per_vertex_diagonal_stiffness(K, n)
    print(f"\n  K_vv diagonal sum: min={diag_stiff.min():.2e}, max={diag_stiff.max():.2e}")

    # 2. Per-edge stiffness quadric cost
    t0 = time.time()
    edges_kq, costs_kq, positions_kq = per_edge_stiffness_quadric_cost(K, mesh)
    print(f"  Stiffness quadric edge costs: {time.time() - t0:.2f}s ({len(edges_kq)} edges)")
    print(f"  Costs: min={costs_kq.min():.2e}, max={costs_kq.max():.2e}")

    # 3. Per-edge QEM cost (for comparison)
    t0 = time.time()
    edges_qem, costs_qem = compute_qem_edge_costs(mesh)
    print(f"  QEM edge costs: {time.time() - t0:.2f}s ({len(edges_qem)} edges)")
    print(f"  Costs: min={costs_qem.min():.2e}, max={costs_qem.max():.2e}")

    # Map to per-vertex min
    vert_min_kq = edges_to_vertex_min(edges_kq, costs_kq, n)
    vert_min_qem = edges_to_vertex_min(edges_qem, costs_qem, n)

    # 4. Optimal position displacement: how far does the stiffness quadric
    #    place the merged vertex from the original positions?
    adj = MeshAdjacency(mesh)
    edges_list = adj.get_edges()
    displacement = np.zeros(len(edges_list))
    for ei, (u, v) in enumerate(edges_list):
        midpoint = 0.5 * (mesh.vertices[u] + mesh.vertices[v])
        displacement[ei] = np.linalg.norm(positions_kq[ei] - midpoint)
    vert_displacement = edges_to_vertex_min(edges_list, -displacement, n)
    vert_displacement = -vert_displacement  # flip sign since we used min of negatives

    # 5. Rank correlation between stiffness quadric and QEM costs
    from scipy.stats import spearmanr
    corr, pval = spearmanr(costs_kq, costs_qem)
    print(f"\n  Rank correlation (stiffness quadric vs QEM): {corr:.3f} (p={pval:.2e})")

    if args.smoke:
        print("\n--- Smoke checks ---")
        assert np.all(costs_kq >= -1e-12), "Stiffness quadric costs negative"
        assert np.all(costs_qem >= -1e-12), "QEM costs negative"
        assert np.all(np.isfinite(positions_kq)), "Non-finite optimal positions"
        assert len(edges_kq) == len(edges_qem), "Edge count mismatch"
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

        quantities = [
            ("K_vv diagonal (self-stiffness)", diag_stiff,
             "how stiff is each vertex"),
            ("min edge cost (stiffness quadric)", vert_min_kq,
             "cheapest collapse via stiffness quadric (low = collapse first)"),
            ("min edge cost (QEM)", vert_min_qem,
             "cheapest collapse via QEM (low = collapse first)"),
            ("optimal position displacement", vert_displacement,
             "how far stiffness quadric moves the vertex from midpoint"),
        ]

        for i, (name, values, _) in enumerate(quantities):
            verts = mesh.vertices.copy()
            verts[:, 0] += i * spacing
            ps_mesh = ps.register_surface_mesh(
                name, verts, mesh.faces, edge_width=0.5
            )
            ps_mesh.add_scalar_quantity(
                "value",
                values,
                defined_on="vertices",
                enabled=True,
                cmap="coolwarm",
                vminmax=(float(values.min()), float(values.max())),
            )

        print(f"\n  Polyscope open. Side-by-side (left to right):")
        for i, (name, _, desc) in enumerate(quantities):
            print(f"    {i+1}. {name} — {desc}")
        print(f"\n  Rank correlation (stiffness quadric vs QEM): {corr:.3f}")
        ps.show()


if __name__ == "__main__":
    main()
