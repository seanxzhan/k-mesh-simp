"""Schur Complement Cost Visualization — Seeing the Math.

Visualizes two stiffness-based edge collapse cost metrics:

1. Simple trace metric (Schur flow per vertex):
   flow(v) = trace(K_vv^{-1} * S), S = sum of coupling products through v.
   "How much stiffness flows THROUGH this vertex?"

2. Full Schur mismatch (per edge):
   ||K_collapsed - K_schur||^2_F on the affected patch.
   "How much mechanical info is lost by the topology change vs. ideal condensation?"

Displays multiple scalar quantities on the mesh in Polyscope:
  - Diagonal stiffness (reference: how stiff IS this vertex)
  - Schur flow (how important is this vertex as a conduit)
  - Min edge cost (simple) per vertex
  - Min edge cost (full) per vertex

Run:
  python scripts/schur_cost.py                        # spot.obj, interactive
  python scripts/schur_cost.py --mesh data/spot.obj   # custom mesh
  python scripts/schur_cost.py --smoke                # headless verification
"""

import argparse
import time

import numpy as np

from kms.mesh import TriMesh, load_obj, make_grid
from kms.adjacency import MeshAdjacency
from kms.stiffness import shell_stiffness
from kms.schur import per_vertex_schur_flow, edge_cost_simple, edge_cost_full, per_edge_costs_full


def per_vertex_diagonal_stiffness(K, n_verts: int) -> np.ndarray:
    diag = K.diagonal()
    return diag.reshape(n_verts, 3).sum(axis=1)


def edges_to_vertex_min(edges, costs, n_verts: int) -> np.ndarray:
    """Map per-edge costs to per-vertex by taking min of incident edges."""
    vert_min = np.full(n_verts, np.inf)
    for (u, v), c in zip(edges, costs):
        if c < vert_min[u]:
            vert_min[u] = c
        if c < vert_min[v]:
            vert_min[v] = c
    # Replace inf with max finite value for visualization
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

    print("=== Schur Complement Cost Visualization ===")
    print(f"  Mesh: {n} verts, {mesh.n_faces} faces")

    # Build stiffness matrix
    t0 = time.time()
    K, K_m, K_b = shell_stiffness(mesh)
    print(f"  Stiffness assembled: {time.time() - t0:.2f}s")

    # 1. Diagonal stiffness (baseline)
    diag_stiff = per_vertex_diagonal_stiffness(K, n)
    print(f"\n  Diagonal stiffness: min={diag_stiff.min():.2e}, max={diag_stiff.max():.2e}")

    # 2. Schur flow (simple metric)
    t0 = time.time()
    flow = per_vertex_schur_flow(K, mesh)
    print(f"  Schur flow computed: {time.time() - t0:.2f}s")
    print(f"  Schur flow: min={flow.min():.2e}, max={flow.max():.2e}, mean={flow.mean():.2e}")

    # 3. Per-edge simple cost
    t0 = time.time()
    edges_s, costs_s = edge_cost_simple(K, mesh)
    print(f"  Simple edge costs: {time.time() - t0:.2f}s ({len(edges_s)} edges)")
    finite_s = costs_s[np.isfinite(costs_s)]
    if len(finite_s) > 0:
        print(f"  Simple costs: min={finite_s.min():.2e}, max={finite_s.max():.2e}")

    # 4. Per-edge full Schur mismatch
    t0 = time.time()
    edges_f, costs_f = per_edge_costs_full(mesh, verbose=True)
    dt_full = time.time() - t0
    print(f"  Full edge costs: {dt_full:.2f}s ({len(edges_f)} edges)")
    finite_f = costs_f[np.isfinite(costs_f)]
    if len(finite_f) > 0:
        print(f"  Full costs: min={finite_f.min():.2e}, max={finite_f.max():.2e}")

    # Map to per-vertex (min of incident edges)
    vert_min_simple = edges_to_vertex_min(edges_s, costs_s, n)
    vert_min_full = edges_to_vertex_min(edges_f, costs_f, n)

    # 5. Ratio: where do the metrics disagree?
    eps = 1e-30
    ratio = (vert_min_full + eps) / (vert_min_simple + eps)

    if args.smoke:
        print("\n--- Smoke checks ---")
        assert np.all(flow >= -1e-10), "Schur flow has negative values"
        assert np.all(finite_s >= -1e-10), "Simple costs have negative values"
        assert np.all(finite_f >= -1e-10), "Full costs have negative values"
        assert len(edges_s) == len(edges_f), "Edge lists differ"
        assert diag_stiff.max() > 0, "Diagonal stiffness is zero"
        assert flow.max() > 0, "Schur flow is zero everywhere"
        print("  All checks PASS")
        print("\nSMOKE: PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        ps.set_front_dir("neg_z_front")

        # Compute bounding box width for spacing
        bbox = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        spacing = bbox[0] * 1.3

        quantities = [
            ("diagonal stiffness", diag_stiff,
             "how stiff IS each vertex (high = stiff)"),
            ("Schur flow", flow,
             "stiffness flowing THROUGH vertex (high = important conduit)"),
            ("min edge cost (simple)", vert_min_simple,
             "cheapest collapse at vertex (low = collapse first)"),
            ("min edge cost (full)", vert_min_full,
             "cheapest collapse, full mismatch (low = collapse first)"),
            ("log ratio (full/simple)", np.log10(ratio + 1e-30),
             "positive = full > simple, negative = full < simple, 0 = agree"),
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
            )

        print("\n  Polyscope open. Side-by-side meshes (left to right):")
        for i, (name, _, desc) in enumerate(quantities):
            print(f"    {i+1}. {name} — {desc}")
        ps.show()


if __name__ == "__main__":
    main()
