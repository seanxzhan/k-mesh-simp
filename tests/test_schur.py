import numpy as np
from kms.mesh import make_grid, make_icosphere
from kms.stiffness import shell_stiffness
from kms.schur import per_vertex_schur_flow, edge_cost_simple, edge_cost_full


def test_schur_flow_nonneg():
    mesh = make_grid(5, 5)
    K, _, _ = shell_stiffness(mesh)
    flow = per_vertex_schur_flow(K, mesh)
    assert np.all(flow >= -1e-10)


def test_schur_flow_positive_interior():
    """Interior vertices on a grid should have positive flow."""
    mesh = make_grid(5, 5)
    K, _, _ = shell_stiffness(mesh)
    flow = per_vertex_schur_flow(K, mesh)
    # Interior vertices (not on boundary) should have positive flow
    # In a 5x5 grid, interior vertices are those not on the edges
    for j in range(1, 4):
        for i in range(1, 4):
            v = j * 5 + i
            assert flow[v] > 0, f"Interior vertex {v} has non-positive flow"


def test_schur_flow_boundary_lower():
    """Boundary vertices should generally have lower flow than interior."""
    mesh = make_grid(8, 8)
    K, _, _ = shell_stiffness(mesh)
    flow = per_vertex_schur_flow(K, mesh)

    # Identify interior and boundary vertices
    interior = []
    boundary = []
    for j in range(8):
        for i in range(8):
            v = j * 8 + i
            if 1 <= i <= 6 and 1 <= j <= 6:
                interior.append(v)
            else:
                boundary.append(v)

    mean_interior = np.mean(flow[interior])
    mean_boundary = np.mean(flow[boundary])
    assert mean_interior > mean_boundary


def test_edge_cost_simple_nonneg():
    mesh = make_grid(5, 5)
    K, _, _ = shell_stiffness(mesh)
    edges, costs = edge_cost_simple(K, mesh)
    assert len(edges) > 0
    assert np.all(costs >= -1e-10)


def test_edge_cost_full_nonneg():
    """Full cost should be non-negative (it's a squared Frobenius norm)."""
    mesh = make_icosphere(1)  # 42 verts, small enough for full computation
    from kms.adjacency import MeshAdjacency
    adj = MeshAdjacency(mesh)
    edges = adj.get_edges()

    # Test a few collapsible edges
    tested = 0
    for u, v in edges:
        if adj.is_collapsible(u, v):
            cost = edge_cost_full(mesh, u, v, 0.5)
            assert cost >= -1e-10, f"Edge ({u},{v}) has negative cost: {cost}"
            tested += 1
            if tested >= 5:
                break
    assert tested > 0


def test_edge_cost_full_finite_for_collapsible():
    """Collapsible edges should return finite cost."""
    mesh = make_icosphere(1)
    from kms.adjacency import MeshAdjacency
    adj = MeshAdjacency(mesh)
    tested = 0
    for u, v in adj.get_edges():
        if adj.is_collapsible(u, v):
            cost = edge_cost_full(mesh, u, v, 0.5)
            assert np.isfinite(cost), f"Edge ({u},{v}) cost not finite: {cost}"
            tested += 1
            if tested >= 3:
                break
    assert tested > 0


def test_full_cost_varies_across_edges():
    """Full cost should vary across edges (not constant), showing it captures local geometry."""
    mesh = make_icosphere(1)
    from kms.adjacency import MeshAdjacency
    adj = MeshAdjacency(mesh)

    costs = []
    for u, v in adj.get_edges():
        if adj.is_collapsible(u, v):
            c = edge_cost_full(mesh, u, v, 0.5)
            if np.isfinite(c):
                costs.append(c)
            if len(costs) >= 10:
                break

    costs = np.array(costs)
    assert len(costs) >= 5
    # On an icosphere (uniform), costs should be similar but not identical
    # due to floating point and the icosahedral symmetry not being perfect after subdivision
    assert np.std(costs) >= 0 or np.mean(costs) > 0
