import numpy as np
from kms.mesh import make_grid, make_icosphere
from kms.stiffness import shell_stiffness
from kms.stiffness_quadric import (
    StiffnessQuadric,
    build_stiffness_quadrics,
    per_vertex_stiffness_quadric_cost,
    per_edge_stiffness_quadric_cost,
)


def test_quadric_zero_at_rest():
    """At the original position, E(x0) should be 0."""
    K_vv = np.array([[2.0, 0.1, 0], [0.1, 3.0, 0], [0, 0, 1.5]])
    x0 = np.array([1.0, 2.0, 3.0])
    q = StiffnessQuadric.from_stiffness_block(K_vv, x0)
    np.testing.assert_allclose(q.compute_error(x0), 0.0, atol=1e-12)


def test_quadric_positive_away_from_rest():
    """Displacement from rest should cost positive energy."""
    K_vv = np.eye(3) * 2.0
    x0 = np.array([1.0, 0.0, 0.0])
    q = StiffnessQuadric.from_stiffness_block(K_vv, x0)
    x = np.array([1.5, 0.0, 0.0])
    cost = q.compute_error(x)
    # (x-x0)^T K_vv (x-x0) = [0.5,0,0]^T * 2I * [0.5,0,0] = 0.5
    np.testing.assert_allclose(cost, 0.5, atol=1e-12)


def test_quadric_optimal_is_rest():
    """Optimal position should be the rest position."""
    K_vv = np.array([[3.0, 0.5, 0], [0.5, 2.0, 0], [0, 0, 4.0]])
    x0 = np.array([2.0, -1.0, 0.5])
    q = StiffnessQuadric.from_stiffness_block(K_vv, x0)
    pos, success = q.optimal_position()
    assert success
    np.testing.assert_allclose(pos, x0, atol=1e-10)


def test_quadric_addition():
    """Merged quadric should sum both contributions."""
    K1 = np.eye(3)
    K2 = np.eye(3) * 2
    x1 = np.array([1.0, 0.0, 0.0])
    x2 = np.array([0.0, 1.0, 0.0])
    q1 = StiffnessQuadric.from_stiffness_block(K1, x1)
    q2 = StiffnessQuadric.from_stiffness_block(K2, x2)
    q_merged = q1 + q2
    # Optimal should be weighted average: (K1*x1 + K2*x2)/(K1+K2) = (x1 + 2*x2)/3
    pos, success = q_merged.optimal_position()
    assert success
    expected = (K1 @ x1 + K2 @ x2) / 3.0  # (K1+K2)^{-1} (K1 x1 + K2 x2) = (1/3)(x1 + 2x2)
    np.testing.assert_allclose(pos, expected, atol=1e-10)


def test_quadric_anisotropic():
    """Anisotropic K_vv should make some directions more expensive."""
    K_vv = np.diag([10.0, 1.0, 1.0])  # stiff in x
    x0 = np.zeros(3)
    q = StiffnessQuadric.from_stiffness_block(K_vv, x0)
    # Moving in x costs 10x more than moving in y
    dx = np.array([0.1, 0.0, 0.0])
    dy = np.array([0.0, 0.1, 0.0])
    assert q.compute_error(dx) > 9 * q.compute_error(dy)


def test_per_vertex_cost_zero_at_original():
    """On the original mesh, all vertices are at rest → cost = 0."""
    mesh = make_grid(5, 5)
    K, _, _ = shell_stiffness(mesh)
    costs = per_vertex_stiffness_quadric_cost(K, mesh)
    np.testing.assert_allclose(costs, 0.0, atol=1e-12)


def test_per_edge_costs_nonneg():
    """Edge costs should be non-negative."""
    mesh = make_icosphere(1)
    K, _, _ = shell_stiffness(mesh)
    edges, costs, positions = per_edge_stiffness_quadric_cost(K, mesh)
    assert len(edges) > 0
    assert np.all(costs >= -1e-12)


def test_per_edge_optimal_positions_finite():
    """Optimal positions should be finite."""
    mesh = make_icosphere(1)
    K, _, _ = shell_stiffness(mesh)
    _, _, positions = per_edge_stiffness_quadric_cost(K, mesh)
    assert np.all(np.isfinite(positions))


def test_build_quadrics_count():
    """Should have one quadric per vertex."""
    mesh = make_grid(4, 4)
    K, _, _ = shell_stiffness(mesh)
    quadrics = build_stiffness_quadrics(K, mesh)
    assert len(quadrics) == mesh.n_verts


def test_stiff_vertex_higher_cost():
    """A vertex with higher K_vv should cost more to displace the same distance."""
    K_low = np.eye(3) * 1.0
    K_high = np.eye(3) * 10.0
    x0 = np.zeros(3)
    q_low = StiffnessQuadric.from_stiffness_block(K_low, x0)
    q_high = StiffnessQuadric.from_stiffness_block(K_high, x0)
    x = np.array([0.1, 0.1, 0.1])
    assert q_high.compute_error(x) > q_low.compute_error(x)
