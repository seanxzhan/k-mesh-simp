import numpy as np
from kms.quadrics import Quadric


def test_plane_error_zero_on_plane():
    normal = np.array([0.0, 0.0, 1.0])
    point = np.array([1.0, 2.0, 3.0])
    q = Quadric.from_plane(normal, point)
    np.testing.assert_allclose(q.compute_error(np.array([5.0, -1.0, 3.0])), 0.0, atol=1e-12)


def test_plane_error_positive_off_plane():
    normal = np.array([0.0, 0.0, 1.0])
    point = np.array([0.0, 0.0, 0.0])
    q = Quadric.from_plane(normal, point)
    np.testing.assert_allclose(q.compute_error(np.array([0.0, 0.0, 2.0])), 4.0, atol=1e-12)


def test_triangle_quadric():
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    q = Quadric.from_triangle(v1, v2, v3)
    np.testing.assert_allclose(q.compute_error(np.array([0.5, 0.5, 0.0])), 0.0, atol=1e-12)
    assert q.compute_error(np.array([0.0, 0.0, 1.0])) > 0


def test_optimal_position_single_plane():
    q = Quadric.from_plane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 5.0]))
    _, success = q.optimal_position()
    assert not success


def test_optimal_position_three_planes():
    q1 = Quadric.from_plane(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    q2 = Quadric.from_plane(np.array([0.0, 1.0, 0.0]), np.array([0.0, 2.0, 0.0]))
    q3 = Quadric.from_plane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 3.0]))
    q = q1 + q2 + q3
    pos, success = q.optimal_position()
    assert success
    np.testing.assert_allclose(pos, [1.0, 2.0, 3.0], atol=1e-10)


def test_addition():
    q1 = Quadric(np.eye(3), np.ones(3), 1.0)
    q2 = Quadric(np.eye(3) * 2, np.ones(3) * 3, 5.0)
    q3 = q1 + q2
    np.testing.assert_array_equal(q3.A, np.eye(3) * 3)
    np.testing.assert_array_equal(q3.b, np.ones(3) * 4)
    assert q3.c == 6.0


def test_scalar_multiply():
    q = Quadric(np.eye(3), np.ones(3), 2.0)
    q2 = q * 3.0
    np.testing.assert_array_equal(q2.A, np.eye(3) * 3)
    np.testing.assert_array_equal(q2.b, np.ones(3) * 3)
    assert q2.c == 6.0


def test_boundary_edge_quadric():
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    face_normal = np.array([0.0, 0.0, 2.0])  # +z (unnormalized is fine)
    q = Quadric.from_boundary_edge(v1, v2, face_normal, weight=1.0)
    # plane contains the edge and the face normal (here: the y = 0 plane), so the
    # endpoints and any off-surface (+z) point have zero error
    np.testing.assert_allclose(q.compute_error(v1), 0.0, atol=1e-12)
    np.testing.assert_allclose(q.compute_error(v2), 0.0, atol=1e-12)
    np.testing.assert_allclose(q.compute_error(np.array([0.5, 0.0, 9.0])), 0.0, atol=1e-12)
    # moving off the boundary line (in-surface, perpendicular to the edge) is penalized
    assert q.compute_error(np.array([0.5, 1.0, 0.0])) > 0.5
    # weight scales the penalty
    off = np.array([0.0, 1.0, 0.0])
    q3 = Quadric.from_boundary_edge(v1, v2, face_normal, weight=3.0)
    np.testing.assert_allclose(q3.compute_error(off), 3.0 * q.compute_error(off), atol=1e-12)


def test_vertex_quadric_from_faces():
    face_quadrics = [
        Quadric.from_triangle(np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([0, 1, 0.0])),
        Quadric.from_triangle(np.array([0, 0, 0.0]), np.array([0, 1, 0.0]), np.array([0, 0, 1.0])),
        Quadric.from_triangle(np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([0, 0, 1.0])),
    ]
    fa = [0.5, 0.5, 0.5]
    q = Quadric.vertex_quadric(face_quadrics, fa)
    assert q.compute_error(np.array([0.0, 0.0, 0.0])) < 1e-10
