import numpy as np
from kms.mesh import make_icosphere, face_areas
from kms.simplify_qem import simplify_qem


def test_reaches_target():
    mesh = make_icosphere(2)  # 162 verts
    result = simplify_qem(mesh, target_verts=50)
    assert result.n_verts == 50


def test_output_manifold():
    mesh = make_icosphere(2)
    result = simplify_qem(mesh, target_verts=50)
    f = result.faces
    assert np.all(f >= 0)
    assert np.all(f < result.n_verts)
    assert not np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2]))


def test_no_optimal_position():
    mesh = make_icosphere(2)
    result = simplify_qem(mesh, target_verts=80, use_optimal_position=False)
    assert result.n_verts == 80


def test_deterministic():
    mesh = make_icosphere(2)
    r1 = simplify_qem(mesh, target_verts=80)
    r2 = simplify_qem(mesh, target_verts=80)
    np.testing.assert_array_equal(r1.vertices, r2.vertices)
    np.testing.assert_array_equal(r1.faces, r2.faces)


def test_restriction_matrix_shape():
    mesh = make_icosphere(2)  # 162 verts
    result, P = simplify_qem(mesh, target_verts=80, compute_restriction=True)
    assert P.shape == (80, 162)


def test_restriction_row_sums():
    mesh = make_icosphere(2)
    _, P = simplify_qem(mesh, target_verts=80, compute_restriction=True)
    row_sums = np.array(P.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, np.ones(80), atol=1e-12)


def test_restriction_nonneg():
    mesh = make_icosphere(2)
    _, P = simplify_qem(mesh, target_verts=80, compute_restriction=True)
    assert np.all(P.toarray() >= -1e-15)


def test_restriction_preserves_constant():
    mesh = make_icosphere(2)
    _, P = simplify_qem(mesh, target_verts=80, compute_restriction=True)
    ones_fine = np.ones(162)
    result = P @ ones_fine
    np.testing.assert_allclose(result, np.ones(80), atol=1e-12)
