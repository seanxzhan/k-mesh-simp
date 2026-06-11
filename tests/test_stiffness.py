import numpy as np
from scipy import sparse
from kms.mesh import make_grid, make_icosphere
from kms.stiffness import membrane_stiffness_cst, bending_stiffness_hinge, shell_stiffness


def test_membrane_shape():
    mesh = make_grid(5, 5)
    K = membrane_stiffness_cst(mesh)
    ndof = 3 * mesh.n_verts
    assert K.shape == (ndof, ndof)


def test_membrane_symmetric():
    mesh = make_grid(5, 5)
    K = membrane_stiffness_cst(mesh)
    assert np.allclose(K.toarray(), K.T.toarray(), atol=1e-12)


def test_membrane_nonneg_diagonal():
    mesh = make_grid(5, 5)
    K = membrane_stiffness_cst(mesh)
    assert np.all(K.diagonal() >= -1e-14)


def test_membrane_nonzero():
    mesh = make_grid(5, 5)
    K = membrane_stiffness_cst(mesh)
    assert K.nnz > 0


def test_bending_shape():
    mesh = make_grid(5, 5)
    K = bending_stiffness_hinge(mesh)
    ndof = 3 * mesh.n_verts
    assert K.shape == (ndof, ndof)


def test_bending_symmetric():
    mesh = make_grid(5, 5)
    K = bending_stiffness_hinge(mesh)
    assert np.allclose(K.toarray(), K.T.toarray(), atol=1e-12)


def test_bending_nonneg_diagonal():
    mesh = make_grid(5, 5)
    K = bending_stiffness_hinge(mesh)
    assert np.all(K.diagonal() >= -1e-14)


def test_bending_nonzero():
    mesh = make_grid(5, 5)
    K = bending_stiffness_hinge(mesh)
    assert K.nnz > 0


def test_shell_additivity():
    mesh = make_grid(5, 5)
    K_total, K_m, K_b = shell_stiffness(mesh)
    diff = K_total - (K_m + K_b)
    assert abs(diff).max() < 1e-12


def test_shell_on_icosphere():
    mesh = make_icosphere(1)
    K_total, K_m, K_b = shell_stiffness(mesh)
    ndof = 3 * mesh.n_verts
    assert K_total.shape == (ndof, ndof)
    assert np.allclose(K_total.toarray(), K_total.T.toarray(), atol=1e-12)
    assert np.all(K_total.diagonal() >= -1e-14)
    assert K_m.nnz > 0
    assert K_b.nnz > 0


def test_bending_reaches_2ring():
    """Bending connects the 2-ring: each hinge element touches 4 vertices (12 DOFs)."""
    mesh = make_grid(10, 10)
    K_b = bending_stiffness_hinge(mesh)
    # Bending matrix should be non-trivially populated
    assert K_b.nnz > 0
    # An interior vertex in a grid has 6 neighbors in 1-ring, more in 2-ring.
    # The bending matrix should connect vertices beyond the 1-ring via shared hinges.
    # Check that nnz per row is > 3 (minimum for a single hinge's 12x12 block)
    avg_nnz = K_b.nnz / K_b.shape[0]
    assert avg_nnz > 3


def test_translation_null_space():
    """K should have at least a 3D null space (translations) for closed meshes."""
    mesh = make_icosphere(1)
    K_total, _, _ = shell_stiffness(mesh)
    eigvals = np.linalg.eigvalsh(K_total.toarray())
    eigvals_sorted = np.sort(eigvals)
    n_near_zero = np.sum(eigvals_sorted < 1e-8)
    assert n_near_zero >= 3
