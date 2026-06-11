import numpy as np
from scipy import sparse
from kms.mesh import make_grid, make_icosphere, make_torus, face_areas
from kms.laplacian import cotangent_laplacian


def test_L_is_symmetric(small_icosphere):
    L, M = cotangent_laplacian(small_icosphere)
    assert sparse.linalg.norm(L - L.T) < 1e-12


def test_L_row_sums_zero(small_icosphere):
    L, M = cotangent_laplacian(small_icosphere)
    row_sums = np.array(L.sum(axis=1)).ravel()
    assert np.max(np.abs(row_sums)) < 1e-12


def test_L_positive_semidefinite(small_icosphere):
    L, M = cotangent_laplacian(small_icosphere)
    eigvals = np.linalg.eigvalsh(L.toarray())
    assert np.min(eigvals) > -1e-10


def test_L_off_diagonal_nonpositive(small_icosphere):
    """For Delaunay meshes (icosphere), off-diag of L <= 0."""
    L, _ = cotangent_laplacian(small_icosphere)
    L_dense = L.toarray()
    np.fill_diagonal(L_dense, 0)
    assert np.all(L_dense <= 1e-12)


def test_M_diagonal_positive(small_icosphere):
    _, M = cotangent_laplacian(small_icosphere)
    assert np.all(M.diagonal() > 0)


def test_M_sums_to_total_area(small_icosphere):
    _, M = cotangent_laplacian(small_icosphere)
    total_mass = M.diagonal().sum()
    total_area = face_areas(small_icosphere).sum()
    np.testing.assert_allclose(total_mass, total_area, rtol=1e-10)


def test_L_shape(small_grid):
    L, M = cotangent_laplacian(small_grid)
    n = small_grid.n_verts
    assert L.shape == (n, n)
    assert M.shape == (n, n)


def test_laplacian_torus(small_torus):
    L, M = cotangent_laplacian(small_torus)
    assert sparse.linalg.norm(L - L.T) < 1e-12
    assert np.all(M.diagonal() > 0)
    row_sums = np.array(L.sum(axis=1)).ravel()
    assert np.max(np.abs(row_sums)) < 1e-12
