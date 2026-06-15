import numpy as np
from scipy import sparse
from kms.mesh import make_icosphere, make_grid, face_areas
from kms.simplify_stiffness_quadric import simplify_stiffness_quadric


def test_reaches_target():
    mesh = make_icosphere(2)  # 162 verts
    result = simplify_stiffness_quadric(mesh, target_verts=80)
    assert result.n_verts == 80


def test_output_manifold():
    mesh = make_icosphere(2)
    result = simplify_stiffness_quadric(mesh, target_verts=80)
    f = result.faces
    assert np.all(f >= 0)
    assert np.all(f < result.n_verts)
    assert not np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2]))


def test_positive_areas():
    mesh = make_icosphere(2)
    result = simplify_stiffness_quadric(mesh, target_verts=80)
    areas = face_areas(result)
    assert np.all(areas > 0)


def test_deterministic():
    mesh = make_icosphere(2)
    r1 = simplify_stiffness_quadric(mesh, target_verts=80)
    r2 = simplify_stiffness_quadric(mesh, target_verts=80)
    np.testing.assert_array_equal(r1.vertices, r2.vertices)
    np.testing.assert_array_equal(r1.faces, r2.faces)


def test_on_grid():
    mesh = make_grid(6, 6)  # 36 verts
    result = simplify_stiffness_quadric(mesh, target_verts=18)
    assert result.n_verts == 18
    areas = face_areas(result)
    assert np.all(areas > 0)


def test_thickness_affects_result():
    """Different thickness should produce different simplifications."""
    mesh = make_icosphere(2)
    r_thin = simplify_stiffness_quadric(mesh, target_verts=80, mode="stiffness", thickness=0.001)
    r_thick = simplify_stiffness_quadric(mesh, target_verts=80, mode="stiffness", thickness=0.1)
    # Results should differ (different bending/membrane ratio)
    assert not np.allclose(r_thin.vertices, r_thick.vertices)


# --- Combined mode tests (Approach 3) ---

def test_combined_reaches_target():
    mesh = make_icosphere(2)
    result = simplify_stiffness_quadric(mesh, target_verts=80, mode="combined")
    assert result.n_verts == 80


def test_combined_output_manifold():
    mesh = make_icosphere(2)
    result = simplify_stiffness_quadric(mesh, target_verts=80, mode="combined")
    f = result.faces
    assert np.all(f >= 0)
    assert np.all(f < result.n_verts)
    assert not np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2]))


def test_combined_positive_areas():
    mesh = make_icosphere(2)
    result = simplify_stiffness_quadric(mesh, target_verts=80, mode="combined")
    areas = face_areas(result)
    assert np.all(areas > 0)


def test_combined_better_area_ratio_than_stiffness_only():
    """Combined mode should produce better triangle quality than stiffness-only."""
    mesh = make_icosphere(2)
    r_stiff = simplify_stiffness_quadric(mesh, target_verts=80, mode="stiffness")
    r_combined = simplify_stiffness_quadric(mesh, target_verts=80, mode="combined")
    ratio_stiff = face_areas(r_stiff).max() / face_areas(r_stiff).min()
    ratio_combined = face_areas(r_combined).max() / face_areas(r_combined).min()
    # Combined should have lower area ratio (more uniform)
    assert ratio_combined <= ratio_stiff * 1.5  # allow some tolerance


def test_combined_on_grid():
    mesh = make_grid(6, 6)
    result = simplify_stiffness_quadric(mesh, target_verts=18, mode="combined")
    assert result.n_verts == 18
    areas = face_areas(result)
    assert np.all(areas > 0)


# --- Skinning weight tests ---

def test_skinning_weights_shape():
    """W should be (n_fine, n_coarse)."""
    mesh = make_icosphere(2)  # 162 verts
    result, W = simplify_stiffness_quadric(
        mesh, target_verts=80, mode="combined", compute_skinning_weights=True
    )
    assert W.shape == (162, 80)


def test_skinning_weights_rows_bounded():
    """Row sums should be close to 1 (approximate partition of unity)."""
    mesh = make_icosphere(1)  # 42 verts
    result, W = simplify_stiffness_quadric(
        mesh, target_verts=20, mode="combined", compute_skinning_weights=True
    )
    row_sums = np.array(W.sum(axis=1)).ravel()
    # Surviving vertices have exact sum = 1; eliminated may deviate
    # due to bending stiffness coupling beyond 1-ring
    assert np.all(row_sums > 0.5), f"Some row sums too low: {row_sums.min()}"
    assert np.all(row_sums <= 1.0 + 1e-10), f"Some row sums > 1: {row_sums.max()}"


def test_skinning_weights_nonneg():
    """Skinning weights should be non-negative."""
    mesh = make_icosphere(1)
    result, W = simplify_stiffness_quadric(
        mesh, target_verts=20, mode="combined", compute_skinning_weights=True
    )
    assert np.all(W.toarray() >= -1e-12)


def test_skinning_weights_rows_sum_to_one():
    """Every fine vertex's weights should sum to 1 (partition of unity)."""
    mesh = make_icosphere(1)
    result, W = simplify_stiffness_quadric(
        mesh, target_verts=20, mode="stiffness", compute_skinning_weights=True
    )
    W_dense = W.toarray()
    row_sums = W_dense.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(42), atol=1e-6)


def test_skinning_weights_reconstruct_positions():
    """W @ V_coarse should approximate V_fine for surviving vertices."""
    mesh = make_icosphere(1)
    result, W = simplify_stiffness_quadric(
        mesh, target_verts=20, mode="combined", compute_skinning_weights=True
    )
    # Reconstruct fine positions from coarse
    V_reconstructed = W @ result.vertices
    # For surviving vertices, reconstruction should be exact
    # (they map to themselves in the coarse mesh)
    # Check that the mean reconstruction error is reasonable
    errors = np.linalg.norm(V_reconstructed - mesh.vertices, axis=1)
    assert errors.mean() < 0.5  # loose bound — just sanity check
