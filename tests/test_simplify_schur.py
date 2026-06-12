import numpy as np
from kms.mesh import make_icosphere, make_grid, face_areas
from kms.simplify_schur import simplify_schur


# --- Flow mode tests ---

def test_flow_reaches_target():
    mesh = make_icosphere(2)  # 162 verts
    result = simplify_schur(mesh, target_verts=80, mode="flow")
    assert result.n_verts == 80


def test_flow_output_manifold():
    mesh = make_icosphere(2)
    result = simplify_schur(mesh, target_verts=80, mode="flow")
    f = result.faces
    assert np.all(f >= 0)
    assert np.all(f < result.n_verts)
    assert not np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2]))


def test_flow_positive_areas():
    mesh = make_icosphere(2)
    result = simplify_schur(mesh, target_verts=80, mode="flow")
    areas = face_areas(result)
    assert np.all(areas > 0)


def test_flow_deterministic():
    mesh = make_icosphere(2)
    r1 = simplify_schur(mesh, target_verts=80, mode="flow")
    r2 = simplify_schur(mesh, target_verts=80, mode="flow")
    np.testing.assert_array_equal(r1.vertices, r2.vertices)
    np.testing.assert_array_equal(r1.faces, r2.faces)


# --- Full mode tests (smaller mesh for speed) ---

def test_full_reaches_target():
    mesh = make_icosphere(1)  # 42 verts
    result = simplify_schur(mesh, target_verts=20, mode="full")
    assert result.n_verts == 20


def test_full_output_manifold():
    mesh = make_icosphere(1)
    result = simplify_schur(mesh, target_verts=20, mode="full")
    f = result.faces
    assert np.all(f >= 0)
    assert np.all(f < result.n_verts)
    assert not np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2]))


def test_full_positive_areas():
    mesh = make_icosphere(1)
    result = simplify_schur(mesh, target_verts=20, mode="full")
    areas = face_areas(result)
    assert np.all(areas > 0)


def test_full_deterministic():
    mesh = make_icosphere(1)
    r1 = simplify_schur(mesh, target_verts=20, mode="full")
    r2 = simplify_schur(mesh, target_verts=20, mode="full")
    np.testing.assert_array_equal(r1.vertices, r2.vertices)
    np.testing.assert_array_equal(r1.faces, r2.faces)


# --- Grid tests ---

def test_flow_on_grid():
    mesh = make_grid(6, 6)  # 36 verts
    result = simplify_schur(mesh, target_verts=18, mode="flow")
    assert result.n_verts == 18
    areas = face_areas(result)
    assert np.all(areas > 0)


def test_full_on_grid():
    mesh = make_grid(4, 4)  # 16 verts
    result = simplify_schur(mesh, target_verts=10, mode="full")
    assert result.n_verts == 10
    areas = face_areas(result)
    assert np.all(areas > 0)
