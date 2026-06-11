import numpy as np
import tempfile, os
from kms.mesh import make_grid, make_icosphere, make_torus, face_areas, load_obj, save_obj


def test_grid_counts():
    m = make_grid(5, 4)
    assert m.n_verts == 20
    assert m.n_faces == 2 * 4 * 3


def test_icosphere_counts():
    assert make_icosphere(0).n_verts == 12
    assert make_icosphere(1).n_verts == 42
    assert make_icosphere(2).n_verts == 162


def test_euler_icosphere(small_icosphere):
    assert small_icosphere.n_verts - small_icosphere.n_edges + small_icosphere.n_faces == 2


def test_euler_torus(small_torus):
    assert small_torus.n_verts - small_torus.n_edges + small_torus.n_faces == 0


def test_face_areas_positive(small_icosphere):
    assert np.all(face_areas(small_icosphere) > 0)


def test_no_degenerate_faces(small_icosphere):
    f = small_icosphere.faces
    assert not np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2]))


def test_obj_roundtrip(small_icosphere):
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
        path = tmp.name
    try:
        save_obj(small_icosphere, path)
        loaded = load_obj(path)
        np.testing.assert_allclose(loaded.vertices, small_icosphere.vertices, atol=1e-6)
        np.testing.assert_array_equal(loaded.faces, small_icosphere.faces)
    finally:
        os.unlink(path)
