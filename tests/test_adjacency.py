import numpy as np
from kms.mesh import make_grid, make_icosphere, load_obj
from kms.adjacency import MeshAdjacency


def test_build_adjacency(small_icosphere):
    adj = MeshAdjacency(small_icosphere)
    assert adj.n_active_verts == 42
    assert adj.n_active_faces == 80


def test_collapse_decreases_counts(small_icosphere):
    adj = MeshAdjacency(small_icosphere)
    edges = adj.get_edges()
    for u, v in edges:
        if adj.is_collapsible(u, v):
            mid = (adj.vertices[u] + adj.vertices[v]) / 2
            adj.collapse_edge(u, v, mid)
            break
    assert adj.n_active_verts == 41
    assert adj.n_active_faces == 78


def test_output_manifold_after_collapses(small_icosphere):
    adj = MeshAdjacency(small_icosphere)
    for _ in range(10):
        for u, v in adj.get_edges():
            if adj.is_collapsible(u, v):
                mid = (adj.vertices[u] + adj.vertices[v]) / 2
                adj.collapse_edge(u, v, mid)
                break
    result = adj.to_trimesh()
    f = result.faces
    assert np.all(f >= 0)
    assert np.all(f < result.n_verts)
    assert not np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2]))


def test_link_condition_rejects_deleted():
    mesh = make_grid(4, 4)
    adj = MeshAdjacency(mesh)
    for u, v in adj.get_edges():
        if adj.is_collapsible(u, v):
            adj.collapse_edge(u, v, adj.vertices[u])
            assert not adj.is_collapsible(u, v)
            assert not adj.is_collapsible(v, u)
            break


def test_boundary_detection():
    mesh = make_grid(4, 4)
    adj = MeshAdjacency(mesh)
    has_boundary = False
    for nb in adj.vert_neighbors[0]:
        if adj.is_boundary_edge(0, nb):
            has_boundary = True
            break
    assert has_boundary
