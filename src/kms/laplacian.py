from __future__ import annotations

import numpy as np
from scipy import sparse
from kms.mesh import TriMesh


def cotangent_laplacian(mesh: TriMesh) -> tuple[sparse.csc_matrix, sparse.dia_matrix]:
    """Build the cotangent stiffness matrix L and lumped mass matrix M.

    Returns (L, M) where:
        - L is symmetric positive semi-definite, rows sum to zero.
            Off-diagonal L_ij = -0.5 * (cot alpha_ij + cot beta_ij) for edge (i,j).
            Diagonal L_ii = -sum of off-diagonal entries in row i.
        - M is diagonal with positive entries (1/3 of total incident face area per vertex).
    """
    v = mesh.vertices
    f = mesh.faces
    n = mesh.n_verts

    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    v0, v1, v2 = v[i0], v[i1], v[i2]

    e01 = v1 - v0
    e02 = v2 - v0
    e12 = v2 - v1

    dot0 = np.sum(e01 * e02, axis=1)
    cross0 = np.linalg.norm(np.cross(e01, e02), axis=1)
    cot0 = dot0 / cross0

    dot1 = np.sum(-e01 * e12, axis=1)
    cross1 = np.linalg.norm(np.cross(-e01, e12), axis=1)
    cot1 = dot1 / cross1

    dot2 = np.sum(-e02 * (-e12), axis=1)
    cross2 = np.linalg.norm(np.cross(-e02, -e12), axis=1)
    cot2 = dot2 / cross2

    rows = np.concatenate([i1, i2, i0, i2, i0, i1])
    cols = np.concatenate([i2, i1, i2, i0, i1, i0])
    vals = np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])

    L = sparse.coo_matrix((-0.5 * vals, (rows, cols)), shape=(n, n)).tocsc()
    L = L - sparse.diags(np.array(L.sum(axis=1)).ravel())

    areas = 0.5 * cross0
    vertex_areas = np.zeros(n)
    np.add.at(vertex_areas, i0, areas / 3.0)
    np.add.at(vertex_areas, i1, areas / 3.0)
    np.add.at(vertex_areas, i2, areas / 3.0)

    M = sparse.diags(vertex_areas)

    return L.tocsc(), M
