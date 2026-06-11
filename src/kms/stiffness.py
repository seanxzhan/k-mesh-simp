"""Thin-shell stiffness matrix assembly for triangle meshes.

Builds membrane (in-plane) + bending (out-of-plane) stiffness matrices using:
  - CST (Constant Strain Triangle) for membrane
  - Discrete hinge (dihedral angle) for bending

Each vertex has 3 DOFs (x, y, z), so the global stiffness matrix is (3n x 3n).
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from scipy import sparse

from kms.mesh import TriMesh


def membrane_stiffness_cst(
    mesh: TriMesh,
    E: float = 1.0,
    nu: float = 0.3,
    thickness: float = 0.001,
) -> sparse.csc_matrix:
    """Assemble the membrane (in-plane) stiffness matrix using CST elements.

    Each triangle is a constant-strain element in its local tangent plane.
    The resulting matrix is 3n x 3n.
    """
    n = mesh.n_verts
    ndof = 3 * n
    v = mesh.vertices
    f = mesh.faces

    D = (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2],
    ])

    rows, cols, vals = [], [], []

    for tri in f:
        i0, i1, i2 = tri
        p0, p1, p2 = v[i0], v[i1], v[i2]

        e1_raw = p1 - p0
        e1 = e1_raw / np.linalg.norm(e1_raw)
        normal = np.cross(e1_raw, p2 - p0)
        area = 0.5 * np.linalg.norm(normal)
        if area < 1e-16:
            continue
        normal = normal / (2 * area)
        e2 = np.cross(normal, e1)

        x1 = np.dot(p1 - p0, e1)
        x2 = np.dot(p2 - p0, e1)
        y2 = np.dot(p2 - p0, e2)

        det_J = x1 * y2
        if abs(det_J) < 1e-16:
            continue

        B = (1.0 / det_J) * np.array([
            [y2 - 0, 0, 0 - y2, 0, 0, 0],
            [0, x2 - x1, 0, 0 - x2, 0, x1],
            [x2 - x1, y2 - 0, 0 - x2, 0 - y2, x1, 0],
        ])

        Ke_local = (area * thickness) * (B.T @ D @ B)

        R = np.column_stack([e1, e2])  # 3x2

        T = np.zeros((9, 6))
        T[0:3, 0:2] = R
        T[3:6, 2:4] = R
        T[6:9, 4:6] = R

        Ke_global = T @ Ke_local @ T.T

        dofs = np.array([3*i0, 3*i0+1, 3*i0+2,
                         3*i1, 3*i1+1, 3*i1+2,
                         3*i2, 3*i2+1, 3*i2+2])

        for a in range(9):
            for b in range(9):
                if abs(Ke_global[a, b]) > 1e-20:
                    rows.append(dofs[a])
                    cols.append(dofs[b])
                    vals.append(Ke_global[a, b])

    K = sparse.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()
    K = 0.5 * (K + K.T)
    return K


def _find_edge_pairs(mesh: TriMesh) -> np.ndarray:
    """Find interior edges and their opposing vertices (hinges).

    Returns array of shape (n_hinges, 4) where each row is [v0, v1, v2, v3]:
      - (v0, v1) is the shared edge
      - v2 is the opposite vertex in triangle A
      - v3 is the opposite vertex in triangle B
    """
    edge_to_faces: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    for fi, tri in enumerate(mesh.faces):
        for local_opp in range(3):
            a = tri[(local_opp + 1) % 3]
            b = tri[(local_opp + 2) % 3]
            edge_key = (min(a, b), max(a, b))
            edge_to_faces[edge_key].append((fi, local_opp))

    hinges = []
    for (ea, eb), face_list in edge_to_faces.items():
        if len(face_list) != 2:
            continue
        fi_a, opp_a = face_list[0]
        fi_b, opp_b = face_list[1]
        v_opp_a = mesh.faces[fi_a][opp_a]
        v_opp_b = mesh.faces[fi_b][opp_b]
        hinges.append([ea, eb, v_opp_a, v_opp_b])

    return np.array(hinges, dtype=np.int64)


def bending_stiffness_hinge(
    mesh: TriMesh,
    E: float = 1.0,
    nu: float = 0.3,
    thickness: float = 0.001,
) -> sparse.csc_matrix:
    """Assemble the bending stiffness matrix using the discrete hinge model.

    For each interior edge shared by two triangles, bending resistance is
    modeled as a stiffness on the dihedral angle.
    """
    n = mesh.n_verts
    ndof = 3 * n
    v = mesh.vertices

    kb = E * thickness**3 / (12.0 * (1.0 - nu**2))

    hinges = _find_edge_pairs(mesh)
    if len(hinges) == 0:
        return sparse.csc_matrix((ndof, ndof))

    rows, cols, vals = [], [], []

    for hinge in hinges:
        v0_idx, v1_idx, v2_idx, v3_idx = hinge
        p0, p1, p2, p3 = v[v0_idx], v[v1_idx], v[v2_idx], v[v3_idx]

        e = p1 - p0
        e_len = np.linalg.norm(e)
        if e_len < 1e-16:
            continue

        n0 = np.cross(e, p2 - p0)
        n1 = np.cross(e, p3 - p0)
        A0 = 0.5 * np.linalg.norm(n0)
        A1 = 0.5 * np.linalg.norm(n1)
        if A0 < 1e-16 or A1 < 1e-16:
            continue

        n0_hat = n0 / (2.0 * A0)
        n1_hat = n1 / (2.0 * A1)

        e_hat = e / e_len

        h0 = 2.0 * A0 / e_len
        h1 = 2.0 * A1 / e_len

        d02 = p2 - p0
        d12 = p2 - p1
        d03 = p3 - p0
        d13 = p3 - p1

        cot_02 = np.dot(e, d02) / (2.0 * A0)
        cot_12 = -np.dot(e, d12) / (2.0 * A0)
        cot_03 = np.dot(e, d03) / (2.0 * A1)
        cot_13 = -np.dot(e, d13) / (2.0 * A1)

        grad_v2 = n0_hat / h0
        grad_v3 = -n1_hat / h1
        grad_v0 = -cot_02 * grad_v2 - cot_03 * grad_v3
        grad_v1 = -cot_12 * grad_v2 - cot_13 * grad_v3

        coeff = kb * e_len**2 / (A0 + A1)

        grad = np.concatenate([grad_v0, grad_v1, grad_v2, grad_v3])  # 12-vector
        Ke = coeff * np.outer(grad, grad)  # 12x12

        dofs = np.array([
            3*v0_idx, 3*v0_idx+1, 3*v0_idx+2,
            3*v1_idx, 3*v1_idx+1, 3*v1_idx+2,
            3*v2_idx, 3*v2_idx+1, 3*v2_idx+2,
            3*v3_idx, 3*v3_idx+1, 3*v3_idx+2,
        ])

        for a in range(12):
            for b in range(12):
                if abs(Ke[a, b]) > 1e-20:
                    rows.append(dofs[a])
                    cols.append(dofs[b])
                    vals.append(Ke[a, b])

    K = sparse.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()
    K = 0.5 * (K + K.T)
    return K


def shell_stiffness(
    mesh: TriMesh,
    E: float = 1.0,
    nu: float = 0.3,
    thickness: float = 0.001,
) -> tuple[sparse.csc_matrix, sparse.csc_matrix, sparse.csc_matrix]:
    """Build the full thin-shell stiffness matrix K = K_membrane + K_bending.

    Returns (K_total, K_membrane, K_bending).
    """
    K_m = membrane_stiffness_cst(mesh, E, nu, thickness)
    K_b = bending_stiffness_hinge(mesh, E, nu, thickness)
    return K_m + K_b, K_m, K_b
