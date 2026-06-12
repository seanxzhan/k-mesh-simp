"""Stiffness quadric: strain energy as a QEM-like quadratic form.

For each vertex v, the stiffness quadric encodes:
    E(x) = (x - x₀)^T K_vv (x - x₀)
         = x^T K_vv x - 2 x^T K_vv x₀ + x₀^T K_vv x₀

where K_vv is the 3×3 diagonal block of the original stiffness matrix
and x₀ is the original vertex position.

This measures "how much strain energy does placing this vertex at x cost,
relative to its original position, weighted by its mechanical self-stiffness."

Accumulation: on collapse (u,v) → w, the Schur complement folds v's stiffness
into u: K_uu_new = K_uu + K_uv K_vv⁻¹ K_vu. The quadric updates accordingly.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from kms.mesh import TriMesh
from kms.adjacency import MeshAdjacency
from kms.stiffness import shell_stiffness


class StiffnessQuadric:
    """Quadratic form E(x) = x^T A x + 2 b^T x + c from stiffness."""

    def __init__(self, A: np.ndarray | None = None, b: np.ndarray | None = None, c: float = 0.0):
        self.A = A if A is not None else np.zeros((3, 3))
        self.b = b if b is not None else np.zeros(3)
        self.c = c

    @classmethod
    def from_stiffness_block(cls, K_vv: np.ndarray, x0: np.ndarray) -> StiffnessQuadric:
        """Build stiffness quadric from diagonal block K_vv and rest position x0.

        E(x) = (x - x0)^T K_vv (x - x0)
             = x^T K_vv x - 2 x^T K_vv x0 + x0^T K_vv x0
        """
        A = K_vv.copy()
        b = -K_vv @ x0
        c = float(x0 @ K_vv @ x0)
        return cls(A, b, c)

    def compute_error(self, x: np.ndarray) -> float:
        """Evaluate E(x) = x^T A x + 2 b^T x + c."""
        return float(x @ self.A @ x + 2.0 * self.b @ x + self.c)

    def optimal_position(self) -> tuple[np.ndarray, bool]:
        """Find x minimizing E(x). Returns (position, success)."""
        try:
            U, S, Vt = np.linalg.svd(self.A, full_matrices=False)
            tol = np.finfo(float).eps * 3 * np.max(S)
            if np.min(S) < tol:
                return np.zeros(3), False
            S_inv = np.diag(1.0 / S)
            A_pinv = Vt.T @ S_inv @ U.T
            x = A_pinv @ (-self.b)
            return x, True
        except np.linalg.LinAlgError:
            return np.zeros(3), False

    def __add__(self, other: StiffnessQuadric) -> StiffnessQuadric:
        return StiffnessQuadric(self.A + other.A, self.b + other.b, self.c + other.c)

    def __mul__(self, scalar: float) -> StiffnessQuadric:
        return StiffnessQuadric(self.A * scalar, self.b * scalar, self.c * scalar)

    def __iadd__(self, other: StiffnessQuadric) -> StiffnessQuadric:
        self.A = self.A + other.A
        self.b = self.b + other.b
        self.c = self.c + other.c
        return self


def _extract_block(K: sparse.spmatrix, dofs_row: list[int], dofs_col: list[int]) -> np.ndarray:
    return np.array(K[np.ix_(dofs_row, dofs_col)].todense())


def build_stiffness_quadrics(
    K: sparse.spmatrix, mesh: TriMesh
) -> dict[int, StiffnessQuadric]:
    """Build per-vertex stiffness quadrics from the global stiffness matrix.

    For each vertex v: Q_v = StiffnessQuadric(K_vv, x0_v)
    """
    quadrics = {}
    for v in range(mesh.n_verts):
        dofs = [3*v, 3*v+1, 3*v+2]
        K_vv = _extract_block(K, dofs, dofs)
        quadrics[v] = StiffnessQuadric.from_stiffness_block(K_vv, mesh.vertices[v])
    return quadrics


def compute_schur_updated_quadric(
    K: sparse.spmatrix, u: int, v: int, quadric_u: StiffnessQuadric, quadric_v: StiffnessQuadric
) -> StiffnessQuadric:
    """Accumulate v's quadric into u using Schur complement correction.

    After eliminating v, u's effective self-stiffness increases by:
        K_uu_new = K_uu_old + K_uv K_vv⁻¹ K_vu

    The merged quadric reflects this increased stiffness at u's position.
    """
    dofs_u = [3*u, 3*u+1, 3*u+2]
    dofs_v = [3*v, 3*v+1, 3*v+2]

    K_vv = _extract_block(K, dofs_v, dofs_v)
    K_uv = _extract_block(K, dofs_u, dofs_v)
    K_vu = _extract_block(K, dofs_v, dofs_u)

    svd_vals = np.linalg.svd(K_vv, compute_uv=False)
    if svd_vals.max() < 1e-20:
        return quadric_u + quadric_v

    K_vv_inv = np.linalg.pinv(K_vv, rcond=1e-12)
    correction = K_uv @ K_vv_inv @ K_vu

    # The merged quadric: add v's quadric + the Schur correction at u's current position
    merged = quadric_u + quadric_v
    # Add the correction as additional stiffness at the merged vertex
    merged.A = merged.A + correction

    return merged


def per_vertex_stiffness_quadric_cost(
    K: sparse.spmatrix, mesh: TriMesh
) -> np.ndarray:
    """Per-vertex cost: E(x_v) evaluated at the current position.

    For the original mesh this should be ~0 (vertices are at their rest positions).
    After collapses move vertices, this grows.
    """
    n = mesh.n_verts
    costs = np.zeros(n)
    quadrics = build_stiffness_quadrics(K, mesh)
    for v in range(n):
        costs[v] = quadrics[v].compute_error(mesh.vertices[v])
    return costs


def per_edge_stiffness_quadric_cost(
    K: sparse.spmatrix, mesh: TriMesh
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Per-edge collapse cost using stiffness quadrics.

    For edge (u,v): merge quadrics Q_u + Q_v, find optimal position,
    evaluate cost at that position.

    Returns (edges, costs, optimal_positions).
    """
    adj = MeshAdjacency(mesh)
    quadrics = build_stiffness_quadrics(K, mesh)
    edges = adj.get_edges()
    n_edges = len(edges)
    costs = np.zeros(n_edges)
    positions = np.zeros((n_edges, 3))

    for ei, (u, v) in enumerate(edges):
        eq = quadrics[u] + quadrics[v]
        pos, success = eq.optimal_position()
        if not success:
            # Fall back to better endpoint
            c_u = eq.compute_error(mesh.vertices[u])
            c_v = eq.compute_error(mesh.vertices[v])
            if c_u <= c_v:
                pos = mesh.vertices[u].copy()
            else:
                pos = mesh.vertices[v].copy()
        costs[ei] = eq.compute_error(pos)
        positions[ei] = pos

    return edges, costs, positions
