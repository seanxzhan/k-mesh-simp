from __future__ import annotations

import numpy as np


class Quadric:
    """Quadric error metric for mesh simplification (Garland & Heckbert 1997).

    Stores the quadratic form E(x) = x^T A x + 2 b^T x + c
    in decomposed form for efficient combination and evaluation.
    """

    def __init__(self, A: np.ndarray | None = None, b: np.ndarray | None = None, c: float = 0.0):
        self.A = A if A is not None else np.zeros((3, 3))  # 3x3 symmetric
        self.b = b if b is not None else np.zeros(3)  # 3-vector
        self.c = c  # scalar

    @classmethod
    def from_plane(cls, normal: np.ndarray, point: np.ndarray) -> Quadric:
        """Quadric measuring squared distance to a plane (n^T x + d = 0)."""
        n_len = np.linalg.norm(normal)
        if n_len < 1e-12:
            return cls.from_point(point)
        n = normal / n_len
        d = -np.dot(n, point)
        # E(x) = (n^T x + d)^2 = x^T (nn^T) x + 2d n^T x + d^2
        A = np.outer(n, n)
        b = d * n
        c = d * d
        return cls(A, b, c)

    @classmethod
    def from_triangle(cls, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> Quadric:
        """Quadric from a triangle face (plane through the triangle)."""
        normal = np.cross(v2 - v1, v3 - v1)
        n_len = np.linalg.norm(normal)
        if n_len < 1e-12:
            centroid = (v1 + v2 + v3) / 3.0
            return cls.from_point(centroid)
        return cls.from_plane(normal, v1)

    @classmethod
    def from_point(cls, point: np.ndarray, weight: float = 1.0) -> Quadric:
        """Quadric penalizing distance to a point: E(x) = w * ||x - p||^2."""
        A = weight * np.eye(3)
        b = -weight * point
        c = weight * np.dot(point, point)
        return cls(A, b, c)

    @classmethod
    def from_line(cls, point: np.ndarray, normal: np.ndarray, face_areas: list[float], weight: float = 1e-3) -> Quadric:
        """Line quadric that penalizes movement away from the vertex normal direction.

        Constrains the vertex to stay on the line defined by its position and normal.
        This preserves sharp features by penalizing tangential drift.

        Args:
            point: Vertex position
            normal: Area-weighted vertex normal (will be normalized internally)
            face_areas: Areas of incident faces (used for weighting)
            weight: Regularization weight
        """
        n_len = np.linalg.norm(normal)
        if n_len < 1e-12:
            total_area = np.sum(face_areas) / 3.0
            return cls.from_point(point, weight * total_area)

        u1 = normal / n_len

        # Build orthonormal basis perpendicular to u1
        e1 = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(u1, e1)) > 0.9:
            e1 = np.array([0.0, 1.0, 0.0])
        u2 = e1 - np.dot(u1, e1) * u1
        u2 = u2 / np.linalg.norm(u2)
        u3 = np.cross(u1, u2)

        # Line quadric = sum of two plane quadrics perpendicular to the line
        q2 = cls.from_plane(u2, point)
        q3 = cls.from_plane(u3, point)
        line_q = q2 + q3
        scale = weight * np.sum(face_areas) / 3.0
        return line_q * scale

    @classmethod
    def vertex_quadric(cls, face_quadrics: list[Quadric], face_areas: list[float]) -> Quadric:
        """Area-weighted sum of face quadrics for a vertex."""
        result = cls()
        for q, area in zip(face_quadrics, face_areas):
            result = result + q * (area / 3.0)
        return result

    @classmethod
    def edge_quadric(cls, q1: Quadric, q2: Quadric) -> Quadric:
        """Sum of vertex quadrics for an edge collapse."""
        return q1 + q2

    def compute_error(self, x: np.ndarray) -> float:
        """Evaluate E(x) = x^T A x + 2 b^T x + c."""
        return float(x @ self.A @ x + 2.0 * self.b @ x + self.c)

    def optimal_position(self) -> tuple[np.ndarray, bool]:
        """Find x minimizing E(x) via SVD pseudoinverse of A.

        Returns (position, success). Success is False if A is near-singular.
        """
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

    def __add__(self, other: Quadric) -> Quadric:
        return Quadric(self.A + other.A, self.b + other.b, self.c + other.c)

    def __mul__(self, scalar: float) -> Quadric:
        return Quadric(self.A * scalar, self.b * scalar, self.c * scalar)

    def __iadd__(self, other: Quadric) -> Quadric:
        self.A = self.A + other.A
        self.b = self.b + other.b
        self.c = self.c + other.c
        return self
