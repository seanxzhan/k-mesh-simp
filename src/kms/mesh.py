from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class TriMesh:
    vertices: np.ndarray  # (n, 3) float64
    faces: np.ndarray  # (f, 3) int64

    @property
    def n_verts(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    @property
    def edges(self) -> np.ndarray:
        f = self.faces
        all_edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        all_edges.sort(axis=1)
        return np.unique(all_edges, axis=0)

    @property
    def n_edges(self) -> int:
        return self.edges.shape[0]


def face_areas(mesh: TriMesh) -> np.ndarray:
    v, f = mesh.vertices, mesh.faces
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def face_normals(mesh: TriMesh) -> np.ndarray:
    v, f = mesh.vertices, mesh.faces
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    n = np.cross(e1, e2)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return n / norms


def load_obj(path: str) -> TriMesh:
    vertices = []
    faces = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                face_verts = []
                for p in parts[1:]:
                    face_verts.append(int(p.split("/")[0]) - 1)
                if len(face_verts) == 3:
                    faces.append(face_verts)
                elif len(face_verts) == 4:
                    faces.append([face_verts[0], face_verts[1], face_verts[2]])
                    faces.append([face_verts[0], face_verts[2], face_verts[3]])
    return TriMesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int64),
    )


def save_obj(mesh: TriMesh, path: str):
    with open(path, "w") as f:
        for v in mesh.vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in mesh.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def make_grid(nx: int, ny: int) -> TriMesh:
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = j * nx + i + 1
            v01 = (j + 1) * nx + i
            v11 = (j + 1) * nx + i + 1
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    return TriMesh(vertices=vertices, faces=np.array(faces, dtype=np.int64))


def make_icosphere(subdivisions: int = 2) -> TriMesh:
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts[0])
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    for _ in range(subdivisions):
        verts, faces = _subdivide(verts, faces)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    return TriMesh(vertices=verts, faces=faces)


def _subdivide(vertices, faces):
    edge_midpoints = {}
    new_verts = list(vertices)

    def get_mid(a, b):
        key = (min(a, b), max(a, b))
        if key in edge_midpoints:
            return edge_midpoints[key]
        idx = len(new_verts)
        new_verts.append((vertices[a] + vertices[b]) / 2.0)
        edge_midpoints[key] = idx
        return idx

    new_faces = []
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        ab, bc, ca = get_mid(a, b), get_mid(b, c), get_mid(c, a)
        new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
    return np.array(new_verts), np.array(new_faces, dtype=np.int64)


def make_torus(R: float = 1.0, r: float = 0.4, n_major: int = 20, n_minor: int = 12) -> TriMesh:
    u = np.linspace(0, 2 * np.pi, n_major, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)
    uu, vv = np.meshgrid(u, v)
    uu, vv = uu.ravel(), vv.ravel()
    x = (R + r * np.cos(vv)) * np.cos(uu)
    y = (R + r * np.cos(vv)) * np.sin(uu)
    z = r * np.sin(vv)
    vertices = np.column_stack([x, y, z])
    faces = []
    for j in range(n_minor):
        for i in range(n_major):
            v00 = j * n_major + i
            v10 = j * n_major + (i + 1) % n_major
            v01 = ((j + 1) % n_minor) * n_major + i
            v11 = ((j + 1) % n_minor) * n_major + (i + 1) % n_major
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    return TriMesh(vertices=vertices, faces=np.array(faces, dtype=np.int64))
