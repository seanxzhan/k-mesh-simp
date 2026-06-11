from __future__ import annotations

import numpy as np
from kms.mesh import TriMesh


class MeshAdjacency:
    """Lightweight mesh topology tracker supporting edge collapse."""

    def __init__(self, mesh: TriMesh):
        self.vertices = mesh.vertices.copy()
        self.faces = mesh.faces.copy()
        self.n_original = mesh.n_verts

        self._deleted_verts = np.zeros(mesh.n_verts, dtype=bool)
        self._deleted_faces = np.zeros(mesh.n_faces, dtype=bool)

        # Build adjacency
        self.vert_faces: list[set[int]] = [set() for _ in range(mesh.n_verts)]
        self.vert_neighbors: list[set[int]] = [set() for _ in range(mesh.n_verts)]
        # edge -> set of face indices (key is always (min, max))
        self.edge_faces: dict[tuple[int, int], set[int]] = {}

        for fi in range(mesh.n_faces):
            a, b, c = int(mesh.faces[fi, 0]), int(mesh.faces[fi, 1]), int(mesh.faces[fi, 2])
            self.vert_faces[a].add(fi)
            self.vert_faces[b].add(fi)
            self.vert_faces[c].add(fi)
            self.vert_neighbors[a].update([b, c])
            self.vert_neighbors[b].update([a, c])
            self.vert_neighbors[c].update([a, b])
            for u, v in [(a, b), (b, c), (a, c)]:
                key = (min(u, v), max(u, v))
                if key not in self.edge_faces:
                    self.edge_faces[key] = set()
                self.edge_faces[key].add(fi)

    @property
    def n_active_verts(self) -> int:
        return int(np.sum(~self._deleted_verts))

    @property
    def n_active_faces(self) -> int:
        return int(np.sum(~self._deleted_faces))

    def is_valid_vertex(self, v: int) -> bool:
        return not self._deleted_verts[v]

    def is_boundary_edge(self, u: int, v: int) -> bool:
        key = (min(u, v), max(u, v))
        faces = self.edge_faces.get(key, set())
        return len(faces) == 1

    def get_edges(self) -> list[tuple[int, int]]:
        edges = []
        for (u, v), faces in self.edge_faces.items():
            if faces and self.is_valid_vertex(u) and self.is_valid_vertex(v):
                edges.append((u, v))
        return sorted(edges)

    def is_collapsible(self, u: int, v: int) -> bool:
        """Link condition: common neighbors == number of shared faces."""
        if self._deleted_verts[u] or self._deleted_verts[v]:
            return False
        if v not in self.vert_neighbors[u]:
            return False

        key = (min(u, v), max(u, v))
        shared_faces = self.edge_faces.get(key, set())
        n_shared = len(shared_faces)

        common_neighbors = self.vert_neighbors[u] & self.vert_neighbors[v]

        if n_shared == 2:
            return len(common_neighbors) == 2
        elif n_shared == 1:
            return len(common_neighbors) == 1
        return False

    def collapse_edge(self, u: int, v: int, new_pos: np.ndarray) -> set[int]:
        """Collapse edge (u, v): merge v into u at new_pos.

        Returns set of affected vertex indices (neighbors of collapse site).
        """
        assert self.is_collapsible(u, v)

        self.vertices[u] = new_pos

        # Faces shared by u and v get deleted
        key = (min(u, v), max(u, v))
        shared_faces = set(self.edge_faces.get(key, set()))

        # Collect affected neighbors before modifying
        affected = (self.vert_neighbors[u] | self.vert_neighbors[v]) - {u, v}

        # Delete shared faces
        for fi in shared_faces:
            self._deleted_faces[fi] = True
            face = self.faces[fi]
            for vi in face:
                vi = int(vi)
                self.vert_faces[vi].discard(fi)
            # Remove from edge_faces
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            for eu, ev in [(a, b), (b, c), (a, c)]:
                ekey = (min(eu, ev), max(eu, ev))
                if ekey in self.edge_faces:
                    self.edge_faces[ekey].discard(fi)

        # Reassign v's remaining faces to u
        for fi in list(self.vert_faces[v]):
            if self._deleted_faces[fi]:
                continue
            face = self.faces[fi]
            # Remove old edge entries for this face
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            for eu, ev in [(a, b), (b, c), (a, c)]:
                ekey = (min(eu, ev), max(eu, ev))
                if ekey in self.edge_faces:
                    self.edge_faces[ekey].discard(fi)

            # Replace v with u in face
            for k in range(3):
                if face[k] == v:
                    face[k] = u

            # Re-add edge entries
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            for eu, ev in [(a, b), (b, c), (a, c)]:
                ekey = (min(eu, ev), max(eu, ev))
                if ekey not in self.edge_faces:
                    self.edge_faces[ekey] = set()
                self.edge_faces[ekey].add(fi)

            self.vert_faces[u].add(fi)

        # Update neighbor lists
        for w in list(self.vert_neighbors[v]):
            if w == u:
                continue
            self.vert_neighbors[w].discard(v)
            self.vert_neighbors[w].add(u)
            self.vert_neighbors[u].add(w)

        # Clean up v
        self.vert_neighbors[u].discard(v)
        self._deleted_verts[v] = True
        self.vert_neighbors[v] = set()
        self.vert_faces[v] = set()

        # Clean up edge_faces for the collapsed edge
        if key in self.edge_faces:
            del self.edge_faces[key]

        return affected | {u}

    def get_face_vertices(self, fi: int) -> tuple[int, int, int]:
        f = self.faces[fi]
        return int(f[0]), int(f[1]), int(f[2])

    def to_trimesh(self) -> TriMesh:
        active_v = np.where(~self._deleted_verts)[0]
        active_f = np.where(~self._deleted_faces)[0]
        remap = np.full(len(self.vertices), -1, dtype=np.int64)
        remap[active_v] = np.arange(len(active_v))
        new_verts = self.vertices[active_v]
        new_faces = remap[self.faces[active_f]]
        return TriMesh(vertices=new_verts, faces=new_faces)
