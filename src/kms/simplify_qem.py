from __future__ import annotations

import heapq
import numpy as np
from scipy import sparse

from kms.mesh import TriMesh, face_areas as compute_face_areas
from kms.adjacency import MeshAdjacency
from kms.quadrics import Quadric


def simplify_qem(
    mesh: TriMesh,
    target_verts: int,
    use_optimal_position: bool = True,
    use_line_quadric: bool = True,
    line_quadric_weight: float = 1e-3,
    use_boundary_quadric: bool = True,
    boundary_quadric_weight: float = 1.0,
    compute_restriction: bool = False,
    verbose: bool = False,
) -> TriMesh | tuple[TriMesh, sparse.csc_matrix]:
    """Simplify a mesh using Quadric Error Metrics (Garland & Heckbert 1997).

    Args:
        mesh: Input triangle mesh
        target_verts: Target number of vertices
        use_optimal_position: If True, use SVD to find optimal merge point.
                             If False, pick the better of the two endpoints.
        use_line_quadric: Add a line quadric per vertex (preserve sharp features).
        use_boundary_quadric: Add a perpendicular-plane quadric per open edge so
            simplification preserves the boundary curve (no-op on closed meshes).
        boundary_quadric_weight: Penalty weight for the boundary quadric.
        compute_restriction: If True, also return the restriction matrix P
            such that V_coarse ~ P @ V_fine. P has non-negative entries with
            rows summing to 1.
        verbose: Print progress

    Returns:
        Simplified TriMesh (if compute_restriction=False)
        (Simplified TriMesh, P) where P is (n_coarse, n_fine) restriction matrix
    """
    adj = MeshAdjacency(mesh)
    n_fine = mesh.n_verts

    # Restriction matrix P: accumulates as P = Q_n ... Q_2 Q_1
    # Each Q_i is the elementary restriction for one collapse.
    # Start with identity; will be updated each collapse.
    P = sparse.eye(n_fine, format="csc") if compute_restriction else None

    # Compute face quadrics
    face_quadrics: dict[int, Quadric] = {}
    areas = compute_face_areas(mesh)
    for fi in range(mesh.n_faces):
        a, b, c = adj.get_face_vertices(fi)
        face_quadrics[fi] = Quadric.from_triangle(
            adj.vertices[a], adj.vertices[b], adj.vertices[c]
        )

    # Compute vertex quadrics (area-weighted sum of incident face quadrics + line quadric)
    vertex_quadrics: dict[int, Quadric] = {}
    for vi in range(mesh.n_verts):
        fq = [face_quadrics[fi] for fi in adj.vert_faces[vi]]
        fa = [areas[fi] for fi in adj.vert_faces[vi]]
        vertex_quadrics[vi] = Quadric.vertex_quadric(fq, fa)

        if use_line_quadric:
            # Compute area-weighted vertex normal
            vertex_normal = np.zeros(3)
            for fi in adj.vert_faces[vi]:
                a, b, c = adj.get_face_vertices(fi)
                n = np.cross(adj.vertices[b] - adj.vertices[a], adj.vertices[c] - adj.vertices[a])
                vertex_normal += n  # already area-weighted (cross product magnitude ~ 2*area)
            line_q = Quadric.from_line(adj.vertices[vi], vertex_normal, fa, line_quadric_weight)
            vertex_quadrics[vi] += line_q

    # Boundary preservation: for each open edge, bake a perpendicular-plane
    # quadric into BOTH endpoints (Garland & Heckbert 1997). No-op when the mesh
    # has no boundary edges (e.g. a closed surface).
    if use_boundary_quadric:
        for u, v in adj.get_edges():
            if not adj.is_boundary_edge(u, v):
                continue
            fi = next(iter(adj.edge_faces[(min(u, v), max(u, v))]))
            a, b, c = adj.get_face_vertices(fi)
            face_normal = np.cross(
                adj.vertices[b] - adj.vertices[a], adj.vertices[c] - adj.vertices[a]
            )
            bq = Quadric.from_boundary_edge(
                adj.vertices[u], adj.vertices[v], face_normal, boundary_quadric_weight
            )
            vertex_quadrics[u] += bq
            vertex_quadrics[v] += bq

    # Timestamp-based stale detection
    vertex_timestamps: dict[int, int] = {v: 0 for v in range(mesh.n_verts)}
    current_ts = 0

    # Build priority queue
    heap: list = []
    counter = 0  # deterministic tie-breaking

    def _push_edge(u: int, v: int):
        nonlocal counter
        eq = Quadric.edge_quadric(vertex_quadrics[u], vertex_quadrics[v])

        if use_optimal_position:
            pos, success = eq.optimal_position()
            if not success:
                pos, _ = _pick_better_endpoint(eq, u, v)
            cost = eq.compute_error(pos)
        else:
            pos, cost = _pick_better_endpoint(eq, u, v)

        heapq.heappush(heap, (
            cost, counter, u, v, pos,
            vertex_timestamps[u], vertex_timestamps[v]
        ))
        counter += 1

    def _pick_better_endpoint(eq: Quadric, u: int, v: int) -> tuple[np.ndarray, float]:
        c_u = eq.compute_error(adj.vertices[u])
        c_v = eq.compute_error(adj.vertices[v])
        if c_u <= c_v:
            return adj.vertices[u].copy(), c_u
        return adj.vertices[v].copy(), c_v

    if verbose:
        print(f"Building heap for {len(adj.get_edges())} edges...")

    for u, v in adj.get_edges():
        _push_edge(u, v)

    if verbose:
        print(f"Simplifying {mesh.n_verts} -> {target_verts} verts...")

    n_collapsed = 0
    while adj.n_active_verts > target_verts and heap:
        cost, _, u, v, pos, ts_u, ts_v = heapq.heappop(heap)

        # Skip stale entries
        if not adj.is_valid_vertex(u) or not adj.is_valid_vertex(v):
            continue
        if ts_u != vertex_timestamps[u] or ts_v != vertex_timestamps[v]:
            continue

        # Check link condition
        if not adj.is_collapsible(u, v):
            continue

        # Build restriction matrix Q for this collapse before modifying adjacency
        if P is not None:
            active = np.where(~adj._deleted_verts)[0]
            n_active = len(active)
            remap = np.full(len(adj.vertices), -1, dtype=np.int64)
            remap[active] = np.arange(n_active)
            u_local = int(remap[u])
            v_local = int(remap[v])

            # Compute alpha: how far along (u -> v) is the new position?
            edge_vec = adj.vertices[v] - adj.vertices[u]
            edge_len_sq = np.dot(edge_vec, edge_vec)
            if edge_len_sq > 1e-30:
                alpha = float(np.dot(pos - adj.vertices[u], edge_vec) / edge_len_sq)
                alpha = np.clip(alpha, 0.0, 1.0)
            else:
                alpha = 0.5

            # Q: (n_active-1) x n_active matrix
            # All rows are identity except merged row which blends u and v
            keep = [i for i in range(n_active) if i != v_local]
            n_out = len(keep)
            new_idx = np.full(n_active, -1, dtype=np.int64)
            for new_i, old_i in enumerate(keep):
                new_idx[old_i] = new_i

            q_rows, q_cols, q_vals = [], [], []
            for old_i in keep:
                new_i = new_idx[old_i]
                if old_i == u_local:
                    q_rows.extend([new_i, new_i])
                    q_cols.extend([u_local, v_local])
                    q_vals.extend([1.0 - alpha, alpha])
                else:
                    q_rows.append(new_i)
                    q_cols.append(old_i)
                    q_vals.append(1.0)

            Q = sparse.csc_matrix((q_vals, (q_rows, q_cols)), shape=(n_out, n_active))
            P = Q @ P

        # Collapse: merge v into u at pos
        affected = adj.collapse_edge(u, v, pos)
        n_collapsed += 1

        # Update quadric for merged vertex
        vertex_quadrics[u] = vertex_quadrics[u] + vertex_quadrics[v]
        del vertex_quadrics[v]

        # Update timestamp
        current_ts += 1
        vertex_timestamps[u] = current_ts
        if v in vertex_timestamps:
            del vertex_timestamps[v]

        # Re-push edges involving u
        for nb in sorted(adj.vert_neighbors[u]):
            if adj.is_valid_vertex(nb):
                _push_edge(u, nb)

        if verbose and n_collapsed % 100 == 0:
            print(f"  {n_collapsed} collapses, {adj.n_active_verts} verts remaining")

    if verbose:
        print(f"Done: {n_collapsed} collapses, final {adj.n_active_verts} verts")

    result = adj.to_trimesh()
    if P is not None:
        return result, P
    return result
