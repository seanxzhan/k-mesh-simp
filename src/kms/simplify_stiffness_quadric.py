"""Mesh simplification using stiffness quadrics.

Two modes:
  - "stiffness" (Approach 2): Pure stiffness quadric E(x) = (x-x₀)^T K_vv (x-x₀)
  - "combined" (Approach 3): Q_plane + λ·Q_stiffness — geometric fidelity + mechanical importance
"""
from __future__ import annotations

import heapq
import numpy as np
from scipy import sparse

from kms.mesh import TriMesh, face_areas as compute_face_areas
from kms.adjacency import MeshAdjacency
from kms.quadrics import Quadric
from kms.stiffness import shell_stiffness
from kms.stiffness_quadric import StiffnessQuadric, build_stiffness_quadrics, _extract_block


def _compute_schur_weights_for_vertex(
    K: sparse.spmatrix, v: int, neighbors: list[int],
) -> dict[int, float]:
    """Compute Schur-derived scalar skinning weights for vertex v.

    From the Schur formula: u_v = -K_vv^{-1} K_vB u_B
    The 3x3 coupling K_vv^{-1} @ K_vj gives a matrix weight per neighbor j.
    We reduce to scalar via Frobenius norm and normalize.

    Returns dict mapping neighbor_id -> scalar weight (summing to 1).
    """
    dofs_v = [3*v, 3*v+1, 3*v+2]
    K_vv = _extract_block(K, dofs_v, dofs_v)

    svd_vals = np.linalg.svd(K_vv, compute_uv=False)
    if svd_vals.max() < 1e-20:
        # K_vv is zero — fallback to uniform weights
        if neighbors:
            w = 1.0 / len(neighbors)
            return {nb: w for nb in neighbors}
        return {}

    K_vv_inv = np.linalg.pinv(K_vv, rcond=1e-12)

    weights = {}
    total = 0.0
    for nb in neighbors:
        dofs_nb = [3*nb, 3*nb+1, 3*nb+2]
        K_v_nb = _extract_block(K, dofs_v, dofs_nb)
        # Schur coupling magnitude
        blend = K_vv_inv @ K_v_nb
        w = np.linalg.norm(blend, 'fro')
        weights[nb] = w
        total += w

    if total > 1e-30:
        for nb in weights:
            weights[nb] /= total
    elif neighbors:
        w = 1.0 / len(neighbors)
        weights = {nb: w for nb in neighbors}

    return weights


def _finalize_skinning_weights(
    W: sparse.lil_matrix, adj: MeshAdjacency, n: int,
    elimination_order: list[tuple[int, dict[int, float]]],
) -> sparse.csc_matrix:
    """Finalize (n_fine, n_coarse) skinning weights by propagating eliminated columns.

    Problem: W[i, j] may reference an eliminated vertex j (its column will be dropped).
    Solution: process eliminated vertices in reverse order. For each eliminated v with
    Schur weights {nb: w_nb}, replace column v in W with the weighted sum of neighbor columns.

    This is equivalent to: W_final = W @ S where S propagates eliminated columns forward.
    """
    active = np.where(~adj._deleted_verts)[0]
    active_set = set(active.tolist())

    # Build lookup: for each eliminated vertex, its Schur weights
    elim_weights = {v: sw for v, sw in elimination_order}

    # Iteratively propagate: redistribute weight from eliminated columns
    # to their Schur neighbors until all weight lands in surviving columns.
    for iteration in range(len(elimination_order) + 1):
        # Find any remaining weight in eliminated columns
        has_eliminated_weight = False
        for v in elim_weights:
            col_v = W[:, v].toarray().ravel()
            nonzero_rows = np.where(np.abs(col_v) > 1e-15)[0]
            if len(nonzero_rows) == 0:
                continue
            has_eliminated_weight = True
            schur_weights = elim_weights[v]
            for i in nonzero_rows:
                w_iv = col_v[i]
                W[i, v] = 0
                for nb, w_nb in schur_weights.items():
                    W[i, nb] = W[i, nb] + w_iv * w_nb
        if not has_eliminated_weight:
            break

    return W[:, active].tocsc()


def simplify_stiffness_quadric(
    mesh: TriMesh,
    target_verts: int,
    mode: str = "combined",
    schur_mode: str = "additive",
    lam: float | None = None,
    E: float = 1.0,
    nu: float = 0.3,
    thickness: float = 0.001,
    use_line_quadric: bool = False,
    line_quadric_weight: float = 1e-3,
    compute_skinning_weights: bool = False,
    verbose: bool = False,
) -> TriMesh | tuple[TriMesh, sparse.csc_matrix]:
    """Simplify using stiffness quadrics.

    Args:
        mesh: Input triangle mesh
        target_verts: Target number of vertices
        mode: "stiffness" for pure stiffness quadric (Approach 2),
              "combined" for QEM + stiffness quadric (Approach 3)
        schur_mode: "additive" for Q_u + Q_v + correction (heuristic, stable),
                    "schur" for K'_ii = K_ii - K_ij K_jj⁻¹ K_ji (physically correct)
        lam: Weight for stiffness term in combined mode.
             If None, auto-calibrated so median stiffness cost ≈ median QEM cost.
        E: Young's modulus
        nu: Poisson's ratio
        thickness: Shell thickness
        use_line_quadric: Add line quadrics to QEM term (combined mode only)
        line_quadric_weight: Weight for line quadrics
        compute_skinning_weights: If True, return Schur-derived prolongation matrix
            W (n_fine × n_coarse) as physically-informed skinning weights.
        verbose: Print progress

    Returns:
        Simplified TriMesh (if compute_skinning_weights=False)
        (Simplified TriMesh, W) where W is (n_fine, n_coarse) skinning weight matrix
    """
    if mode == "stiffness":
        return _simplify_stiffness_only(mesh, target_verts, E, nu, thickness,
                                        schur_mode, compute_skinning_weights, verbose)
    elif mode == "combined":
        return _simplify_combined(mesh, target_verts, lam, E, nu, thickness,
                                  use_line_quadric, line_quadric_weight,
                                  schur_mode, compute_skinning_weights, verbose)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'stiffness' or 'combined'.")


def _simplify_stiffness_only(
    mesh: TriMesh,
    target_verts: int,
    E: float,
    nu: float,
    thickness: float,
    schur_mode: str,
    compute_skinning_weights: bool,
    verbose: bool,
) -> TriMesh | tuple[TriMesh, sparse.csc_matrix]:
    """Simplify using pure stiffness quadrics (Approach 2)."""
    adj = MeshAdjacency(mesh)
    n = mesh.n_verts

    # Build stiffness matrix and per-vertex quadrics
    K, _, _ = shell_stiffness(mesh, E, nu, thickness)
    vertex_quadrics = build_stiffness_quadrics(K, mesh)

    # W tracks Schur-derived prolongation weights.
    # W[i, j] = how much fine vertex i depends on original vertex j's displacement.
    # Starts as identity; when v is eliminated, its row becomes a blend of neighbors' rows.
    W = sparse.eye(n, format="lil") if compute_skinning_weights else None
    elimination_order: list[tuple[int, dict[int, float]]] = []

    if verbose:
        print(f"Setup: {n} verts, mode=stiffness_quadric")

    # Timestamp-based stale detection
    vertex_timestamps: dict[int, int] = {v: 0 for v in range(n)}
    current_ts = 0

    heap: list = []
    counter = 0

    def _push_edge(u: int, v: int):
        nonlocal counter
        if not adj.is_collapsible(u, v):
            return

        eq = vertex_quadrics[u] + vertex_quadrics[v]
        pos, success = eq.optimal_position()
        if not success:
            c_u = eq.compute_error(adj.vertices[u])
            c_v = eq.compute_error(adj.vertices[v])
            if c_u <= c_v:
                pos = adj.vertices[u].copy()
            else:
                pos = adj.vertices[v].copy()

        cost = eq.compute_error(pos)

        heapq.heappush(heap, (
            cost, counter, u, v, pos,
            vertex_timestamps[u], vertex_timestamps[v]
        ))
        counter += 1

    edges = adj.get_edges()
    for u, v in edges:
        _push_edge(u, v)

    if verbose:
        print(f"Heap built ({len(heap)} entries). Simplifying {n} -> {target_verts}...")

    total_collapses = n - target_verts
    n_collapsed = 0

    while adj.n_active_verts > target_verts and heap:
        cost, _, u, v, pos, ts_u, ts_v = heapq.heappop(heap)

        if not adj.is_valid_vertex(u) or not adj.is_valid_vertex(v):
            continue
        if ts_u != vertex_timestamps[u] or ts_v != vertex_timestamps[v]:
            continue
        if not adj.is_collapsible(u, v):
            continue

        # BEFORE collapse: compute Schur-derived skinning weights for v
        if W is not None:
            neighbors_v = [nb for nb in adj.vert_neighbors[v] if adj.is_valid_vertex(nb)]
            schur_w = _compute_schur_weights_for_vertex(K, v, neighbors_v)
            elimination_order.append((v, schur_w))
            # Set v's row to the blend (references original vertex columns)
            new_row = sparse.lil_matrix((1, n))
            for nb, w_nb in schur_w.items():
                new_row[0, nb] = w_nb  # direct column reference, not W[nb,:]
            W[v, :] = new_row

        # Collapse
        adj.collapse_edge(u, v, pos)
        n_collapsed += 1

        # Compute Schur correction blocks
        dofs_u = [3*u, 3*u+1, 3*u+2]
        dofs_v = [3*v, 3*v+1, 3*v+2]
        K_vv = _extract_block(K, dofs_v, dofs_v)
        K_uv = _extract_block(K, dofs_u, dofs_v)
        K_vu = _extract_block(K, dofs_v, dofs_u)

        svd_vals = np.linalg.svd(K_vv, compute_uv=False)
        if svd_vals.max() > 1e-20:
            K_vv_inv = np.linalg.pinv(K_vv, rcond=1e-12)
            correction = K_uv @ K_vv_inv @ K_vu
        else:
            correction = np.zeros((3, 3))

        if schur_mode == "additive":
            # Q_u + Q_v + correction as proper quadric centered at pos
            correction_quadric = StiffnessQuadric.from_stiffness_block(correction, pos)
            merged = vertex_quadrics[u] + vertex_quadrics[v]
            merged.A = merged.A + correction_quadric.A
            merged.b = merged.b + correction_quadric.b
            merged.c = merged.c + correction_quadric.c
            vertex_quadrics[u] = merged
        else:
            # Pure Schur: K'_ii = K_ii - K_ij K_jj⁻¹ K_ji
            K_ii_new = vertex_quadrics[u].A - correction
            vertex_quadrics[u] = StiffnessQuadric.from_stiffness_block(K_ii_new, pos)

        del vertex_quadrics[v]

        # Update timestamps
        current_ts += 1
        vertex_timestamps[u] = current_ts
        if v in vertex_timestamps:
            del vertex_timestamps[v]

        # Re-push edges involving u
        for nb in sorted(adj.vert_neighbors[u]):
            if adj.is_valid_vertex(nb):
                _push_edge(u, nb)

        if verbose and n_collapsed % 100 == 0:
            print(f"  {n_collapsed}/{total_collapses} collapses, {adj.n_active_verts} verts remaining")

    if verbose:
        print(f"Done: {n_collapsed} collapses, final {adj.n_active_verts} verts")

    result = adj.to_trimesh()

    if W is not None:
        W_out = _finalize_skinning_weights(W, adj, n, elimination_order)
        return result, W_out
    return result


def _simplify_combined(
    mesh: TriMesh,
    target_verts: int,
    lam: float | None,
    E: float,
    nu: float,
    thickness: float,
    use_line_quadric: bool,
    line_quadric_weight: float,
    schur_mode: str,
    compute_skinning_weights: bool,
    verbose: bool,
) -> TriMesh | tuple[TriMesh, sparse.csc_matrix]:
    """Simplify using combined QEM + stiffness quadric (Approach 3).

    Total quadric per vertex: Q_v = Q_plane_v + λ · Q_stiffness_v

    Both terms are standard quadratic forms (A, b, c) that accumulate additively.
    Q_plane keeps the mesh on-surface; Q_stiffness penalizes displacement in
    mechanically stiff directions.
    """
    adj = MeshAdjacency(mesh)
    n = mesh.n_verts

    # Build stiffness matrix and stiffness quadrics
    K, _, _ = shell_stiffness(mesh, E, nu, thickness)
    stiffness_quadrics = build_stiffness_quadrics(K, mesh)

    # W tracks Schur-derived prolongation weights
    W = sparse.eye(n, format="lil") if compute_skinning_weights else None
    elimination_order: list[tuple[int, dict[int, float]]] = []

    # Build QEM plane quadrics
    areas = compute_face_areas(mesh)
    face_quadrics: dict[int, Quadric] = {}
    for fi in range(mesh.n_faces):
        a, b, c = adj.get_face_vertices(fi)
        face_quadrics[fi] = Quadric.from_triangle(
            adj.vertices[a], adj.vertices[b], adj.vertices[c]
        )

    plane_quadrics: dict[int, Quadric] = {}
    for vi in range(n):
        fq = [face_quadrics[fi] for fi in adj.vert_faces[vi]]
        fa = [areas[fi] for fi in adj.vert_faces[vi]]
        plane_quadrics[vi] = Quadric.vertex_quadric(fq, fa)

        if use_line_quadric:
            vertex_normal = np.zeros(3)
            for fi in adj.vert_faces[vi]:
                a, b, c = adj.get_face_vertices(fi)
                normal = np.cross(adj.vertices[b] - adj.vertices[a], adj.vertices[c] - adj.vertices[a])
                vertex_normal += normal
            line_q = Quadric.from_line(adj.vertices[vi], vertex_normal, fa, line_quadric_weight)
            plane_quadrics[vi] += line_q

    # Auto-calibrate lambda if not specified
    if lam is None:
        # Sample edge costs from both metrics, set lambda so medians match
        sample_edges = adj.get_edges()[:min(200, len(adj.get_edges()))]
        qem_costs = []
        stiff_costs = []
        for u, v in sample_edges:
            eq_p = Quadric.edge_quadric(plane_quadrics[u], plane_quadrics[v])
            pos_p, _ = eq_p.optimal_position()
            if pos_p is not None:
                qem_costs.append(eq_p.compute_error(pos_p))

            eq_s = stiffness_quadrics[u] + stiffness_quadrics[v]
            pos_s, _ = eq_s.optimal_position()
            if pos_s is not None:
                stiff_costs.append(eq_s.compute_error(pos_s))

        med_qem = np.median(qem_costs) if qem_costs else 1.0
        med_stiff = np.median(stiff_costs) if stiff_costs else 1.0
        lam = med_qem / (med_stiff + 1e-30)
        if verbose:
            print(f"  Auto-calibrated lambda: {lam:.4e} (median QEM={med_qem:.2e}, median stiff={med_stiff:.2e})")

    # Build combined quadrics: Q_combined = Q_plane + lambda * Q_stiffness
    vertex_quadrics: dict[int, Quadric] = {}
    for vi in range(n):
        sq = stiffness_quadrics[vi]
        pq = plane_quadrics[vi]
        vertex_quadrics[vi] = Quadric(
            pq.A + lam * sq.A,
            pq.b + lam * sq.b,
            pq.c + lam * sq.c,
        )

    if verbose:
        print(f"Setup: {n} verts, mode=combined, lambda={lam:.4e}")

    # Timestamp-based stale detection
    vertex_timestamps: dict[int, int] = {v: 0 for v in range(n)}
    current_ts = 0

    heap: list = []
    counter = 0

    def _push_edge(u: int, v: int):
        nonlocal counter
        if not adj.is_collapsible(u, v):
            return

        eq = Quadric.edge_quadric(vertex_quadrics[u], vertex_quadrics[v])
        pos, success = eq.optimal_position()
        if not success:
            c_u = eq.compute_error(adj.vertices[u])
            c_v = eq.compute_error(adj.vertices[v])
            if c_u <= c_v:
                pos = adj.vertices[u].copy()
            else:
                pos = adj.vertices[v].copy()

        cost = eq.compute_error(pos)

        heapq.heappush(heap, (
            cost, counter, u, v, pos,
            vertex_timestamps[u], vertex_timestamps[v]
        ))
        counter += 1

    edges = adj.get_edges()
    for u, v in edges:
        _push_edge(u, v)

    if verbose:
        print(f"Heap built ({len(heap)} entries). Simplifying {n} -> {target_verts}...")

    total_collapses = n - target_verts
    n_collapsed = 0

    while adj.n_active_verts > target_verts and heap:
        cost, _, u, v, pos, ts_u, ts_v = heapq.heappop(heap)

        if not adj.is_valid_vertex(u) or not adj.is_valid_vertex(v):
            continue
        if ts_u != vertex_timestamps[u] or ts_v != vertex_timestamps[v]:
            continue
        if not adj.is_collapsible(u, v):
            continue

        # BEFORE collapse: compute Schur weights for v from its current neighbors
        if W is not None:
            neighbors_v = [nb for nb in adj.vert_neighbors[v] if adj.is_valid_vertex(nb)]
            schur_w = _compute_schur_weights_for_vertex(K, v, neighbors_v)
            elimination_order.append((v, schur_w))
            new_row = sparse.lil_matrix((1, n))
            for nb, w_nb in schur_w.items():
                new_row[0, nb] = w_nb
            W[v, :] = new_row

        # Collapse
        adj.collapse_edge(u, v, pos)
        n_collapsed += 1

        # Compute Schur correction
        dofs_u = [3*u, 3*u+1, 3*u+2]
        dofs_v = [3*v, 3*v+1, 3*v+2]
        K_vv = _extract_block(K, dofs_v, dofs_v)
        K_uv = _extract_block(K, dofs_u, dofs_v)
        K_vu = _extract_block(K, dofs_v, dofs_u)

        svd_vals = np.linalg.svd(K_vv, compute_uv=False)
        if svd_vals.max() > 1e-20:
            K_vv_inv = np.linalg.pinv(K_vv, rcond=1e-12)
            correction = K_uv @ K_vv_inv @ K_vu
        else:
            correction = np.zeros((3, 3))

        if schur_mode == "additive":
            # Q_u + Q_v + λ·correction as proper quadric
            corr_A = lam * correction
            corr_b = -corr_A @ pos
            corr_c = float(pos @ corr_A @ pos)
            merged = Quadric.edge_quadric(vertex_quadrics[u], vertex_quadrics[v])
            merged.A = merged.A + corr_A
            merged.b = merged.b + corr_b
            merged.c = merged.c + corr_c
            vertex_quadrics[u] = merged
        else:
            # QEM accumulates additively, stiffness uses pure Schur (subtract)
            qem_part = Quadric.edge_quadric(vertex_quadrics[u], vertex_quadrics[v])
            # Extract current stiffness A (subtract QEM contribution to isolate stiffness)
            # Approximation: treat the stiffness portion as the correction-affected part
            K_ii_new = vertex_quadrics[u].A - correction
            sq = StiffnessQuadric.from_stiffness_block(lam * K_ii_new, pos)
            merged = Quadric(
                qem_part.A - lam * vertex_quadrics[u].A + sq.A,  # replace old stiffness with new
                qem_part.b - lam * vertex_quadrics[u].b + sq.b,
                qem_part.c - lam * vertex_quadrics[u].c + sq.c,
            )
            vertex_quadrics[u] = merged

        del vertex_quadrics[v]

        # Update timestamps
        current_ts += 1
        vertex_timestamps[u] = current_ts
        if v in vertex_timestamps:
            del vertex_timestamps[v]

        # Re-push edges involving u
        for nb in sorted(adj.vert_neighbors[u]):
            if adj.is_valid_vertex(nb):
                _push_edge(u, nb)

        if verbose and n_collapsed % 100 == 0:
            print(f"  {n_collapsed}/{total_collapses} collapses, {adj.n_active_verts} verts remaining")

    if verbose:
        print(f"Done: {n_collapsed} collapses, final {adj.n_active_verts} verts")

    result = adj.to_trimesh()

    if W is not None:
        W_out = _finalize_skinning_weights(W, adj, n, elimination_order)
        return result, W_out
    return result
