"""Stiffness-based mesh simplification using Schur complement costs.

Three modes:
  - "flow": Simple Schur flow metric (fast, no position dependence)
  - "full": Full Schur mismatch cost (accurate, position-dependent with 1D quadratic fit)
  - "qem+flow": QEM cost augmented with Schur flow weighting (fast, good positions)
"""
from __future__ import annotations

import heapq
import numpy as np
from scipy import sparse

from kms.mesh import TriMesh, face_areas as compute_face_areas
from kms.adjacency import MeshAdjacency
from kms.quadrics import Quadric
from kms.stiffness import shell_stiffness
from kms.schur import per_vertex_schur_flow, edge_cost_full


def simplify_schur(
    mesh: TriMesh,
    target_verts: int,
    mode: str = "full",
    E: float = 1.0,
    nu: float = 0.3,
    thickness: float = 0.1,
    use_line_quadric: bool = False,
    line_quadric_weight: float = 1e-3,
    verbose: bool = False,
) -> TriMesh:
    """Simplify a mesh using Schur complement stiffness cost.

    Args:
        mesh: Input triangle mesh
        target_verts: Target number of vertices
        mode: "flow" for simple Schur flow, "full" for full Schur mismatch,
              "qem+flow" for QEM augmented with Schur flow
        E: Young's modulus
        nu: Poisson's ratio
        thickness: Shell thickness
        use_line_quadric: Enable line quadrics in qem+flow mode (preserves sharp features)
        line_quadric_weight: Weight for line quadric regularization
        verbose: Print progress

    Returns:
        Simplified TriMesh
    """
    if mode == "flow":
        return _simplify_flow(mesh, target_verts, E, nu, thickness, verbose)
    elif mode == "full":
        return _simplify_full(mesh, target_verts, E, nu, thickness, verbose)
    elif mode == "qem+flow":
        return _simplify_qem_flow(mesh, target_verts, E, nu, thickness,
                                  use_line_quadric, line_quadric_weight, verbose)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'flow', 'full', or 'qem+flow'.")


def _simplify_flow(
    mesh: TriMesh,
    target_verts: int,
    E: float,
    nu: float,
    thickness: float,
    verbose: bool,
) -> TriMesh:
    """Simplify using Schur flow (simple trace metric).

    Cost of collapsing edge (u,v) = min(flow[u], flow[v]).
    The vertex with lower flow is eliminated (merged into the other).
    Position: midpoint (flow is position-independent).
    """
    adj = MeshAdjacency(mesh)
    n = mesh.n_verts

    K, _, _ = shell_stiffness(mesh, E, nu, thickness)
    flow = per_vertex_schur_flow(K, mesh)

    if verbose:
        print(f"Setup: {n} verts, mode=flow")
        print(f"  Flow range: [{flow.min():.2e}, {flow.max():.2e}]")

    # Timestamp-based stale detection
    vertex_timestamps: dict[int, int] = {v: 0 for v in range(n)}
    current_ts = 0

    # Build priority queue
    heap: list = []
    counter = 0

    def _push_edge(u: int, v: int):
        nonlocal counter
        if not adj.is_collapsible(u, v):
            return
        cost = min(flow[u], flow[v])
        heapq.heappush(heap, (
            cost, counter, u, v,
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
        cost, _, u, v, ts_u, ts_v = heapq.heappop(heap)

        if not adj.is_valid_vertex(u) or not adj.is_valid_vertex(v):
            continue
        if ts_u != vertex_timestamps[u] or ts_v != vertex_timestamps[v]:
            continue
        if not adj.is_collapsible(u, v):
            continue

        # Eliminate the vertex with lower flow (merge into the higher-flow one)
        if flow[u] < flow[v]:
            keep, elim = v, u
        else:
            keep, elim = u, v

        new_pos = 0.5 * (adj.vertices[keep] + adj.vertices[elim])
        adj.collapse_edge(keep, elim, new_pos)
        n_collapsed += 1

        # Update flow for the kept vertex: accumulate eliminated vertex's flow
        flow[keep] = flow[keep] + flow[elim]

        # Update timestamps
        current_ts += 1
        vertex_timestamps[keep] = current_ts
        if elim in vertex_timestamps:
            del vertex_timestamps[elim]

        # Re-push edges involving the kept vertex
        for nb in sorted(adj.vert_neighbors[keep]):
            if adj.is_valid_vertex(nb):
                _push_edge(keep, nb)

        if verbose and n_collapsed % 100 == 0:
            print(f"  {n_collapsed}/{total_collapses} collapses, {adj.n_active_verts} verts remaining")

    if verbose:
        print(f"Done: {n_collapsed} collapses, final {adj.n_active_verts} verts")

    return adj.to_trimesh()


def _simplify_full(
    mesh: TriMesh,
    target_verts: int,
    E: float,
    nu: float,
    thickness: float,
    verbose: bool,
) -> TriMesh:
    """Simplify using full Schur mismatch cost with 1D quadratic fit for alpha.

    For each edge, evaluates the full cost at alpha=0, 0.5, 1, fits a parabola,
    and uses the best alpha for collapse position.
    """
    adj = MeshAdjacency(mesh)
    n = mesh.n_verts

    if verbose:
        print(f"Setup: {n} verts, mode=full")

    # Timestamp-based stale detection
    vertex_timestamps: dict[int, int] = {v: 0 for v in range(n)}
    current_ts = 0

    # Build priority queue
    heap: list = []
    counter = 0

    def _current_mesh() -> TriMesh:
        return adj.to_trimesh()

    def _compute_edge_cost(u: int, v: int) -> tuple[float, float]:
        """Compute cost and best alpha for edge (u,v) using 1D quadratic fit."""
        current = adj.to_trimesh()

        # Remap u, v to new indices in the compacted mesh
        active = np.where(~adj._deleted_verts)[0]
        remap = np.full(len(adj.vertices), -1, dtype=np.int64)
        remap[active] = np.arange(len(active))
        u_new = int(remap[u])
        v_new = int(remap[v])

        if u_new < 0 or v_new < 0:
            return float('inf'), 0.5

        c0 = edge_cost_full(current, u_new, v_new, 0.0, E, nu, thickness)
        c5 = edge_cost_full(current, u_new, v_new, 0.5, E, nu, thickness)
        c1 = edge_cost_full(current, u_new, v_new, 1.0, E, nu, thickness)

        # Fit parabola: p(a) = A*a^2 + B*a + C
        # p(0)=c0, p(0.5)=c5, p(1)=c1
        a_coef = 2.0 * (c1 + c0 - 2.0 * c5)
        b_coef = c1 - c0 - a_coef

        best_alpha = 0.5
        best_cost = c5

        if abs(a_coef) > 1e-15 and a_coef > 0:
            alpha_star = float(np.clip(-b_coef / (2.0 * a_coef), 0.0, 1.0))
            cost_star = edge_cost_full(current, u_new, v_new, alpha_star, E, nu, thickness)
            if cost_star < best_cost:
                best_cost = cost_star
                best_alpha = alpha_star

        if c0 < best_cost:
            best_cost = c0
            best_alpha = 0.0
        if c1 < best_cost:
            best_cost = c1
            best_alpha = 1.0

        return best_cost, best_alpha

    def _push_edge(u: int, v: int):
        nonlocal counter
        if not adj.is_collapsible(u, v):
            return
        cost, alpha = _compute_edge_cost(u, v)
        heapq.heappush(heap, (
            cost, counter, u, v, alpha,
            vertex_timestamps[u], vertex_timestamps[v]
        ))
        counter += 1

    if verbose:
        print("Computing initial edge costs...")

    edges = adj.get_edges()
    n_edges = len(edges)
    for ei, (u, v) in enumerate(edges):
        _push_edge(u, v)
        if verbose and (ei + 1) % 50 == 0:
            print(f"  edges: {ei+1}/{n_edges}", end="\r")
    if verbose:
        print(f"  edges: {n_edges}/{n_edges}")
        print(f"Heap built ({len(heap)} entries). Simplifying {n} -> {target_verts}...")

    total_collapses = n - target_verts
    n_collapsed = 0

    while adj.n_active_verts > target_verts and heap:
        cost, _, u, v, alpha, ts_u, ts_v = heapq.heappop(heap)

        if not adj.is_valid_vertex(u) or not adj.is_valid_vertex(v):
            continue
        if ts_u != vertex_timestamps[u] or ts_v != vertex_timestamps[v]:
            continue
        if not adj.is_collapsible(u, v):
            continue

        # Collapse at optimal alpha: new_pos = (1-alpha)*u + alpha*v
        new_pos = (1 - alpha) * adj.vertices[u] + alpha * adj.vertices[v]
        adj.collapse_edge(u, v, new_pos)
        n_collapsed += 1

        # Update timestamps
        current_ts += 1
        vertex_timestamps[u] = current_ts
        if v in vertex_timestamps:
            del vertex_timestamps[v]

        # Re-push edges involving u
        for nb in sorted(adj.vert_neighbors[u]):
            if adj.is_valid_vertex(nb):
                _push_edge(u, nb)

        if verbose:
            pct = 100 * n_collapsed / max(total_collapses, 1)
            if n_collapsed % 10 == 0:
                print(f"  [{pct:5.1f}%] {n_collapsed}/{total_collapses} collapses, {adj.n_active_verts} verts remaining")

    if verbose:
        print(f"Done: {n_collapsed} collapses, final {adj.n_active_verts} verts")

    return adj.to_trimesh()


def _simplify_qem_flow(
    mesh: TriMesh,
    target_verts: int,
    E: float,
    nu: float,
    thickness: float,
    use_line_quadric: bool,
    line_quadric_weight: float,
    verbose: bool,
) -> TriMesh:
    """QEM simplification augmented with Schur flow weighting.

    Combined cost = QEM_error * (1 + flow_norm_of_eliminated_vertex)

    QEM gives good vertex positions (optimal placement via quadric minimization).
    Schur flow biases the priority: edges where the eliminated vertex has high
    mechanical importance get penalized, preserving stiff conduits longer.
    """
    adj = MeshAdjacency(mesh)
    n = mesh.n_verts

    # Build stiffness and compute flow
    K, _, _ = shell_stiffness(mesh, E, nu, thickness)
    flow = per_vertex_schur_flow(K, mesh)

    # Normalize flow to [0, 1] for stable weighting
    flow_max = flow.max()
    if flow_max > 0:
        flow_norm = flow / flow_max
    else:
        flow_norm = np.zeros(n)

    # Build QEM quadrics
    areas = compute_face_areas(mesh)
    face_quadrics: dict[int, Quadric] = {}
    for fi in range(mesh.n_faces):
        a, b, c = adj.get_face_vertices(fi)
        face_quadrics[fi] = Quadric.from_triangle(
            adj.vertices[a], adj.vertices[b], adj.vertices[c]
        )

    vertex_quadrics: dict[int, Quadric] = {}
    for vi in range(n):
        fq = [face_quadrics[fi] for fi in adj.vert_faces[vi]]
        fa = [areas[fi] for fi in adj.vert_faces[vi]]
        vertex_quadrics[vi] = Quadric.vertex_quadric(fq, fa)

        if use_line_quadric:
            vertex_normal = np.zeros(3)
            for fi in adj.vert_faces[vi]:
                a, b, c = adj.get_face_vertices(fi)
                normal = np.cross(adj.vertices[b] - adj.vertices[a], adj.vertices[c] - adj.vertices[a])
                vertex_normal += normal
            line_q = Quadric.from_line(adj.vertices[vi], vertex_normal, fa, line_quadric_weight)
            vertex_quadrics[vi] += line_q

    if verbose:
        print(f"Setup: {n} verts, mode=qem+flow")
        print(f"  Flow range: [{flow.min():.2e}, {flow.max():.2e}]")

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

        qem_cost = eq.compute_error(pos)

        # The eliminated vertex is the one with lower flow
        elim_flow = min(flow_norm[u], flow_norm[v])

        # Combined cost: QEM error weighted by mechanical importance
        combined_cost = qem_cost * (1.0 + elim_flow)

        heapq.heappush(heap, (
            combined_cost, counter, u, v, pos,
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

        adj.collapse_edge(u, v, pos)
        n_collapsed += 1

        # Accumulate quadric
        vertex_quadrics[u] = vertex_quadrics[u] + vertex_quadrics[v]
        del vertex_quadrics[v]

        # Accumulate flow
        flow_norm[u] = min(1.0, flow_norm[u] + flow_norm[v])

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

    return adj.to_trimesh()
