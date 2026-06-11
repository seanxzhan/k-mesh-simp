"""Schur complement cost metrics for stiffness-based mesh simplification.

Two metrics for evaluating edge collapse cost:

1. Simple trace metric (per-vertex "Schur flow"):
   flow(v) = trace(K_vv^{-1} · S)  where S = Σ_j K_vj @ K_jv
   Measures how much stiffness flows THROUGH vertex v to its neighbors.

2. Full Schur mismatch (per-edge):
   Compares the actual post-collapse stiffness against the ideal
   Schur-complement-condensed stiffness on the affected patch.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from kms.mesh import TriMesh
from kms.adjacency import MeshAdjacency
from kms.stiffness import membrane_stiffness_cst, bending_stiffness_hinge


def _extract_block(K: sparse.spmatrix, row_dofs: list[int], col_dofs: list[int]) -> np.ndarray:
    """Extract a dense submatrix from sparse K."""
    return np.array(K[np.ix_(row_dofs, col_dofs)].todense())


def per_vertex_schur_flow(K: sparse.spmatrix, mesh: TriMesh) -> np.ndarray:
    """Per-vertex mechanical importance via Schur flow.

    For each vertex v, computes:
        flow(v) = trace(K_vv^{-1} · S)
    where S = Σ_{j ∈ N(v)} K_vj @ K_jv  (sum of coupling products over neighbors).

    High flow = vertex is an important mechanical conduit between its neighbors.
    Low flow = vertex can be removed cheaply.
    """
    n = mesh.n_verts
    flow = np.zeros(n)

    adj = MeshAdjacency(mesh)

    for v in range(n):
        dofs_v = [3*v, 3*v+1, 3*v+2]
        K_vv = _extract_block(K, dofs_v, dofs_v)

        det = np.linalg.det(K_vv)
        if abs(det) < 1e-30:
            flow[v] = 0.0
            continue

        K_vv_inv = np.linalg.inv(K_vv)

        S = np.zeros((3, 3))
        for j in adj.vert_neighbors[v]:
            dofs_j = [3*j, 3*j+1, 3*j+2]
            K_vj = _extract_block(K, dofs_v, dofs_j)
            K_jv = _extract_block(K, dofs_j, dofs_v)
            S += K_vj @ K_jv

        flow[v] = np.trace(K_vv_inv @ S)

    return flow


def edge_cost_simple(K: sparse.spmatrix, mesh: TriMesh) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Per-edge simple Schur cost: the Schur flow of the vertex being eliminated.

    For edge (u,v), we consider eliminating v. The cost is the total stiffness
    that would need to be redistributed = trace(K_vv^{-1} · K_vB @ K_Bv)
    where B = all neighbors of v.

    Returns (edges, costs) where edges[i] = (u,v) and costs[i] = cost.
    """
    adj = MeshAdjacency(mesh)
    edges = adj.get_edges()
    n_edges = len(edges)
    costs = np.zeros(n_edges)

    flow = per_vertex_schur_flow(K, mesh)

    for ei, (u, v) in enumerate(edges):
        # Cost of eliminating v (the second vertex in each edge pair)
        # Use the min of eliminating either endpoint
        costs[ei] = min(flow[u], flow[v])

    return edges, costs


def _local_element_stiffness_membrane(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
    E: float, nu: float, thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 9x9 membrane element stiffness for a single triangle.

    Returns (Ke_global, dofs) where dofs would be [3*i0,...,3*i2+2].
    Returns None for degenerate triangles.
    """
    D = (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2],
    ])

    e1_raw = p1 - p0
    e1_norm = np.linalg.norm(e1_raw)
    if e1_norm < 1e-16:
        return None
    e1 = e1_raw / e1_norm
    normal = np.cross(e1_raw, p2 - p0)
    area = 0.5 * np.linalg.norm(normal)
    if area < 1e-16:
        return None
    normal = normal / (2 * area)
    e2 = np.cross(normal, e1)

    x1 = np.dot(p1 - p0, e1)
    x2 = np.dot(p2 - p0, e1)
    y2 = np.dot(p2 - p0, e2)

    det_J = x1 * y2
    if abs(det_J) < 1e-16:
        return None

    B = (1.0 / det_J) * np.array([
        [y2, 0, -y2, 0, 0, 0],
        [0, x2 - x1, 0, -x2, 0, x1],
        [x2 - x1, y2, -x2, -y2, x1, 0],
    ])

    Ke_local = (area * thickness) * (B.T @ D @ B)

    R = np.column_stack([e1, e2])
    T = np.zeros((9, 6))
    T[0:3, 0:2] = R
    T[3:6, 2:4] = R
    T[6:9, 4:6] = R

    return T @ Ke_local @ T.T


def _local_element_stiffness_bending(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
    E: float, nu: float, thickness: float,
) -> np.ndarray | None:
    """Compute 12x12 bending element stiffness for a hinge [v0,v1,v2,v3].

    Edge (v0,v1), flap vertices v2 and v3.
    """
    kb = E * thickness**3 / (12.0 * (1.0 - nu**2))

    e = p1 - p0
    e_len = np.linalg.norm(e)
    if e_len < 1e-16:
        return None

    n0 = np.cross(e, p2 - p0)
    n1 = np.cross(e, p3 - p0)
    A0 = 0.5 * np.linalg.norm(n0)
    A1 = 0.5 * np.linalg.norm(n1)
    if A0 < 1e-16 or A1 < 1e-16:
        return None

    n0_hat = n0 / (2.0 * A0)
    n1_hat = n1 / (2.0 * A1)

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
    grad = np.concatenate([grad_v0, grad_v1, grad_v2, grad_v3])
    return coeff * np.outer(grad, grad)


def edge_cost_full(
    mesh: TriMesh,
    u: int,
    v: int,
    alpha: float = 0.5,
    E: float = 1.0,
    nu: float = 0.3,
    thickness: float = 0.001,
) -> float:
    """Full Schur mismatch cost for collapsing edge (u,v) at alpha.

    Compares the actual post-collapse local stiffness against the
    Schur-complement-condensed stiffness.

    Steps:
    1. Assemble K_local on the affected patch with u moved to p_w
    2. Schur-eliminate v's DOFs from K_local → K*
    3. Assemble K' on the collapsed topology (shared faces removed, v→u)
    4. Return ||K'_patch - K*_patch||²_F
    """
    adj = MeshAdjacency(mesh)

    if not adj.is_collapsible(u, v):
        return float('inf')

    p_w = (1 - alpha) * mesh.vertices[u] + alpha * mesh.vertices[v]

    # Identify the affected patch: all vertices in 1-ring of u or v
    patch_verts = sorted({u, v} | adj.vert_neighbors[u] | adj.vert_neighbors[v])
    patch_set = set(patch_verts)

    # Identify affected faces (touching u or v)
    affected_faces = set()
    for fi in adj.vert_faces[u]:
        affected_faces.add(fi)
    for fi in adj.vert_faces[v]:
        affected_faces.add(fi)

    # Identify shared faces (will be deleted on collapse)
    edge_key = (min(u, v), max(u, v))
    shared_faces = adj.edge_faces.get(edge_key, set())

    # Local DOF mapping for the patch
    patch_to_local = {vi: i for i, vi in enumerate(patch_verts)}
    n_patch = len(patch_verts)
    ndof_patch = 3 * n_patch

    def _get_positions_moved(vertex_id):
        """Get position with u moved to p_w."""
        if vertex_id == u:
            return p_w
        return mesh.vertices[vertex_id]

    # --- Step 1: Assemble K_moved on patch (u at p_w, full topology) ---
    K_moved = np.zeros((ndof_patch, ndof_patch))

    for fi in affected_faces:
        a, b, c = adj.get_face_vertices(fi)
        if a not in patch_set or b not in patch_set or c not in patch_set:
            continue
        pa = _get_positions_moved(a)
        pb = _get_positions_moved(b)
        pc = _get_positions_moved(c)

        Ke = _local_element_stiffness_membrane(pa, pb, pc, E, nu, thickness)
        if Ke is not None:
            local_dofs = []
            for vi in [a, b, c]:
                li = patch_to_local[vi]
                local_dofs.extend([3*li, 3*li+1, 3*li+2])
            for row in range(9):
                for col in range(9):
                    K_moved[local_dofs[row], local_dofs[col]] += Ke[row, col]

    # Add bending for hinges in the patch
    for fi in affected_faces:
        a, b, c = adj.get_face_vertices(fi)
        for local_opp in range(3):
            verts = [a, b, c]
            opp_v = verts[local_opp]
            ea = verts[(local_opp + 1) % 3]
            eb = verts[(local_opp + 2) % 3]
            ekey = (min(ea, eb), max(ea, eb))
            partner_faces = adj.edge_faces.get(ekey, set())
            if len(partner_faces) != 2:
                continue
            for fi2 in partner_faces:
                if fi2 == fi:
                    continue
                a2, b2, c2 = adj.get_face_vertices(fi2)
                opp2 = (set([a2, b2, c2]) - {ea, eb}).pop()
                if opp_v not in patch_set or opp2 not in patch_set:
                    continue
                if ea not in patch_set or eb not in patch_set:
                    continue
                p_ea = _get_positions_moved(ea)
                p_eb = _get_positions_moved(eb)
                p_opp = _get_positions_moved(opp_v)
                p_opp2 = _get_positions_moved(opp2)
                Ke = _local_element_stiffness_bending(p_ea, p_eb, p_opp, p_opp2, E, nu, thickness)
                if Ke is not None:
                    hinge_verts = [ea, eb, opp_v, opp2]
                    local_dofs = []
                    for vi in hinge_verts:
                        li = patch_to_local[vi]
                        local_dofs.extend([3*li, 3*li+1, 3*li+2])
                    for row in range(12):
                        for col in range(12):
                            K_moved[local_dofs[row], local_dofs[col]] += Ke[row, col]

    K_moved = 0.5 * (K_moved + K_moved.T)

    # --- Step 2: Schur-eliminate v from K_moved ---
    v_local = patch_to_local[v]
    dofs_v = [3*v_local, 3*v_local+1, 3*v_local+2]
    dofs_B = [i for i in range(ndof_patch) if i not in dofs_v]

    K_vv = K_moved[np.ix_(dofs_v, dofs_v)]
    K_vB = K_moved[np.ix_(dofs_v, dofs_B)]
    K_Bv = K_moved[np.ix_(dofs_B, dofs_v)]
    K_BB = K_moved[np.ix_(dofs_B, dofs_B)]

    det = np.linalg.det(K_vv)
    if abs(det) < 1e-30:
        return 0.0

    K_vv_inv = np.linalg.inv(K_vv)
    K_schur = K_BB - K_Bv @ K_vv_inv @ K_vB

    # --- Step 3: Assemble K' on collapsed topology ---
    # After collapse: v removed from patch, shared faces gone, v→u in remaining faces
    patch_after = [vi for vi in patch_verts if vi != v]
    patch_after_to_local = {vi: i for i, vi in enumerate(patch_after)}
    n_after = len(patch_after)
    ndof_after = 3 * n_after

    K_collapsed = np.zeros((ndof_after, ndof_after))

    for fi in affected_faces:
        if fi in shared_faces:
            continue
        a, b, c = adj.get_face_vertices(fi)
        # Remap v → u
        face_verts = [u if x == v else x for x in [a, b, c]]
        if any(x not in patch_after_to_local for x in face_verts):
            continue
        # Positions: u is at p_w
        positions = [p_w if x == u else mesh.vertices[x] for x in face_verts]

        Ke = _local_element_stiffness_membrane(positions[0], positions[1], positions[2], E, nu, thickness)
        if Ke is not None:
            local_dofs = []
            for vi in face_verts:
                li = patch_after_to_local[vi]
                local_dofs.extend([3*li, 3*li+1, 3*li+2])
            for row in range(9):
                for col in range(9):
                    K_collapsed[local_dofs[row], local_dofs[col]] += Ke[row, col]

    # Bending on collapsed topology
    # Rebuild edge-face mapping for affected faces after collapse
    collapsed_edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi in affected_faces:
        if fi in shared_faces:
            continue
        a, b, c = adj.get_face_vertices(fi)
        face_verts = tuple(u if x == v else x for x in [a, b, c])
        for local_opp in range(3):
            ea = face_verts[(local_opp + 1) % 3]
            eb = face_verts[(local_opp + 2) % 3]
            ekey = (min(ea, eb), max(ea, eb))
            if ekey not in collapsed_edge_faces:
                collapsed_edge_faces[ekey] = []
            collapsed_edge_faces[ekey].append((fi, face_verts[local_opp]))

    for ekey, face_list in collapsed_edge_faces.items():
        if len(face_list) != 2:
            continue
        ea, eb = ekey
        _, opp_a = face_list[0]
        _, opp_b = face_list[1]
        if any(x not in patch_after_to_local for x in [ea, eb, opp_a, opp_b]):
            continue
        p_ea = p_w if ea == u else mesh.vertices[ea]
        p_eb = p_w if eb == u else mesh.vertices[eb]
        p_oa = p_w if opp_a == u else mesh.vertices[opp_a]
        p_ob = p_w if opp_b == u else mesh.vertices[opp_b]

        Ke = _local_element_stiffness_bending(p_ea, p_eb, p_oa, p_ob, E, nu, thickness)
        if Ke is not None:
            hinge_verts = [ea, eb, opp_a, opp_b]
            local_dofs = []
            for vi in hinge_verts:
                li = patch_after_to_local[vi]
                local_dofs.extend([3*li, 3*li+1, 3*li+2])
            for row in range(12):
                for col in range(12):
                    K_collapsed[local_dofs[row], local_dofs[col]] += Ke[row, col]

    K_collapsed = 0.5 * (K_collapsed + K_collapsed.T)

    # --- Step 4: Compare K_schur vs K_collapsed ---
    # Map K_schur DOFs to match K_collapsed ordering
    # K_schur is indexed by dofs_B (patch ordering minus v)
    # K_collapsed is indexed by patch_after ordering
    # These should be the same vertex set, just need to align DOF indices.

    # Build mapping: patch_after uses its own local indices.
    # dofs_B corresponds to patch_verts minus v, in the original patch ordering.
    # We need to reorder K_schur to match patch_after ordering.
    patch_minus_v = [vi for vi in patch_verts if vi != v]
    # patch_minus_v == patch_after (same order since patch_verts is sorted and we just remove v)

    # K_schur rows/cols correspond to patch_minus_v in original patch order.
    # The original patch DOF ordering for vertex at position i in patch_verts is DOFs 3i,3i+1,3i+2.
    # After removing v's 3 DOFs, the remaining DOFs shift down.
    # dofs_B is already the correct list: it's the patch DOF indices minus v's DOFs.
    # We need to map from dofs_B ordering to patch_after_to_local ordering.

    # Actually both orderings are of patch_minus_v in sorted order (since patch_verts is sorted).
    # So K_schur and K_collapsed use the same vertex ordering.
    # The only difference: K_schur's indices are dense [0..ndof_after-1].

    cost = np.sum((K_collapsed - K_schur) ** 2)
    return float(cost)


def per_edge_costs_full(
    mesh: TriMesh,
    alpha: float = 0.5,
    E: float = 1.0,
    nu: float = 0.3,
    thickness: float = 0.001,
    verbose: bool = False,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Compute full Schur mismatch cost for all collapsible edges.

    Returns (edges, costs).
    """
    adj = MeshAdjacency(mesh)
    edges = adj.get_edges()
    n_edges = len(edges)
    costs = np.zeros(n_edges)

    for ei, (u, v) in enumerate(edges):
        if adj.is_collapsible(u, v):
            costs[ei] = edge_cost_full(mesh, u, v, alpha, E, nu, thickness)
        else:
            costs[ei] = float('inf')
        if verbose and (ei + 1) % 200 == 0:
            print(f"  edges: {ei+1}/{n_edges}", end="\r")

    if verbose:
        print(f"  edges: {n_edges}/{n_edges}")

    return edges, costs
