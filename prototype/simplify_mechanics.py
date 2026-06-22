"""Mechanical-QEM edge-collapse decimation --- Phase 3 of docs/mech_qem.tex.

Greedy edge collapse driven by the mechanical cost from mech_qem.py.  An edge
(u, v) collapsing to a merged position x* is priced by how much it perturbs the
homogenized membrane stiffness G = sum_e G_e:

    cost(u, v -> x*) = || G_after - G_fine ||_F^2
                     = || sum_{e in affected} ( G_e^new(x*) - G_e^old ) ||_F^2 ,

minimized over candidate placements x* in {midpoint, u, v}.  Only the triangles
incident to u or v change, so the cost is local.  This is the doc's "portable
additive backbone" (cost fork a), with metric W = identity and candidate-set
placement.

Structure mirrors kms.simplify_qem: a MeshAdjacency topology engine plus a
timestamped lazy priority queue.  The ONE structural difference is that the
mechanical cost depends on geometry, not on a frozen accumulated quadric -- so a
collapse changes the cost of every edge touching the merged 1-ring, and we must
re-push that whole neighborhood (QEM only re-pushes edges incident to u).

An optional geometric QEM term (geom_weight) blends in visual fidelity -- the
doc's alpha * E_geom fork.  Both terms are normalized by their initial medians so
geom_weight is dimensionless (geom_weight = 1 => "equal typical weight").

Membrane only.  Bending is a documented fork: collapsing one edge moves the
dihedral reference of every edge in the 1-ring, so the hinge must be re-evaluated
per collapse (see prototype/README.md).
"""

from __future__ import annotations

import heapq
import os
import sys

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech_qem as mq  # noqa: E402

from kms.mesh import TriMesh  # noqa: E402
from kms.adjacency import MeshAdjacency  # noqa: E402
from kms.quadrics import Quadric  # noqa: E402


def _face_G(adj: MeshAdjacency, fi: int, D, t, order="affine") -> np.ndarray:
    """Membrane probe quadric G_e of face fi at the current geometry."""
    a, b, c = adj.get_face_vertices(fi)
    pa, pb, pc = adj.vertices[a], adj.vertices[b], adj.vertices[c]
    Ke, _ = mq.cst_membrane_Ke(pa, pb, pc, D, t)
    dim = mq.probe_dim(order)
    if Ke is None:
        return np.zeros((dim, dim))
    V = mq._V_tri(pa, pb, pc, order)
    return V.T @ Ke @ V


def _hinge_G_sum(tris, posfn, kb, order, dim) -> np.ndarray:
    """Sum bending probe-quadrics over interior edges of `tris` that touch a
    *moved* triangle.  `tris`: iterable of (a, b, c, is_moved); `posfn(idx)`->xyz.

    Used to price the bending part of a collapse: enumerate the local hinges
    before vs after, with the merged geometry, and difference them.  Hinges with
    no moved triangle are skipped (they cancel in the before/after difference).
    """
    em: dict = {}
    for (a, b, c, mv) in tris:
        for p, q, w in ((a, b, c), (b, c, a), (a, c, b)):
            key = (p, q) if p < q else (q, p)
            em.setdefault(key, []).append((w, mv))
    G = np.zeros((dim, dim))
    for (a, b), lst in em.items():
        if len(lst) != 2:               # boundary within this local set -> no hinge
            continue
        (w1, mv1), (w2, mv2) = lst
        if not (mv1 or mv2):            # unaffected hinge -> cancels, skip
            continue
        nodes = (posfn(a), posfn(b), posfn(w1), posfn(w2))
        Ke = mq.hinge_bend_Ke(nodes[0], nodes[1], nodes[2], nodes[3], kb)
        if Ke is None:
            continue
        V = mq._stack_P(nodes, order)
        G += V.T @ Ke @ V
    return G


# --------------------------------------------------------------------------- #
#  Design B: emit the skinning prolongation P DURING decimation
#
#  Each collapse eliminates one vertex v (the survivor u keeps its index, moved to
#  x*).  Eliminating v is one Schur/condensation step: at the current geometry v's
#  motion follows its 1-ring by the LOCAL harmonic blend
#       d_v = sum_{j in N(v)} s_vj d_j,   s_vj = -K_vv^{-1} K_vj   (3x3 blocks),
#  with K assembled from the live elements touching v (membrane on incident faces
#  + bending on incident hinges).  Every element is rigid-translation invariant, so
#  sum_j s_vj = I -- the blend is a partition of unity (and that property composes).
#  Back-substituting the blends through the collapse sequence yields the full
#  fine<-coarse map P, d_fine = P d_coarse: the energy-optimal "harmonic blend"
#  cousin of the geometric alpha-blend a QEM decimator records.
#
#  The blend's local K uses prolong_thickness (~0.05), NOT the (thin) decimation
#  thickness: a too-thin shell leaves K_vv ill-conditioned out-of-plane -- the same
#  effect that breaks partition-of-unity in harmonic_skinning at t=1e-3.
# --------------------------------------------------------------------------- #
def _hinges_at_vertex(adj: MeshAdjacency, v: int):
    """Live interior-edge hinge stencils [e0, e1, w0, w1] whose 4-vertex stencil
    contains v -- as an edge endpoint AND as a wing (v's motion bends both)."""
    hinges = []
    seen = set()
    for w in adj.vert_neighbors[v]:                         # v an edge endpoint
        key = (min(v, w), max(v, w))
        fs = list(adj.edge_faces.get(key, ()))
        if len(fs) != 2:
            continue
        wings = [x for fi in fs for x in adj.get_face_vertices(fi) if x != v and x != w]
        if len(wings) != 2:
            continue
        hk = (key, tuple(sorted(wings)))
        if hk not in seen:
            seen.add(hk)
            hinges.append((v, w, wings[0], wings[1]))
    for fi in adj.vert_faces[v]:                            # v a wing (opposite edge)
        opp = [x for x in adj.get_face_vertices(fi) if x != v]
        if len(opp) != 2:
            continue
        p, q = opp
        key = (min(p, q), max(p, q))
        fs = list(adj.edge_faces.get(key, ()))
        if len(fs) != 2:
            continue
        e = next((x for fj in fs if fj != fi
                  for x in adj.get_face_vertices(fj) if x != p and x != q), None)
        if e is None:
            continue
        hk = (key, tuple(sorted((v, e))))
        if hk not in seen:
            seen.add(hk)
            hinges.append((p, q, v, e))
    return hinges


def _v_harmonic_blend(adj: MeshAdjacency, v: int, D, t, kb, reg=1e-8):
    """Local harmonic condensation of v: {neighbor j -> 3x3 s_vj} with
    sum_j s_vj = I.  Returns None if degenerate (caller hard-binds to survivor)."""
    row: dict = {}

    def add(Ke, stencil):
        iv = stencil.index(v)
        rblk = Ke[3 * iv:3 * iv + 3]                        # v's row block, 3 x 3k
        for p, j in enumerate(stencil):
            blk = rblk[:, 3 * p:3 * p + 3]
            row[j] = row[j] + blk if j in row else blk.copy()

    for fi in adj.vert_faces[v]:
        a, b, c = adj.get_face_vertices(fi)
        Ke, _ = mq.cst_membrane_Ke(adj.vertices[a], adj.vertices[b], adj.vertices[c], D, t)
        if Ke is not None:
            add(Ke, [a, b, c])
    for (h0, h1, h2, h3) in _hinges_at_vertex(adj, v):
        Ke = mq.hinge_bend_Ke(adj.vertices[h0], adj.vertices[h1],
                              adj.vertices[h2], adj.vertices[h3], kb)
        if Ke is not None:
            add(Ke, [h0, h1, h2, h3])

    Kvv = row.pop(v, None)
    if Kvv is None or not row:
        return None
    Kvv = Kvv + reg * (np.trace(Kvv) / 3.0 + 1e-30) * np.eye(3)
    return {j: -np.linalg.solve(Kvv, Kvj) for j, Kvj in row.items()}


def _v_meanvalue_blend(adj: MeshAdjacency, v: int, eps=1e-12):
    """Mean-value coordinates (Floater 2003) of v w.r.t. its 1-ring: a STIFFNESS-FREE
    blend {neighbor j -> w_vj * I3} with w_vj >= 0 and sum_j w_vj = 1, reproducing
    v = sum_j w_vj p_j (affine precision).  Computed at the current geometry from the
    fan angles at v and the neighbor distances -- no K, the smooth geometric cousin of
    the elastic _v_harmonic_blend.  Returns None if the 1-ring can't be ordered into a
    manifold fan (caller then binds v to its survivor)."""
    pv = adj.vertices[v]
    # order the 1-ring: each incident face (v, a, b) is a directed link edge a -> b
    succ: dict = {}
    for fi in adj.vert_faces[v]:
        tri = [int(x) for x in adj.get_face_vertices(fi)]
        i = tri.index(v)
        a, b = tri[(i + 1) % 3], tri[(i + 2) % 3]
        if a in succ:
            return None                                    # non-manifold fan
        succ[a] = b
    if not succ:
        return None
    starts = list(set(succ) - set(succ.values()))
    if not starts:                                         # closed fan (interior)
        start, closed = next(iter(succ)), True
    elif len(starts) == 1:                                 # open fan (boundary)
        start, closed = starts[0], False
    else:
        return None
    order, cur = [start], start
    while cur in succ and len(order) <= len(succ):
        nxt = succ[cur]
        if nxt == start:
            break
        order.append(nxt)
        cur = nxt
    if set(order) != set(adj.vert_neighbors[v]):
        return None
    n = len(order)
    d = np.array([adj.vertices[j] - pv for j in order], dtype=float)
    r = np.linalg.norm(d, axis=1)
    if np.any(r < eps):
        return None
    d = d / r[:, None]
    m = n if closed else n - 1                             # number of fan angles
    ht = np.array([np.tan(0.5 * np.arccos(np.clip(float(d[k] @ d[(k + 1) % n]),
                                                  -1.0, 1.0)))
                   for k in range(m)])
    w = np.zeros(n)
    for j in range(n):
        if closed:
            w[j] = (ht[(j - 1) % n] + ht[j]) / r[j]
        else:                                              # open: ends drop a term
            t_prev = ht[j - 1] if j >= 1 else 0.0
            t_next = ht[j] if j <= n - 2 else 0.0
            w[j] = (t_prev + t_next) / r[j]
    sw = w.sum()
    if not np.isfinite(sw) or sw <= eps:
        return None
    w /= sw
    return {order[j]: w[j] * np.eye(3) for j in range(n)}


def _compose_prolongation(elim, survivors, n_fine, mode="harmonic"):
    """Back-substitute the per-collapse blends into the sparse prolongation P,
    shape (3 n_fine, 3 n_coarse), with d_fine = P d_coarse.

    mode="harmonic" uses each collapse's recorded 1-ring blend; mode="edge" uses
    {survivor u -> I} (the faithful 2-point edge blend -> piecewise-constant).

    Resolved in REVERSE collapse order: when v was eliminated all its neighbors
    were live, hence each is either a coarse handle or a vertex eliminated LATER
    (already resolved earlier in this reversed pass) -- a clean back-substitution.
    """
    coarse_of = {int(o): k for k, o in enumerate(survivors)}
    handle_set = set(coarse_of)
    n_coarse = len(survivors)

    P_rows: dict = {int(h): {coarse_of[int(h)]: np.eye(3)} for h in survivors}
    for (v, u, blends) in reversed(elim):
        blend = {u: np.eye(3)} if mode == "edge" else blends[mode]
        rowP: dict = {}
        for j, B in blend.items():
            if j in handle_set:                             # neighbor is a handle
                k = coarse_of[j]
                rowP[k] = rowP[k] + B if k in rowP else B.copy()
            else:                                           # eliminated later -> resolved
                for k, Bjh in P_rows.get(j, {}).items():
                    BB = B @ Bjh
                    rowP[k] = rowP[k] + BB if k in rowP else BB
        P_rows[v] = rowP

    rows, cols, data = [], [], []
    for i in range(n_fine):
        for k, B in P_rows.get(i, {}).items():
            for rr in range(3):
                for cc in range(3):
                    val = B[rr, cc]
                    if val != 0.0:
                        rows.append(3 * i + rr)
                        cols.append(3 * k + cc)
                        data.append(val)
    return sparse.csr_matrix((data, (rows, cols)), shape=(3 * n_fine, 3 * n_coarse))


def prolongation_scalar_weights(P, n_fine=None, n_coarse=None) -> np.ndarray:
    """Scalar LBS weights w[i,k] = (1/3) tr(P_ik) from the 3x3-block prolongation;
    rows sum to 1 (partition of unity), so they paint as classic skinning weights."""
    n_fine = n_fine or P.shape[0] // 3
    n_coarse = n_coarse or P.shape[1] // 3
    Pc = P.tocoo()
    mask = (Pc.row % 3) == (Pc.col % 3)                     # diagonal of each 3x3 block
    W = np.zeros((n_fine, n_coarse))
    np.add.at(W, (Pc.row[mask] // 3, Pc.col[mask] // 3), Pc.data[mask] / 3.0)
    return W


def simplify_mechanics(
    mesh: TriMesh,
    target_verts: int,
    thickness: float = 1e-3,
    E: float = 1.0,
    nu: float = 0.3,
    placement: str = "quadratic",
    probe: str = "affine",
    bending_weight: float = 0.0,
    geom_weight: float = 0.0,
    line_quadric_weight: float = 1e-3,
    verbose: bool = False,
    return_survivors: bool = False,
    return_prolongation: bool = False,
    prolong_thickness: float = 0.05,
    prolong_mode: str = "harmonic",
):
    """Decimate `mesh` to `target_verts` using the mechanical-QEM cost.

    The collapse cost is a weighted sum of three independently-normalized terms:

        cost = membrane  +  bending_weight * bending  +  geom_weight * geom

    Each term is divided by the median of its own initial per-edge cost, so the
    membrane term is the implicit unit and bending_weight / geom_weight are
    DIMENSIONLESS: a weight of 1 means "this term matters as much as a typical
    membrane collapse". Both default to 0 (pure membrane). Useful range ~0-10.

    placement: how the merged vertex is positioned ON the edge w(a)=(1-a)u + a v:
        "quadratic" (default) -- sample cost(a) at a in {0, 0.5, 1}, fit a
            parabola p(a) through them, and place at its constrained minimizer
            a* in [0,1] (refined by re-evaluating the true cost there; we keep
            the best of {0, 0.5, 1, a*}). Optimal-position in spirit, but on the
            edge (the cost is not quadratic in a free 3-D x, so no QEM-style
            linear solve -- this is a 1-D edge line-search instead).
        "endpoints" -- classic min over {u, midpoint, v} only (no fit).
    probe="affine" (12-D) | "curvature" (30-D). bending requires the curvature
        probe to register (affine fields don't bend), so bending_weight > 0
        should be paired with probe="curvature" (viz_simplify does this for you).
    bending_weight > 0 adds the hinge term: the local hinges are re-evaluated
        before/after each candidate collapse (a collapse moves the dihedral
        reference of every 1-ring edge). Normalization makes it thickness-free.
    geom_weight > 0 blends a geometric QEM term (visual fidelity).

    return_prolongation=True (Design B) ALSO emits the fine<-coarse skinning map P
        (sparse, 3 n_fine x 3 n_coarse, d_fine = P d_coarse), built by composing a
        local harmonic condensation per collapse -- see the section comment above.
        The handle/proxy positions are kept (the optimized x*), so P drives the fine
        mesh directly from a deformation of the decimated coarse mesh (the PBD use
        case). Returns (coarse_mesh, survivors, P). prolong_thickness conditions P's
        local solves (decoupled from the decimation `thickness`).
    prolong_mode selects the per-collapse local rule for P (used only when
        return_prolongation=True):
        "harmonic" (default) -- v follows its whole 1-ring by the local elastic
            condensation s_vj = -K_vv^{-1} K_vj (smooth, signed weights, broad
            support; the only mode that uses K). Returns a single P.
        "edge" -- the FAITHFUL 2-POINT EDGE BLEND: the eliminated v simply follows
            its survivor u. The placement weight a is a fine->coarse averaging
            weight that cancels in any consistent coarse->fine map, so this composes
            to a piecewise-constant cluster skin (each fine vertex bound to exactly
            one handle, weight 1 -- sparsest, strictly nonneg/PoU, blockiest).
            Returns a single P.
        "geometric" -- STIFFNESS-FREE mean-value coordinates (Floater 2003) of v over
            its 1-ring: a convex blend (w_vj >= 0, sum = 1) reproducing v = sum w_vj
            p_j (affine precision), geometry only. Smoother than "edge", no K.
            Returns a single P.
        "all" -- returns {"harmonic", "edge", "geometric"} from ONE decimation (the
            collapse sequence is identical; only the recorded local rule differs).
    """
    D = mq.plane_stress_D(E, nu)
    kb = mq.bending_coeff(E, nu, thickness)
    prolong_kb = mq.bending_coeff(E, nu, prolong_thickness)  # for the skinning map P
    order = probe
    dim = mq.probe_dim(order)
    with_bending = bending_weight > 0.0
    assert prolong_mode in ("harmonic", "edge", "geometric", "all"), \
        f"bad prolong_mode: {prolong_mode}"
    adj = MeshAdjacency(mesh)
    elim: list = []      # (eliminated v, survivor u, {neighbor: 3x3 s_vj}) per collapse

    # per-face membrane quadric cache (current geometry, in the probe space)
    face_G = {fi: _face_G(adj, fi, D, thickness, order) for fi in range(mesh.n_faces)}

    def _core_neighbors(core):
        """Faces sharing an edge with a core face but not in core (for rim hinges)."""
        nb = set()
        for fi in core:
            a, b, c = adj.get_face_vertices(fi)
            for p, q in ((a, b), (b, c), (a, c)):
                for fj in adj.edge_faces.get((min(p, q), max(p, q)), ()):
                    if fj not in core:
                        nb.add(fj)
        return nb

    # optional geometric QEM quadrics (mirrors kms.simplify_qem)
    use_geom = geom_weight > 0.0
    vertex_quadrics: dict[int, Quadric] = {}
    if use_geom:
        from kms.mesh import face_areas as _fa
        areas = _fa(mesh)
        face_q = {}
        for fi in range(mesh.n_faces):
            a, b, c = adj.get_face_vertices(fi)
            face_q[fi] = Quadric.from_triangle(adj.vertices[a], adj.vertices[b], adj.vertices[c])
        for vi in range(mesh.n_verts):
            fq = [face_q[fi] for fi in adj.vert_faces[vi]]
            fa = [areas[fi] for fi in adj.vert_faces[vi]]
            vertex_quadrics[vi] = Quadric.vertex_quadric(fq, fa)
            n = np.zeros(3)
            for fi in adj.vert_faces[vi]:
                a, b, c = adj.get_face_vertices(fi)
                n += np.cross(adj.vertices[b] - adj.vertices[a], adj.vertices[c] - adj.vertices[a])
            vertex_quadrics[vi] += Quadric.from_line(adj.vertices[vi], n, fa, line_quadric_weight)

    # normalization scales: the median initial per-edge cost of each term, so a
    # weight of 1 means "this term counts as much as a typical membrane collapse".
    # All three weights (membrane is the implicit 1.0, bending_weight, geom_weight)
    # are therefore dimensionless and on the same footing.  (Normalizing bending
    # also makes its influence thickness-independent: dG_bend ~ t^3 cancels.)
    mem_scale = bend_scale = geom_scale = 1.0

    def raw_costs(u, v):
        """(cost, mem, bend, geom, pos): the on-edge placement minimizing the cost.

        cost = mem/mem_scale + bending_weight*bend/bend_scale + geom_weight*geom/geom_scale
        where mem = ||dG_membrane||_F^2 and bend = ||dG_bending||_F^2 are the
        squared changes in the (probe-space) membrane and bending operators, and
        geom is the QEM error.  The three terms are penalized separately so each
        weight normalizes independently (the cross term of the fully-assembled
        operator is dropped -- membrane and bending excite near-disjoint modes).

        The merged vertex is placed on the edge, w(a) = (1-a)u + a v.  For
        placement="quadratic" we fit a parabola to cost(a) at a in {0,0.5,1} and
        take its constrained minimizer a* in [0,1], then keep the best (by true
        cost) of {0, 0.5, 1, a*}.
        """
        key = (min(u, v), max(u, v))
        shared = adj.edge_faces.get(key, set())
        incident = adj.vert_faces[u] | adj.vert_faces[v]
        survivors = [fi for fi in incident if fi not in shared]
        Gm_old = np.zeros((dim, dim))
        for fi in incident:
            Gm_old += face_G[fi]

        # bending: pre-collapse local hinge sum (independent of the merged position)
        Gb_old = neighbors = None
        if with_bending:
            core = set(incident)
            neighbors = _core_neighbors(core)
            old_tris = [(*adj.get_face_vertices(fi), True) for fi in core] \
                     + [(*adj.get_face_vertices(fi), False) for fi in neighbors]
            Gb_old = _hinge_G_sum(old_tris, lambda i: adj.vertices[i], kb, order, dim)

        eq = (vertex_quadrics[u] + vertex_quadrics[v]) if use_geom else None
        pu, pv = adj.vertices[u], adj.vertices[v]

        def eval_pos(x):
            """(cost, mem, bend, geom) for the merged vertex placed at x."""
            Gm_new = np.zeros((dim, dim))
            for fi in survivors:
                tri = adj.faces[fi]
                pts = [x if (int(i) == u or int(i) == v) else adj.vertices[int(i)] for i in tri]
                Ke, _ = mq.cst_membrane_Ke(pts[0], pts[1], pts[2], D, thickness)
                if Ke is None:
                    continue
                Vv = mq._V_tri(pts[0], pts[1], pts[2], order)
                Gm_new += Vv.T @ Ke @ Vv
            mem = float(np.sum((Gm_new - Gm_old) ** 2))

            bend = 0.0
            if with_bending:
                new_tris = [(u if int(a) == v else int(a),
                             u if int(b) == v else int(b),
                             u if int(c) == v else int(c), True)
                            for fi in survivors for (a, b, c) in (adj.faces[fi],)] \
                         + [(*adj.get_face_vertices(fi), False) for fi in neighbors]
                Gb_new = _hinge_G_sum(new_tris, lambda i: x if i == u else adj.vertices[i],
                                      kb, order, dim)
                bend = float(np.sum((Gb_new - Gb_old) ** 2))

            geom = float(eq.compute_error(x)) if use_geom else 0.0
            cost = (mem / mem_scale
                    + bending_weight * bend / bend_scale
                    + geom_weight * geom / geom_scale)
            return cost, mem, bend, geom

        # sample the on-edge cost at alpha = 0 (u), 0.5 (midpoint), 1 (v)
        samples = {0.0: eval_pos(pu),
                   0.5: eval_pos(0.5 * (pu + pv)),
                   1.0: eval_pos(pv)}

        if placement == "quadratic":
            c0, ch, c1 = samples[0.0][0], samples[0.5][0], samples[1.0][0]
            # parabola p(a) = A a^2 + B a + C through (0,c0),(0.5,ch),(1,c1)
            A = 2.0 * (c0 + c1 - 2.0 * ch)
            B = 4.0 * ch - 3.0 * c0 - c1
            if A > 1e-30:                         # convex: interior min, clamped on-edge
                astar = min(1.0, max(0.0, -B / (2.0 * A)))
            else:                                 # flat/concave: minimizer is an endpoint
                astar = 0.0 if c0 <= c1 else 1.0
            if all(abs(astar - a) > 1e-6 for a in samples):
                samples[astar] = eval_pos((1.0 - astar) * pu + astar * pv)

        a_best = min(samples, key=lambda a: samples[a][0])
        cost, mem, bend, geom = samples[a_best]
        pos = (1.0 - a_best) * pu + a_best * pv
        return cost, mem, bend, geom, pos

    # --- compute the normalization scales from the initial mesh (scales = 1 here) ---
    if with_bending or use_geom:
        ms, bs, gs = [], [], []
        for (u, v) in adj.get_edges():
            _, mem, bend, geom, _ = raw_costs(u, v)
            ms.append(mem)
            bs.append(bend)
            gs.append(geom)
        mem_scale = float(np.median(ms)) + 1e-30
        if with_bending:
            bend_scale = float(np.median(bs)) + 1e-30
        if use_geom:
            geom_scale = float(np.median(gs)) + 1e-30

    # --- timestamped lazy heap (same scheme as kms.simplify_qem) ---
    vertex_ts = {v: 0 for v in range(mesh.n_verts)}
    current_ts = 0
    heap: list = []
    counter = 0

    def push_edge(u, v):
        nonlocal counter
        cost, _, _, _, pos = raw_costs(u, v)
        heapq.heappush(heap, (cost, counter, u, v, pos, vertex_ts[u], vertex_ts[v]))
        counter += 1

    if verbose:
        print(f"  probe={order}  bending_weight={bending_weight:g}  geom_weight={geom_weight:g}")
        print(f"  scales (median init cost): mem={mem_scale:.3e}"
              + (f" bend={bend_scale:.3e}" if with_bending else "")
              + (f" geom={geom_scale:.3e}" if use_geom else ""))
        print(f"  building heap over {len(adj.get_edges())} edges...")
    for u, v in adj.get_edges():
        push_edge(u, v)

    n_start = adj.n_active_verts
    if verbose:
        print(f"  simplifying {n_start} -> {target_verts} verts...")

    n_collapsed = 0
    while adj.n_active_verts > target_verts and heap:
        cost, _, u, v, pos, ts_u, ts_v = heapq.heappop(heap)

        if not adj.is_valid_vertex(u) or not adj.is_valid_vertex(v):
            continue
        if ts_u != vertex_ts[u] or ts_v != vertex_ts[v]:
            continue
        if not adj.is_collapsible(u, v):
            continue

        shared = set(adj.edge_faces.get((min(u, v), max(u, v)), set()))

        if return_prolongation:
            # record how the eliminated vertex v follows its 1-ring (current geom),
            # BEFORE the collapse rewires topology. u (survivor) is among v's
            # neighbors. Each requested map gets its own local blend: "harmonic" =
            # elastic condensation -K_vv^-1 K_vj; "geometric" = stiffness-free
            # mean-value coordinates; "edge" needs only the survivor u (stored below)
            # -> piecewise-constant. "all" emits every blend from one decimation.
            blends: dict = {}
            if prolong_mode in ("harmonic", "all"):
                s = _v_harmonic_blend(adj, v, D, prolong_thickness, prolong_kb)
                blends["harmonic"] = s if s is not None else {u: np.eye(3)}
            if prolong_mode in ("geometric", "all"):
                g = _v_meanvalue_blend(adj, v)
                blends["geometric"] = g if g is not None else {u: np.eye(3)}
            elim.append((int(v), int(u),
                         {m: {int(j): b for j, b in bl.items()}
                          for m, bl in blends.items()}))

        affected = adj.collapse_edge(u, v, pos)
        n_collapsed += 1

        if use_geom:
            vertex_quadrics[u] = vertex_quadrics[u] + vertex_quadrics[v]
            del vertex_quadrics[v]

        # refresh the face-quadric cache: drop collapsed faces, recompute the
        # merged vertex's incident faces (their geometry changed)
        for fi in shared:
            face_G.pop(fi, None)
        for fi in adj.vert_faces[u]:
            face_G[fi] = _face_G(adj, fi, D, thickness, order)

        # geometry-dependent cost => invalidate + re-push the whole merged 1-ring,
        # not just edges incident to u (QEM can; we cannot).
        ring = {u} | set(adj.vert_neighbors[u])
        current_ts += 1
        for a in ring:
            if adj.is_valid_vertex(a):
                vertex_ts[a] = current_ts
        if v in vertex_ts:
            del vertex_ts[v]

        seen = set()
        for a in ring:
            if not adj.is_valid_vertex(a):
                continue
            for nb in adj.vert_neighbors[a]:
                if not adj.is_valid_vertex(nb):
                    continue
                e = (min(a, nb), max(a, nb))
                if e not in seen:
                    seen.add(e)
                    push_edge(e[0], e[1])

        if verbose and n_collapsed % 50 == 0:
            cur = adj.n_active_verts
            frac = (n_start - cur) / max(1, n_start - target_verts)
            print(f"    [{min(100, int(100 * frac)):3d}%] {cur} verts "
                  f"({n_collapsed} collapses, last cost={cost:.3e})")

    if verbose:
        print(f"  done: {n_collapsed} collapses, final {adj.n_active_verts} verts")

    result = adj.to_trimesh()
    # original vertex indices that survived (same order to_trimesh uses, so coarse
    # vertex k corresponds to original index survivors[k])
    survivors = np.where(~adj._deleted_verts)[0]
    if return_prolongation:
        if prolong_mode == "all":
            P = {m: _compose_prolongation(elim, survivors, mesh.n_verts, m)
                 for m in ("harmonic", "edge", "geometric")}
            items = list(P.items())
        else:
            P = _compose_prolongation(elim, survivors, mesh.n_verts, prolong_mode)
            items = [(prolong_mode, P)]
        if verbose:
            for nm, M in items:
                print(f"  prolongation[{nm}]: {M.shape}, {M.nnz} nnz "
                      f"({M.nnz / (3 * mesh.n_verts):.1f} per fine dof)")
        return result, survivors, P
    if return_survivors:
        return result, survivors
    return result


if __name__ == "__main__":
    # tiny self-check on a coarse grid
    from kms.mesh import make_grid

    m = make_grid(21, 21)
    out = simplify_mechanics(m, target_verts=150, verbose=True)
    print(f"grid {m.n_verts} -> {out.n_verts} verts, {out.n_faces} faces")

    # Design B: emit + validate ALL skinning prolongations (bending-aware):
    #   harmonic = local elastic 1-ring condensation;  edge = faithful 2-point edge
    #   blend (-> piecewise constant);  geometric = stiffness-free mean-value coords.
    coarse, survivors, Ps = simplify_mechanics(
        m, target_verts=150, probe="curvature", bending_weight=1.0,
        return_prolongation=True, prolong_mode="all", verbose=True)
    nf, nc = m.n_verts, len(survivors)
    coarse_of = {int(o): k for k, o in enumerate(survivors)}
    t = np.array([0.3, -0.7, 1.1])
    for name, P in Ps.items():
        assert P.shape == (3 * nf, 3 * nc), (name, P.shape)
        W = prolongation_scalar_weights(P)
        pou = float(np.abs(W.sum(axis=1) - 1.0).max())
        df = (P @ np.tile(t, nc)).reshape(-1, 3)
        trans = float(np.abs(df - t).max())
        hid = max(float(np.abs(P[3 * h:3 * h + 3].toarray()[:, 3 * coarse_of[int(h)]:
                                                             3 * coarse_of[int(h)] + 3]
                               - np.eye(3)).max()) for h in survivors[:20])
        extra = ""
        if name == "edge":
            nz = (W > 1e-9).sum(axis=1)
            assert int(nz.max()) == 1 and int(nz.min()) == 1, \
                f"edge not piecewise-constant: {int(nz.min())}..{int(nz.max())} handles/vertex"
            extra = "  (piecewise-constant: 1 handle/vertex)"
        print(f"  P[{name}]: partition-of-unity err={pou:.2e}  translation err={trans:.2e}  "
              f"handle-identity err={hid:.2e}{extra}")
        assert pou < 1e-5 and trans < 1e-5 and hid < 1e-10, (name, pou, trans, hid)
    print("  prolongation self-check: PASS (harmonic + edge + geometric)")
