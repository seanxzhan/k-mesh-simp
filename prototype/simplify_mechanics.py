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
) -> TriMesh:
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
    """
    D = mq.plane_stress_D(E, nu)
    kb = mq.bending_coeff(E, nu, thickness)
    order = probe
    dim = mq.probe_dim(order)
    with_bending = bending_weight > 0.0
    adj = MeshAdjacency(mesh)

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

    return adj.to_trimesh()


if __name__ == "__main__":
    # tiny self-check on a coarse grid
    from kms.mesh import make_grid

    m = make_grid(21, 21)
    out = simplify_mechanics(m, target_verts=150, verbose=True)
    print(f"grid {m.n_verts} -> {out.n_verts} verts, {out.n_faces} faces")
