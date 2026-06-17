"""Mechanical-QEM cost prototype --- Phases 0-2 of docs/mech_qem.tex.

A *self-contained, principled* re-implementation of the costs in mech_qem.tex.
Deliberately ignores the earlier exploratory modules (schur.py, stiffness_quadric.py,
simplify_schur.py, simplify_stiffness_quadric.py); we restart from the doc.

What it implements
------------------
Phase 0 -- assemble per-ELEMENT stiffness blocks (not the global matrix):
    * CST membrane triangle   K_e in R^{9x9},  rank 3   (3 verts x 3 dof)
    * discrete bending hinge   K_h in R^{12x12}, rank 1  (4 verts x 3 dof)
  (Element formulas follow kms.stiffness in structure but FIX a sign bug in its
   CST B-matrix -- see cst_membrane_Ke and prototype/README.md.)

Phase 1 -- choose the probe subspace V.  Default fork: the 6 affine /
  constant-strain modes (F symmetric, c = 0).  A node's displacement under an
  affine field d_n = F X_n + c factors through that node's rest position:
        d_n = P(X_n) a,   a = (vec(F), c) in R^12,   P(X) in R^{3x12}.

Phase 2 -- project each element into probe coordinates and accumulate:
        G_e = V_e^T K_e V_e  in R^{12x12}            (per element)
        G_v = sum_{e ni v} G_e                        (per vertex, QEM-style)
        G   = sum_e G_e                               (global, PSD, eff. rank 6)

Costs
-----
Layer A (response-template magnitude -- "how much homogenized response lives here"):
    per-triangle : probe energy of the membrane G_e
    per-vertex   : probe energy of accumulated membrane / bending / total G_v

Layer B (the actual decimation cost -- fine-vs-coarse discrepancy):
    per-edge -> per-vertex : ||G_after - G_fine||_W^2 over the affected elements,
    membrane-only in this v1, additive backbone (fork a), candidate-set placement,
    metric W = identity.

Honest caveat (doc divergence 2): the per-element / per-vertex quadric is a
*response template*, not yet an error.  Mechanical error is the fine-vs-coarse
discrepancy, which is what Layer B measures.  Layer A is therefore a "response
lives here" field, not a collapse cost --- both are shown so the distinction is
visible.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from kms.mesh import TriMesh


# --------------------------------------------------------------------------- #
#  Constitutive
# --------------------------------------------------------------------------- #
def _cross(a, b):
    """Fast 3-vector cross product (np.cross has heavy axis overhead in tight loops)."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def _norm3(a) -> float:
    return float((a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) ** 0.5)


def plane_stress_D(E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    """Plane-stress constitutive matrix (in-plane Voigt 3x3)."""
    return (E / (1.0 - nu**2)) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
    )


def bending_coeff(E: float = 1.0, nu: float = 0.3, thickness: float = 1e-3) -> float:
    """Hinge bending modulus  E t^3 / (12 (1 - nu^2))."""
    return E * thickness**3 / (12.0 * (1.0 - nu**2))


# --------------------------------------------------------------------------- #
#  Phase 0 -- per-element stiffness blocks
# --------------------------------------------------------------------------- #
def cst_membrane_Ke(p0, p1, p2, D: np.ndarray, thickness: float):
    """CST membrane element stiffness as a 9x9 block in global xyz DOFs.

    Node / DOF order: [p0(xyz), p1(xyz), p2(xyz)].  Mirrors
    kms.stiffness.membrane_stiffness_cst.  Returns (Ke_9x9 or None, area).
    """
    e1_raw = p1 - p0
    d20 = p2 - p0
    n1 = _norm3(e1_raw)
    if n1 < 1e-16:
        return None, 0.0
    e1 = e1_raw / n1
    normal = _cross(e1_raw, d20)
    area = 0.5 * _norm3(normal)
    if area < 1e-16:
        return None, area
    nhat = normal / (2.0 * area)
    e2 = _cross(nhat, e1)

    x1 = e1_raw @ e1
    x2 = d20 @ e1
    y2 = d20 @ e2

    det_J = x1 * y2
    if abs(det_J) < 1e-16:
        return None, area

    # Standard CST strain-displacement matrix (rows: eps_xx, eps_yy, gamma_xy).
    # WARNING: kms/stiffness.py has sign errors here -- its row 0 is negated and
    # the two y2 entries of row 2 are flipped.  Those errors (a) silently negate
    # the Poisson coupling and (b) break rigid-rotation invariance, so its CST
    # element fails the patch test and does not annihilate in-plane rotation.
    # We use the correct form; see prototype/README.md ("kms.stiffness bug").
    B = (1.0 / det_J) * np.array(
        [
            [-y2, 0.0, y2, 0.0, 0.0, 0.0],
            [0.0, x2 - x1, 0.0, -x2, 0.0, x1],
            [x2 - x1, -y2, -x2, y2, x1, 0.0],
        ]
    )
    Ke_local = (area * thickness) * (B.T @ D @ B)  # 6x6 (local 2D dof per node)

    R = np.empty((3, 2))  # 3x2 local->global  (= [e1 | e2])
    R[:, 0] = e1
    R[:, 1] = e2
    T = np.zeros((9, 6))
    T[0:3, 0:2] = R
    T[3:6, 2:4] = R
    T[6:9, 4:6] = R
    return T @ Ke_local @ T.T, area


def _dihedral_angle(p0, p1, p2, p3) -> float:
    """Angle between the two triangles sharing edge (p0, p1), wings p2 and p3.

    Any rotation-invariant scalar works as the bending residual; its gradient is
    automatically rigid-invariant.  (We do NOT use kms.stiffness's analytic
    gradient: it badly violates translation invariance -- see prototype/README.)
    """
    e = p1 - p0
    nA = np.cross(p2 - p0, e)
    nB = np.cross(e, p3 - p0)
    eh = e / np.linalg.norm(e)
    return float(np.arctan2(np.cross(nA, nB) @ eh, nA @ nB))


def dihedral_grad(p0, p1, p2, p3):
    """grad(theta) wrt [p0, p1 (shared edge), p2, p3 (wings)] as a 12-vector.

    Analytic standard discrete-shells gradient.  Rigid-invariant by construction
    (Sum gi = 0 and Sum Xi x gi = 0, both to ~1e-16) and matches the
    finite-difference gradient of `_dihedral_angle` up to an overall sign -- which
    is irrelevant because the stiffness uses g g^T.  Returns None if degenerate.
    """
    e = p1 - p0
    el = _norm3(e)
    if el < 1e-16:
        return None, 0.0, 0.0
    nL = _cross(p1 - p0, p2 - p0)
    nR = _cross(p3 - p0, p1 - p0)
    AL = 0.5 * _norm3(nL)
    AR = 0.5 * _norm3(nR)
    if AL < 1e-16 or AR < 1e-16:
        return None, AL, AR
    eh = e / el
    g2 = (nL / (2.0 * AL)) / (2.0 * AL / el)   # n_hat_L / h_L
    g3 = (nR / (2.0 * AR)) / (2.0 * AR / el)   # n_hat_R / h_R
    t2 = ((p2 - p0) @ eh) / el                 # altitude-foot fraction toward p1
    t3 = ((p3 - p0) @ eh) / el
    g0 = -(1.0 - t2) * g2 - (1.0 - t3) * g3
    g1 = -t2 * g2 - t3 * g3
    return np.concatenate([g0, g1, g2, g3]), AL, AR


def hinge_bend_Ke(p0, p1, p2, p3, kb: float):
    """Discrete dihedral-hinge bending stiffness, 12x12 rank-1: c * g g^T.

    Node / DOF order: [v0, v1 (shared edge), v2 (opp tri A), v3 (opp tri B)].
    Uses the analytic rigid-invariant dihedral gradient (see dihedral_grad).
    """
    g, AL, AR = dihedral_grad(p0, p1, p2, p3)
    if g is None:
        return None
    e_len = _norm3(p1 - p0)
    coeff = kb * e_len**2 / (AL + AR)
    return coeff * np.outer(g, g)


# --------------------------------------------------------------------------- #
#  Phase 1 -- affine probe subspace
# --------------------------------------------------------------------------- #
def affine_P(X: np.ndarray) -> np.ndarray:
    """P(X) in R^{3x12} with d = F X + c = P(X) a,  a = (vec_col(F), c).

    vec_col(F) is column-stacked, so a[0:9] = F.flatten(order='F').
    Built directly (no np.kron) -- this is on the decimation hot path.
    """
    P = np.zeros((3, 12))
    P[0, 0] = P[1, 1] = P[2, 2] = X[0]
    P[0, 3] = P[1, 4] = P[2, 5] = X[1]
    P[0, 6] = P[1, 7] = P[2, 8] = X[2]
    P[0, 9] = P[1, 10] = P[2, 11] = 1.0
    return P


# ---- curvature-enriched probe:  u(X) = 1/2 X^T H X + F X + c ---------------- #
#  Affine fields have zero second derivative -> they never bend a flat patch, so
#  the hinge term is ~null under the affine probe.  Adding quadratic monomials of
#  X gives fields with non-zero curvature, which DO excite bending.  The enriched
#  parameter is a = (quad(18), vec F(9), c(3)) in R^30; P(X) still factors through
#  the node's rest position, so the per-vertex additive reduction survives.
def quad_monomials(X) -> np.ndarray:
    """The 6 second-order monomials of X (sqrt2 on cross terms for even scaling)."""
    x, y, z = X[0], X[1], X[2]
    s = 1.4142135623730951
    return np.array([x * x, y * y, z * z, s * x * y, s * x * z, s * y * z])


def probe_dim(order: str = "affine") -> int:
    return 12 if order == "affine" else 30


def probe_P(X: np.ndarray, order: str = "affine") -> np.ndarray:
    """Probe sampler P(X): 3x12 (affine) or 3x30 (curvature = [quad | F | c])."""
    aff = affine_P(X)
    if order == "affine":
        return aff
    m = quad_monomials(X)
    Q = np.zeros((3, 18))
    Q[0, 0:6] = m
    Q[1, 6:12] = m
    Q[2, 12:18] = m
    return np.hstack([Q, aff])


def strain_mode_basis(order: str = "affine") -> np.ndarray:
    """The 6 constant-strain affine modes as columns of A in R^{dim x 6}.

    F symmetric (Frobenius-orthonormal), c = 0.  The energetic part of the affine
    block; translation + infinitesimal rotation are rigid and excluded.  For the
    curvature probe these live in the trailing 12 (affine) columns.
    """
    off = probe_dim(order) - 12  # affine block offset
    mats = []
    for i in range(3):  # 3 normal strains
        M = np.zeros((3, 3))
        M[i, i] = 1.0
        mats.append(M)
    for i, j in [(0, 1), (0, 2), (1, 2)]:  # 3 shear strains
        M = np.zeros((3, 3))
        M[i, j] = M[j, i] = 1.0 / np.sqrt(2.0)
        mats.append(M)
    A = np.zeros((probe_dim(order), 6))
    for k, M in enumerate(mats):
        A[off:off + 9, k] = M.flatten(order="F")  # c = 0
    return A


def curvature_mode_basis() -> np.ndarray:
    """The 18 constant-curvature (quadratic) modes as columns of A in R^{30x18}.
    Only meaningful for the curvature probe; these are what excite bending."""
    A = np.zeros((30, 18))
    A[0:18, 0:18] = np.eye(18)
    return A


def energy_mode_basis(order: str = "affine") -> np.ndarray:
    """All non-rigid probe modes (the deformations that store energy): the 6
    constant-strain modes, plus the 18 curvature modes for the curvature probe."""
    if order == "affine":
        return strain_mode_basis(order)
    return np.hstack([curvature_mode_basis(), strain_mode_basis(order)])


def rigid_mode_basis(order: str = "affine") -> np.ndarray:
    """The 6 rigid modes (3 translations + 3 infinitesimal rotations) as columns
    in R^{dim x 6}.  A correct element carries ~zero energy on these."""
    off = probe_dim(order) - 12
    A = np.zeros((probe_dim(order), 6))
    for k in range(3):  # translations: F = 0, c = e_k
        A[off + 9 + k, k] = 1.0
    for k, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):  # rotations: F skew
        M = np.zeros((3, 3))
        M[i, j] = 1.0 / np.sqrt(2.0)
        M[j, i] = -1.0 / np.sqrt(2.0)
        A[off:off + 9, 3 + k] = M.flatten(order="F")
    return A


# --------------------------------------------------------------------------- #
#  Phase 2 -- project elements into probe coords
# --------------------------------------------------------------------------- #
def _stack_P(nodes, order: str = "affine") -> np.ndarray:
    """Stack P(X_n) for the given nodes into a (3k x dim) probe sampler.
    The affine path is built directly (no np.kron / vstack) -- it is on the
    decimation hot path; the curvature path uses probe_P (opt-in, less hot).
    """
    if order != "affine":
        return np.vstack([probe_P(X, order) for X in nodes])
    V = np.zeros((3 * len(nodes), 12))
    for r, X in enumerate(nodes):
        b = 3 * r
        V[b, 0] = V[b + 1, 1] = V[b + 2, 2] = X[0]
        V[b, 3] = V[b + 1, 4] = V[b + 2, 5] = X[1]
        V[b, 6] = V[b + 1, 7] = V[b + 2, 8] = X[2]
        V[b, 9] = V[b + 1, 10] = V[b + 2, 11] = 1.0
    return V


def _V_tri(X0, X1, X2, order: str = "affine") -> np.ndarray:
    return _stack_P((X0, X1, X2), order)


def _V_hinge(X0, X1, X2, X3, order: str = "affine") -> np.ndarray:
    return _stack_P((X0, X1, X2, X3), order)


def probe_energy(G: np.ndarray, A_strain: np.ndarray) -> float:
    """Homogenized response magnitude  0.5 * sum_k a_k^T G a_k = 0.5 tr(A^T G A).

    Linear in G, so per-vertex totals decompose additively over elements.
    """
    return 0.5 * float(np.trace(A_strain.T @ G @ A_strain))


def find_hinges(mesh: TriMesh) -> np.ndarray:
    """Interior-edge hinges as rows [v0, v1, v2, v3] (shared edge v0-v1).

    Skips boundary edges (1 face -> no bending partner) and non-manifold edges
    (>2 faces -> ambiguous dihedral; needs an explicit convention, see doc).
    """
    edge_to_faces: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for fi, tri in enumerate(mesh.faces):
        for opp in range(3):
            a = int(tri[(opp + 1) % 3])
            b = int(tri[(opp + 2) % 3])
            edge_to_faces[(min(a, b), max(a, b))].append((fi, opp))
    hinges = []
    for (ea, eb), fl in edge_to_faces.items():
        if len(fl) != 2:
            continue
        (fa, oa), (fb, ob) = fl
        hinges.append([ea, eb, int(mesh.faces[fa][oa]), int(mesh.faces[fb][ob])])
    return np.array(hinges, dtype=np.int64) if hinges else np.zeros((0, 4), np.int64)


@dataclass
class MechModel:
    """All per-element / per-vertex mechanical-QEM data for a fixed mesh."""

    mesh: TriMesh
    D: np.ndarray
    kb: float
    thickness: float
    A_strain: np.ndarray          # 6 constant-strain modes (dim x 6)
    A_energy: np.ndarray          # all non-rigid modes (strain [+ curvature])
    probe: str                    # "affine" or "curvature"
    face_Ke: list                 # 9x9 membrane block (or None) per face
    face_area: np.ndarray         # (F,)
    face_G: list                  # (dim x dim) membrane probe quadric per face
    hinges: np.ndarray            # (H, 4) int
    hinge_G: list                 # (dim x dim) bending probe quadric per hinge
    vert_faces: list[set]         # incident triangles
    vert_hinges: list[set]        # incident hinges (4-vertex stencil membership)


def build_model(
    mesh: TriMesh, E: float = 1.0, nu: float = 0.3, thickness: float = 1e-3,
    probe: str = "affine",
) -> MechModel:
    """Run Phases 0-2: per-element K_e, probe projection G_e, vertex adjacency.

    probe="affine" (default, 12-D) preserves homogenized membrane stiffness;
    probe="curvature" (30-D) adds constant-curvature modes so bending registers.
    """
    D = plane_stress_D(E, nu)
    kb = bending_coeff(E, nu, thickness)
    A_strain = strain_mode_basis(probe)
    A_energy = energy_mode_basis(probe)
    dim = probe_dim(probe)
    v, f = mesh.vertices, mesh.faces

    face_Ke: list = []
    face_G: list = []
    face_area = np.zeros(mesh.n_faces)
    for fi, tri in enumerate(f):
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        Ke, area = cst_membrane_Ke(v[i0], v[i1], v[i2], D, thickness)
        face_area[fi] = area
        if Ke is None:
            face_Ke.append(None)
            face_G.append(np.zeros((dim, dim)))
            continue
        V = _V_tri(v[i0], v[i1], v[i2], probe)
        face_Ke.append(Ke)
        face_G.append(V.T @ Ke @ V)

    hinges = find_hinges(mesh)
    hinge_G: list = []
    for h in hinges:
        a, b, c, d = int(h[0]), int(h[1]), int(h[2]), int(h[3])
        Ke = hinge_bend_Ke(v[a], v[b], v[c], v[d], kb)
        if Ke is None:
            hinge_G.append(np.zeros((dim, dim)))
            continue
        V = _V_hinge(v[a], v[b], v[c], v[d], probe)
        hinge_G.append(V.T @ Ke @ V)

    vert_faces: list[set] = [set() for _ in range(mesh.n_verts)]
    for fi, tri in enumerate(f):
        for vi in tri:
            vert_faces[int(vi)].add(fi)
    vert_hinges: list[set] = [set() for _ in range(mesh.n_verts)]
    for hi, h in enumerate(hinges):
        for vi in h:
            vert_hinges[int(vi)].add(hi)

    return MechModel(
        mesh, D, kb, thickness, A_strain, A_energy, probe,
        face_Ke, face_area, face_G, hinges, hinge_G, vert_faces, vert_hinges,
    )


# --------------------------------------------------------------------------- #
#  Layer A -- response-template magnitude fields
# --------------------------------------------------------------------------- #
def per_triangle_membrane_cost(model: MechModel) -> np.ndarray:
    """Probe energy of each face's membrane quadric G_e.  ~ proportional to
    area * t * tr(D) for homogeneous material (doc sanity check 1)."""
    return np.array([probe_energy(G, model.A_strain) for G in model.face_G])


def per_triangle_Ke_norm(model: MechModel) -> np.ndarray:
    """||K_e||_F per face --- a conditioning / sliver indicator (blows up as
    area -> 0).  Motivates the E_triq sliver term in the composite cost."""
    out = np.zeros(model.mesh.n_faces)
    for fi, Ke in enumerate(model.face_Ke):
        out[fi] = 0.0 if Ke is None else float(np.linalg.norm(Ke))
    return out


def per_vertex_costs(model: MechModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate G_v = sum_{e ni v} G_e and return probe energies
    (membrane, bending, total).  Membrane energy uses the strain modes; bending
    uses all non-rigid modes (so the curvature probe lets bending register)."""
    n = model.mesh.n_verts
    dim = probe_dim(model.probe)
    Z = np.zeros((dim, dim))
    mem = np.zeros(n)
    ben = np.zeros(n)
    for vi in range(n):
        Gm = sum((model.face_G[fi] for fi in model.vert_faces[vi]), Z)
        Gb = sum((model.hinge_G[hi] for hi in model.vert_hinges[vi]), Z)
        mem[vi] = probe_energy(Gm, model.A_strain)
        ben[vi] = probe_energy(Gb, model.A_energy)
    return mem, ben, mem + ben


def global_G(model: MechModel) -> np.ndarray:
    """Global homogenized stiffness G = sum_e G_e (dim x dim, PSD, eff. rank 6)."""
    dim = probe_dim(model.probe)
    G = np.zeros((dim, dim))
    for Gf in model.face_G:
        G += Gf
    for Gh in model.hinge_G:
        G += Gh
    return G


# --------------------------------------------------------------------------- #
#  Layer B -- the actual collapse cost  ||G_after - G_fine||_W^2  (membrane)
# --------------------------------------------------------------------------- #
def _edge_faces(mesh: TriMesh) -> dict[tuple[int, int], set]:
    ef: dict[tuple[int, int], set] = defaultdict(set)
    for fi, tri in enumerate(mesh.faces):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, w in [(a, b), (b, c), (a, c)]:
            ef[(min(u, w), max(u, w))].add(fi)
    return ef


def membrane_collapse_costs(
    model: MechModel, candidates: tuple[str, ...] = ("mid", "u", "v")
):
    """Layer B: per-edge collapse cost (membrane).

    For candidate merged position x*, the only elements whose G_e change are the
    triangles incident to u or w; everything else cancels in (G_after - G_fine).
    The two collapse triangles vanish (G_new = 0); the surviving incident
    triangles are re-evaluated with the merged vertex at x*.  Cost is the
    minimum over the candidate placements of  ||G_after - G_fine||_F^2 .

    Returns (edges, costs, best_positions).  Bending is intentionally NOT
    re-evaluated here (open fork: dihedral references move across the 1-ring).
    """
    mesh = model.mesh
    v, f, D, t = mesh.vertices, mesh.faces, model.D, model.thickness
    ef = _edge_faces(mesh)
    edges = sorted(ef.keys())
    costs = np.zeros(len(edges))
    best = np.zeros((len(edges), 3))
    Z = np.zeros((12, 12))

    for ei, (u, w) in enumerate(edges):
        shared = ef[(u, w)]
        incident = model.vert_faces[u] | model.vert_faces[w]
        survivors = [fi for fi in incident if fi not in shared]
        G_old = sum((model.face_G[fi] for fi in incident), Z)  # affected old block

        cand = []
        if "mid" in candidates:
            cand.append(0.5 * (v[u] + v[w]))
        if "u" in candidates:
            cand.append(v[u].copy())
        if "v" in candidates:
            cand.append(v[w].copy())

        best_c, best_p = np.inf, cand[0]
        for x in cand:
            G_new = Z.copy()
            for fi in survivors:
                tri = f[fi]
                pts = [x if (int(idx) == u or int(idx) == w) else v[int(idx)] for idx in tri]
                Ke, _ = cst_membrane_Ke(pts[0], pts[1], pts[2], D, t)
                if Ke is None:
                    continue
                V = _V_tri(pts[0], pts[1], pts[2])
                G_new = G_new + V.T @ Ke @ V
            dG = G_new - G_old
            c = float(np.sum(dG * dG))
            if c < best_c:
                best_c, best_p = c, x
        costs[ei] = best_c
        best[ei] = best_p
    return edges, costs, best


def edges_to_vertex_min(edges, costs, n_verts: int) -> np.ndarray:
    """Map per-edge costs to per-vertex via the min of incident edges (matches
    the repo convention): 'cheapest collapse available at this vertex'."""
    vmin = np.full(n_verts, np.inf)
    for (u, w), c in zip(edges, costs):
        if c < vmin[u]:
            vmin[u] = c
        if c < vmin[w]:
            vmin[w] = c
    finite = np.isfinite(vmin)
    if np.any(finite):
        vmin[~finite] = np.max(vmin[finite])
    else:
        vmin[:] = 0.0
    return vmin


# --------------------------------------------------------------------------- #
#  Cross-validation helper (used by the smoke test)
# --------------------------------------------------------------------------- #
def assemble_global_membrane(model: MechModel) -> np.ndarray:
    """Scatter per-element membrane blocks into a dense global matrix, so we can
    check equality against kms.stiffness.membrane_stiffness_cst."""
    n = model.mesh.n_verts
    K = np.zeros((3 * n, 3 * n))
    for fi, tri in enumerate(model.mesh.faces):
        Ke = model.face_Ke[fi]
        if Ke is None:
            continue
        dofs = []
        for idx in tri:
            idx = int(idx)
            dofs += [3 * idx, 3 * idx + 1, 3 * idx + 2]
        K[np.ix_(dofs, dofs)] += Ke
    return K
