"""Visualize the mechanical-QEM costs of docs/mech_qem.tex with Polyscope.

Shows, on one mesh (toggle quantities in the UI):

  PER-TRIANGLE (face quantities)
    * area                       -- reference
    * membrane cost              -- probe energy of the face quadric G_e (Layer A)
    * |Ke|_F                     -- sliver / conditioning indicator

  PER-VERTEX (vertex quantities)
    * membrane cost              -- probe energy of accumulated membrane G_v (Layer A)
    * bending cost               -- probe energy of accumulated bending G_v (Layer A)
    * total cost                 -- membrane + bending (Layer A)
    * min collapse cost          -- ||G_after - G_fine||^2, min incident edge (Layer B)
    * min collapse cost (log10)  -- same, log-scaled for the heavy tail

Layer A = "how much homogenized response lives here" (a response template, NOT an
error -- see doc divergence 2).  Layer B = the actual edge-collapse decimation cost.

Run:
  python prototype/viz_costs.py                              # dress mesh, interactive
  python prototype/viz_costs.py --mesh data/spot.obj         # custom mesh
  python prototype/viz_costs.py --smoke                      # headless verification
  python prototype/viz_costs.py --thickness 0.02             # thicker shell (more bending)
  python prototype/viz_costs.py --no-collapse-cost           # skip Layer B (faster)
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech_qem as mq  # noqa: E402

from kms.mesh import load_obj  # noqa: E402
from kms import colors


DEFAULT_MESH = "data/9423122485_cleaned.obj"


def run(mesh, thickness: float, do_collapse: bool):
    print(f"  Mesh: {mesh.n_verts} verts, {mesh.n_faces} faces")
    t0 = time.time()
    model = mq.build_model(mesh, thickness=thickness)
    print(f"  Model built (Phases 0-2): {time.time() - t0:.2f}s")
    print(f"  Hinges (interior edges): {len(model.hinges)}  "
          f"(thickness={thickness}, bending coeff={model.kb:.3e})")

    # ---- Layer A ----
    tri_membrane = mq.per_triangle_membrane_cost(model)
    tri_ke_norm = mq.per_triangle_Ke_norm(model)
    vtx_membrane, vtx_bending, vtx_total = mq.per_vertex_costs(model)

    print(f"\n  [Layer A] per-triangle membrane cost: "
          f"min={tri_membrane.min():.3e} max={tri_membrane.max():.3e}")
    print(f"  [Layer A] per-vertex   membrane cost: "
          f"min={vtx_membrane.min():.3e} max={vtx_membrane.max():.3e}")
    print(f"  [Layer A] per-vertex   bending  cost: "
          f"min={vtx_bending.min():.3e} max={vtx_bending.max():.3e}")

    # ---- Layer B ----
    vtx_collapse = None
    edges = edge_costs = None
    if do_collapse:
        t0 = time.time()
        edges, edge_costs, _ = mq.membrane_collapse_costs(model)
        edges = np.asarray(edges, dtype=np.int64)  # (E, 2) for a curve network
        vtx_collapse = mq.edges_to_vertex_min(edges, edge_costs, mesh.n_verts)
        print(f"\n  [Layer B] {len(edges)} edge collapse costs: {time.time() - t0:.2f}s")
        print(f"  [Layer B] per-edge   collapse cost: "
              f"min={edge_costs.min():.3e} max={edge_costs.max():.3e}")
        print(f"  [Layer B] per-vertex min collapse cost: "
              f"min={vtx_collapse.min():.3e} max={vtx_collapse.max():.3e}")

    return {
        "model": model,
        "tri_area": model.face_area,
        "tri_membrane": tri_membrane,
        "tri_ke_norm": tri_ke_norm,
        "vtx_membrane": vtx_membrane,
        "vtx_bending": vtx_bending,
        "vtx_total": vtx_total,
        "vtx_collapse": vtx_collapse,
        "edges": edges,
        "edge_costs": edge_costs,
    }


def smoke(mesh, q):
    print("\n--- Smoke checks ---")
    model = q["model"]
    v = mesh.vertices

    # 1. CST element correctness (bug-independent): patch test + rigid annihilation.
    #    Apply a uniform strain and check the element stores exactly
    #    0.5 * A * t * eps^T D eps; and check all 6 rigid modes carry zero energy.
    A_rig = mq.rigid_mode_basis()
    eps = np.array([0.013, -0.007, 0.004])  # exx, eyy, gxy
    worst_patch, worst_rigid = 0.0, 0.0
    for fi in range(0, mesh.n_faces, 37):  # sample a spread of faces
        tri = mesh.faces[fi]
        p = [v[int(tri[0])], v[int(tri[1])], v[int(tri[2])]]
        Ke, area = mq.cst_membrane_Ke(p[0], p[1], p[2], model.D, model.thickness)
        if Ke is None:
            continue
        # local frame -> build the uniform-strain displacement field
        e1 = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        nh = nrm / np.linalg.norm(nrm)
        e2 = np.cross(nh, e1)
        R = np.column_stack([e1, e2])
        d = []
        for pi in p:
            xl = np.array([(pi - p[0]) @ e1, (pi - p[0]) @ e2])
            ul = np.array([eps[0] * xl[0] + 0.5 * eps[2] * xl[1],
                           0.5 * eps[2] * xl[0] + eps[1] * xl[1]])
            d.append(R @ ul)
        d = np.concatenate(d)
        energy = 0.5 * d @ Ke @ d
        exact = 0.5 * area * model.thickness * (eps @ model.D @ eps)
        worst_patch = max(worst_patch, abs(energy - exact) / exact)
        # rigid modes (9-dof displacement = affine mode sampled at the 3 nodes)
        for k in range(6):
            dr = np.concatenate([mq.affine_P(pi) @ A_rig[:, k] for pi in p])
            worst_rigid = max(worst_rigid, float(np.linalg.norm(Ke @ dr)))
    print(f"  CST patch test (uniform strain): worst rel err = {worst_patch:.2e}")
    print(f"  CST annihilates 6 rigid modes  : worst ||Ke d|| = {worst_rigid:.2e}")
    assert worst_patch < 1e-10, "CST element fails the patch test"
    assert worst_rigid < 1e-10, "CST element does not annihilate rigid modes"

    # 1b. bending hinge: g = grad(dihedral) must be rigid-invariant (FD precision),
    #     else the assembled operator loses its rank-6 affine null space.
    worst_h = 0.0
    for h in model.hinges[::53]:
        p = [v[int(h[k])] for k in range(4)]
        Ke = mq.hinge_bend_Ke(p[0], p[1], p[2], p[3], model.kb)
        if Ke is None:
            continue
        nrm = np.linalg.norm(Ke)
        for k in range(6):
            dr = np.concatenate([mq.affine_P(p[i]) @ A_rig[:, k] for i in range(4)])
            denom = nrm * np.linalg.norm(dr) + 1e-30
            worst_h = max(worst_h, float(np.linalg.norm(Ke @ dr)) / denom)
    print(f"  bending hinge annihilates rigid modes (rel): worst = {worst_h:.2e}")
    assert worst_h < 1e-5, "bending hinge not rigid-invariant"

    # 2. each face quadric is symmetric PSD
    worst = 0.0
    for G in model.face_G:
        worst = min(worst, float(np.linalg.eigvalsh(0.5 * (G + G.T))[0]))
        assert np.allclose(G, G.T, atol=1e-10)
    print(f"  all face quadrics symmetric PSD          (min eig={worst:.2e})")

    # 3. global G: PSD, effective rank 6 (null space = 6 rigid affine modes)
    G = mq.global_G(model)
    eig = np.linalg.eigvalsh(0.5 * (G + G.T))
    emax = eig[-1]
    nullity = int(np.sum(eig < 1e-7 * emax))
    print(f"  global G eigenvalues: {np.array2string(eig, precision=3)}")
    print(f"  global G nullity (rigid modes) = {nullity}  (expect 6)")
    assert eig[0] >= -1e-7 * emax, "global G not PSD"
    assert nullity == 6, f"expected effective rank 6 (nullity 6), got nullity {nullity}"
    rig_e = np.array([mq.probe_energy(G, A_rig[:, [k]]) for k in range(6)])
    str_e = np.array([mq.probe_energy(G, model.A_strain[:, [k]]) for k in range(6)])
    print(f"  rigid-mode energies (~0): max={np.abs(rig_e).max():.2e}")
    print(f"  strain-mode energies (>0): min={str_e.min():.3e} max={str_e.max():.3e}")
    assert np.abs(rig_e).max() < 1e-7 * emax, "rigid modes carry energy"
    assert str_e.min() > 0, "a strain mode carries no energy"

    # 4. doc sanity check 1: per-triangle membrane cost ~ proportional to area
    from scipy.stats import spearmanr, pearsonr
    rho, _ = spearmanr(q["tri_membrane"], q["tri_area"])
    r, _ = pearsonr(q["tri_membrane"], q["tri_area"])
    ratio = q["tri_membrane"] / np.maximum(q["tri_area"], 1e-30)
    print(f"  per-tri membrane cost vs area: spearman={rho:.4f} pearson={r:.4f}")
    print(f"  cost/area ratio: std/mean={ratio.std() / ratio.mean():.2e} (should be ~0: cost == c*area)")
    assert rho > 0.99, f"membrane cost not ~area (spearman {rho:.3f})"
    assert ratio.std() / ratio.mean() < 1e-6, "cost/area not constant (doc check 1)"

    # 5. collapse costs non-negative and finite
    if q["vtx_collapse"] is not None:
        vc = q["vtx_collapse"]
        assert np.all(np.isfinite(vc)) and np.all(vc >= -1e-12), "bad collapse costs"
        print(f"  Layer B collapse costs finite & >= 0  (max={vc.max():.3e})")

    # 6. DIAGNOSTIC (non-asserting): expose the kms.stiffness bugs this prototype fixes.
    from kms.stiffness import membrane_stiffness_cst
    Km_mine = mq.assemble_global_membrane(model)
    Km_kms = membrane_stiffness_cst(mesh, thickness=model.thickness).toarray()
    dm = np.abs(Km_mine - Km_kms).max()
    print(f"\n  [diagnostic] max|K_membrane(fixed) - K_membrane(kms)| = {dm:.3e}")
    print("  [diagnostic] kms.stiffness has TWO bugs this prototype fixes:")
    print("    (1) CST B-matrix sign error -> fails patch test above (membrane diff != 0)")
    print("    (2) hinge gradient violates translation invariance (||sum g_i||/||g|| ~ 1)")
    print("  See prototype/README.md ('kms.stiffness bugs').")

    print("\nSMOKE: ALL PASS")


def visualize(mesh, q):
    import polyscope as ps

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_front_dir("neg_z_front")

    m = ps.register_surface_mesh("dress", mesh.vertices, mesh.faces, edge_width=0.4, color=colors.RENDER_COLORS["gray"])

    # default view: the most informative field that is present
    default_name = "vtx min collapse cost" if q["vtx_collapse"] is not None else "vtx total cost"

    def add(name, vals, where, cmap):
        # polyscope 2.6 add_scalar_quantity returns None; use the enabled= kwarg
        m.add_scalar_quantity(name, vals, defined_on=where, cmap=cmap,
                              vminmax=(float(vals.min()), float(vals.max())),
                            #   enabled=(name == default_name))
                            enabled=False)

    # per-triangle (face) quantities
    add("tri area", q["tri_area"], "faces", "viridis")
    add("tri membrane cost", q["tri_membrane"], "faces", "viridis")
    add("tri |Ke|_F (sliver)", q["tri_ke_norm"], "faces", "inferno")

    # per-vertex quantities
    add("vtx membrane cost", q["vtx_membrane"], "vertices", "coolwarm")
    add("vtx bending cost", q["vtx_bending"], "vertices", "coolwarm")
    # bending is heavy-tailed + ~1e5x smaller than membrane; log10 makes it legible
    add("vtx bending cost (log10)", np.log10(q["vtx_bending"] + 1e-30), "vertices", "coolwarm")
    add("vtx total cost", q["vtx_total"], "vertices", "coolwarm")
    if q["vtx_collapse"] is not None:
        vc = q["vtx_collapse"]
        add("vtx min collapse cost", vc, "vertices", "coolwarm")
        add("vtx min collapse cost (log10)", np.log10(vc + 1e-30), "vertices", "coolwarm")

    # per-EDGE collapse cost (the *native* Layer B quantity) as a curve-network overlay.
    # Hide the "dress" surface in the UI to see the colored edge tubes clearly.
    if q.get("edges") is not None:
        ec = q["edge_costs"]
        cn = ps.register_curve_network("collapse edges", mesh.vertices, q["edges"])
        cn.set_radius(0.0015, relative=True)
        cn.add_scalar_quantity("collapse cost", ec, defined_on="edges", cmap="coolwarm")
        cn.add_scalar_quantity("collapse cost (log10)", np.log10(ec + 1e-30),
                               defined_on="edges", cmap="coolwarm", enabled=True)

    print("\n  Polyscope open. Two structures in the left panel:")
    print(f"    'dress'         surface mesh -- showing '{default_name}'")
    print("    'collapse edges' curve network -- per-EDGE collapse cost (log10) on tubes")
    print("  Toggle quantities per structure; hide 'dress' to see edge costs clearly.")
    print("\n  Reading the fields:")
    print("    - tri/vtx membrane cost ~ area (Layer A == response template, doc check 1)")
    print("    - vtx bending cost      = curvature-coupling under the affine probe (use log10)")
    print("    - collapse cost (edges) = cost to collapse THAT edge (Layer B == real cost)")
    print("    - vtx min collapse cost = cheapest incident-edge collapse at that vertex")
    ps.show()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=DEFAULT_MESH, help="Path to .obj mesh")
    p.add_argument("--thickness", type=float, default=1e-3, help="Shell thickness")
    p.add_argument("--no-collapse-cost", action="store_true",
                   help="Skip Layer B edge-collapse cost (faster)")
    p.add_argument("--smoke", action="store_true", help="Headless verification")
    args = p.parse_args()

    print("=== Mechanical-QEM Cost Visualization (mech_qem.tex) ===")
    mesh = load_obj(args.mesh)
    q = run(mesh, args.thickness, do_collapse=not args.no_collapse_cost)

    if args.smoke:
        smoke(mesh, q)
    else:
        visualize(mesh, q)


if __name__ == "__main__":
    main()
