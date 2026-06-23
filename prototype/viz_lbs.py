"""Design B: drive the fine mesh from a deformation of the coarse proxy (the PBD
workflow), via a fine<-coarse skinning map  d_fine = M d_coarse.

Four ways to build M, selectable with --weights:
  local  -- accumulated DURING decimation: each collapse contributes a local
            harmonic condensation s_vj = -K_vv^{-1} K_vj of the eliminated vertex on
            its 1-ring, composed through the collapse sequence
            (simplify_mechanics(..., return_prolongation=True)).  Bounded support,
            but only an APPROXIMATION to the harmonic extension (no Schur fill-in).
  global -- the one-shot harmonic solve  S = -K_ee^{-1} K_er  on the fine K with the
            decimation's survivors as handles (harmonic_skinning.
            harmonic_prolongation_matrix).  The exact energy-minimizing extension
            -> smooth, but denser.  Mathematically the exact-Schur limit of `local`.
  edge   -- the FAITHFUL 2-POINT EDGE BLEND (stiffness-FREE): each eliminated vertex
            follows its survivor (the placement weight a is a fine->coarse averaging
            weight that cancels coarse->fine), so it composes to a PIECEWISE-CONSTANT
            cluster skin -- 1 handle/vertex, sparsest, strictly nonneg/PoU, blockiest.
            The prolongation cousin of the spectral restriction (Lescoat et al.).
  geom   -- STIFFNESS-FREE mean-value coordinates over each eliminated vertex's
            1-ring (Floater 2003), composed through the SAME machinery as `local`.
            Convex, affine-precise, no K -- the smooth geometric middle ground
            between `edge` and `local`.
  all    -- (default) all four side by side, reporting each map's elastic energy
            d^T K d (lower = smoother), so the roughness gap is explicit.

The optimized proxy positions are KEPT either way (the map carries displacements).

Views (rows): REST -- coarse proxy (+ painted handle, with its quadratic-placement
drift marked) and the fine mesh PAINTED by its LBS weight to that handle; DEFORMED
-- the coarse proxy under an elastic deformation and the fine mesh driven from it.

Run:
  python prototype/viz_lbs.py                       # compare local, global, edge
  python prototype/viz_lbs.py --weights edge
  python prototype/viz_lbs.py --handles 200 --paint-handle 12
  python prototype/viz_lbs.py --screenshot out/lbs.png
  python prototype/viz_lbs.py --smoke               # headless checks (all maps)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harmonic_skinning as hs  # noqa: E402
import mech_qem as mq  # noqa: E402
import simplify_mechanics as sm  # noqa: E402

from kms.mesh import load_obj, save_obj  # noqa: E402


DEFAULT_MESH = "data/9423122485_cleaned.obj"

# Decimation thickness matches viz_simplify/viz_skinning (greedy collapse is
# path-dependent).  The prolongation's local harmonic solves use a thicker shell
# (well-conditioned out-of-plane) -- both decoupled, as elsewhere.
DECIM_THICKNESS = 1e-3
PROLONG_THICKNESS = 0.05


def get_lbs(mesh, n_handles, stem, bending_weight=1.0, decim_thickness=DECIM_THICKNESS,
            prolong_thickness=PROLONG_THICKNESS, cache_dir="out"):
    """Decimate + emit ALL decimation-time maps in ONE pass (Design B), cached:
        "local" -- the harmonic 1-ring condensation accumulated per collapse;
        "edge"  -- the faithful 2-point edge blend (eliminated vertex follows its
                   survivor -> piecewise-constant), cousin of the spectral restriction;
        "geom"  -- stiffness-free mean-value coordinates over the 1-ring (Floater).
    The collapse sequence is identical for all, so survivors/coarse are shared.
    Returns (survivors, coarse_mesh, {"local":.., "edge":.., "geom":..}).

    The coarse proxy keeps its stiffness-optimal (drifted) rest positions; co-rotational
    LBS absorbs the rest-to-anchor drift, so there is no snap-to-anchor step."""
    probe = "curvature" if bending_weight > 0.0 else "affine"
    base = os.path.join(
        cache_dir,
        f"_lbs_{stem}_{n_handles}_{decim_thickness:g}_b{bending_weight:g}_{probe}"
        f"_pt{prolong_thickness:g}")
    npy, obj = base + ".npy", base + ".obj"
    ph_npz, pe_npz, pg_npz = base + "_Ph.npz", base + "_Pe.npz", base + "_Pg.npz"
    if all(os.path.exists(f) for f in (npy, obj, ph_npz, pe_npz, pg_npz)):
        print(f"  LBS proxy: loaded cached {obj}")
        return (np.load(npy), load_obj(obj),
                {"local": sparse.load_npz(ph_npz), "edge": sparse.load_npz(pe_npz),
                 "geom": sparse.load_npz(pg_npz)})
    t0 = time.time()
    coarse, survivors, Ps = sm.simplify_mechanics(
        mesh, target_verts=n_handles, thickness=decim_thickness, probe=probe,
        bending_weight=bending_weight, prolong_thickness=prolong_thickness,
        return_prolongation=True, prolong_mode="all")
    Ph, Pe, Pg = Ps["harmonic"], Ps["edge"], Ps["geometric"]
    print(f"  LBS proxy: decimated {mesh.n_verts} -> {len(survivors)} verts, "
          f"P_harmonic {Ph.shape} ({Ph.nnz} nnz), P_edge ({Pe.nnz} nnz), "
          f"P_geom ({Pg.nnz} nnz) in {time.time()-t0:.1f}s "
          f"(probe={probe}, bending_weight={bending_weight:g})")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(npy, survivors)
    save_obj(coarse, obj)
    sparse.save_npz(ph_npz, Ph)
    sparse.save_npz(pe_npz, Pe)
    sparse.save_npz(pg_npz, Pg)
    return survivors, coarse, {"local": Ph, "edge": Pe, "geom": Pg}


def coarse_elastic_field(coarse, thickness=PROLONG_THICKNESS, seed=3):
    """A soft-mode (elastic) displacement of the COARSE mesh -- the deformation the
    proxy would undergo, the same family viz_skinning visualizes but on the proxy."""
    model = mq.build_model(coarse, thickness=thickness)
    Kc = hs.assemble_K_sparse(model)
    return hs.elastic_fields(Kc, coarse.vertices, ntrials=1, seed=seed)[0]


def _k_energy(K, d):
    """Elastic energy d^T K d (lower = smoother; the global harmonic map minimizes it)."""
    f = d.ravel()
    return float(f @ (K @ f))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=DEFAULT_MESH)
    p.add_argument("--handles", type=int, default=128)
    p.add_argument("--bending-weight", type=float, default=1.0,
                   help="bending weight for the mechanics decimation (>0 -> curvature probe)")
    p.add_argument("--weights",
                   choices=["local", "global", "edge", "geom", "both", "all"],
                   default="all",
                   help="fine<-coarse map(s) to show: 'local' (harmonic accumulated), "
                        "'global' (one-shot harmonic solve), 'edge' (2-point edge "
                        "blend), 'geom' (mean-value coords), 'both' (local+global), "
                        "or 'all' (default: all four)")
    p.add_argument("--paint-handle", type=int, default=-1,
                   help="coarse-handle index whose LBS weights are painted on the fine "
                        "mesh (default: a broad, camera-facing handle)")
    p.add_argument("--scale", type=float, default=1.0, help="deformation exaggeration")
    p.add_argument("--seed", type=int, default=3, help="coarse elastic-field seed")
    p.add_argument("--screenshot", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    print("=== LBS weights: step-by-step accumulation vs global solve (Design B) ===")
    mesh = load_obj(args.mesh)
    verts = mesh.vertices
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    survivors, coarse, P_decim = get_lbs(mesh, args.handles, Path(args.mesh).stem,
                                         bending_weight=args.bending_weight)
    n_fine, n_coarse = mesh.n_verts, len(survivors)

    # four fine<-coarse maps, SAME survivor/handle indexing so interchangeable:
    #   local  = harmonic 1-ring condensation accumulated during decimation
    #   global = one-shot harmonic solve on the fine K (smooth, denser)
    #   edge   = faithful 2-point edge blend (piecewise-constant, stiffness-free)
    #   geom   = mean-value coordinates over the 1-ring (smooth, stiffness-free)
    K = hs.assemble_K_sparse(mq.build_model(mesh, thickness=PROLONG_THICKNESS))
    # small reg: thickness 0.05 keeps K_ee well-conditioned, so a tiny regularizer
    # suffices and keeps translation reproduction crisp (reg leaks into PoU).
    S = hs.harmonic_prolongation_matrix(K, survivors, n_fine, reg=1e-10)
    maps = {"local": P_decim["local"], "global": S, "edge": P_decim["edge"],
            "geom": P_decim["geom"]}
    W = {k: sm.prolongation_scalar_weights(M, n_fine, n_coarse) for k, M in maps.items()}

    # which handle to paint: broad support on the camera-facing (+z) side so the
    # weight blob is visible; chosen from the (smoother) global weights.
    ph = args.paint_handle
    if ph < 0:
        front = verts[:, 2] > np.median(verts[:, 2])
        ph = int(np.argmax((W["global"][front] > 0.05).sum(axis=0)))
    ph = max(0, min(ph, n_coarse - 1))

    # deform the coarse proxy, propagate to the fine mesh through each map
    d_coarse = coarse_elastic_field(coarse, seed=args.seed)
    d_fine = {k: (M @ d_coarse.ravel()).reshape(n_fine, 3) for k, M in maps.items()}

    # ---- report ----
    print(f"\n  fine {n_fine} verts -> coarse {n_coarse} handles; painting handle {ph} "
          f"(fine vertex {int(survivors[ph])})")
    print(f"  {'map':<8s} {'nnz':>9s} {'density':>8s} {'eff.handles':>12s} "
          f"{'PoU err':>9s} {'elastic energy':>15s}")
    energies = {}
    for k, M in maps.items():
        Wk = W[k]
        energies[k] = _k_energy(K, d_fine[k])
        eff = float(np.mean(1.0 / np.sum(Wk**2, axis=1).clip(1e-30)))
        pou = float(np.abs(Wk.sum(axis=1) - 1.0).max())
        dens = M.nnz / (M.shape[0] * M.shape[1])
        print(f"  {k:<8s} {M.nnz:>9d} {dens:>7.1%} {eff:>12.1f} {pou:>9.1e} {energies[k]:>15.4e}")
    base_e = energies["global"]
    for k in maps:
        if k != "global":
            print(f"  -> {k:<6s} is {energies[k]/base_e:5.1f}x rougher than global "
                  f"(higher elastic energy = less smooth)")

    if args.smoke:
        print("\n--- Smoke checks ---")
        coarse_of = {int(o): k for k, o in enumerate(survivors)}
        t = np.array([0.3, -0.7, 1.1])
        for k, M in maps.items():
            assert M.shape == (3 * n_fine, 3 * n_coarse), (k, M.shape)
            # PoU/translation are exact up to the harmonic solve's reg (1e-8): the
            # local map hits ~1e-7, the global solve is reg-limited at ~1e-5.
            pou = float(np.abs(W[k].sum(axis=1) - 1.0).max())
            assert pou < 1e-4, f"{k}: partition of unity broken: {pou:.2e}"
            trans = float(np.abs((M @ np.tile(t, n_coarse)).reshape(-1, 3) - t).max())
            assert trans < 1e-4, f"{k}: translation not reproduced: {trans:.2e}"
            hid = max(float(np.abs(
                M[3 * h:3 * h + 3].toarray()[:, 3 * coarse_of[int(h)]:3 * coarse_of[int(h)] + 3]
                - np.eye(3)).max()) for h in survivors[:25])
            assert hid < 1e-10, f"{k}: handle not identity: {hid:.2e}"
            assert np.isfinite(d_fine[k]).all(), f"{k}: non-finite displacement"
            extra = ""
            if k == "edge":                       # piecewise-constant: 1 handle/vertex
                nz = int((W[k] > 1e-9).sum(axis=1).max())
                assert nz == 1, f"edge not piecewise-constant: {nz} handles/vertex"
                extra = "  (piecewise-constant)"
            print(f"  {k:<8s}: shape OK, PoU={pou:.1e}, translation={trans:.1e}, "
                  f"handle-id={hid:.1e}{extra}")
        for k in maps:                            # global harmonic is THE energy min
            assert energies["global"] <= energies[k] * (1 + 1e-9), \
                f"global solve must minimize elastic energy (<= {k})"
        print("  global is the energy-minimizing map: "
              + ", ".join(f"{k}={energies[k]:.3e}" for k in maps))
        print("\nSMOKE: PASS")
        return

    show = {"both": ["local", "global"],
            "all": ["local", "global", "edge", "geom"]}.get(args.weights, [args.weights])
    _render(mesh, coarse, survivors, ph, W, d_coarse, d_fine, show, args.scale, diag,
            args.screenshot)


def _render(mesh, coarse, survivors, ph, W, d_coarse, d_fine, show, scale, diag, screenshot):
    import polyscope as ps
    verts, faces = mesh.vertices, mesh.faces

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")

    bbox = verts.max(0) - verts.min(0)
    dx = bbox[0] * 1.5          # col 0 = coarse proxy, cols 1.. = one fine mesh per map
    dy = bbox[1] * 1.7          # row 0 = rest, row 1 = deformed
    s = scale * 0.25 * diag / (max([np.linalg.norm(d_coarse, axis=1).max()]
                                   + [np.linalg.norm(d_fine[k], axis=1).max() for k in show])
                               + 1e-30)

    def place(X, ci, ri):
        return X + np.array([ci * dx, -ri * dy, 0.0])

    # --- col 0: coarse proxy (rest + deformed), with the painted handle's drift ---
    Pc0 = place(coarse.vertices, 0, 0)
    ps.register_surface_mesh("coarse | rest", Pc0, coarse.faces, edge_width=1.0,
                             color=[0.95, 0.6, 0.25])
    pc = ps.register_point_cloud("handles", Pc0); pc.set_radius(0.006)
    pc.set_color([0.6, 0.1, 0.1])
    hp = ps.register_point_cloud("painted handle (optimized rest)", Pc0[ph][None, :])
    hp.set_radius(0.014); hp.set_color([0.1, 0.5, 1.0])
    # SAME vertex, original position: quadratic placement slid the survivor, so handle
    # k's rest drifts from its ancestor mesh.vertices[survivors[k]] (green=original,
    # blue=optimized, line=drift; the weight blob sits at the green spot).
    anc0 = place(verts[int(survivors[ph])][None, :], 0, 0)
    ap = ps.register_point_cloud("painted handle (original position)", anc0)
    ap.set_radius(0.010); ap.set_color([0.1, 0.8, 0.2])
    dn = ps.register_curve_network("painted handle drift", np.vstack([Pc0[ph], anc0[0]]),
                                   np.array([[0, 1]]))
    dn.set_radius(0.003); dn.set_color([0.1, 0.5, 1.0])
    ps.register_surface_mesh("coarse | deformed", place(coarse.vertices + s * d_coarse, 0, 1),
                             coarse.faces, edge_width=1.0, color=[0.95, 0.6, 0.25])

    # --- cols 1..: one fine mesh per map (painted at rest, driven when deformed) ---
    for ci, k in enumerate(show, start=1):
        wmax = float(W[k][:, ph].max()) + 1e-30
        mf = ps.register_surface_mesh(f"fine {k} | rest (painted)", place(verts, ci, 0),
                                      faces, edge_width=0.0, color=[0.7, 0.7, 0.72])
        mf.add_scalar_quantity(f"LBS weight -> handle {ph}", W[k][:, ph], enabled=True,
                               cmap="viridis", vminmax=(0.0, wmax))
        sp = ps.register_point_cloud(f"fine {k} | handle source",
                                     place(verts[int(survivors[ph])][None, :], ci, 0))
        sp.set_radius(0.010); sp.set_color([0.1, 0.8, 0.2])
        mfd = ps.register_surface_mesh(f"fine {k} | deformed", place(verts + s * d_fine[k], ci, 1),
                                       faces, edge_width=0.0, color=[0.55, 0.75, 0.95])
        mfd.add_scalar_quantity("driven displacement", np.linalg.norm(d_fine[k], axis=1),
                                enabled=False, cmap="viridis")

    ps.reset_camera_to_home_view()
    print("\n  Polyscope open. Col 0 = coarse proxy (rest/deformed; blue=handle, green=its")
    print(f"  original position, line=drift). Fine columns ({' , '.join(show)}):")
    print("  top = painted LBS weight to the handle; bottom = driven by d_fine = map @ d_coarse.")
    if screenshot:
        # size the window to the column count so all (coarse + N maps) columns fit
        # the orthographic home view (default aspect crops a 4th column otherwise).
        ncols = 1 + len(show)
        agg = (bbox[0] * (1.5 * ncols - 0.5)) / (2.7 * bbox[1] + 1e-30)
        ps.set_window_size(int(720 * max(agg, 1.0)), 720)
        ps.set_view_projection_mode("orthographic")
        ps.reset_camera_to_home_view()
        ps.screenshot(screenshot, transparent_bg=False)
        print(f"  saved {screenshot}")
    else:
        ps.show()


if __name__ == "__main__":
    main()
