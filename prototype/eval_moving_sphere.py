"""Ball-across-the-skirt inspector: a rigid sphere sweeps through the hanging cloth,
shown in THREE panels side by side --

  panel 1  coarse proxy            -- PBD sim of the decimated proxy, the sphere colliding
                                      into it (the low-DOF mesh we actually simulate).
  panel 2  fine: corotational LBS  -- the SAME proxy motion driving the fine mesh through
                                      the global harmonic skin  S = -K_ee^{-1} K_er  with
                                      a per-handle finite rotation (reconstruct_corotational,
                                      exactly as in eval_sim_harmonic.py).  No fine sim here:
                                      the fine surface is PROLONGED from the proxy.
  panel 3  fine: full PBD sim      -- the fine mesh simulated DIRECTLY with the same moving
                                      sphere -- the ground truth panel 2 is approximating.

This is the moving-sphere analogue of eval_sim_harmonic.py (which sways the pinned top via
DancingSway); here, instead, a sphere translates across the cloth and pushes it.  The
scenario and ALL of its parameters -- trajectory (sphere radius, start/end x, the bbox-
derived path), proxy sim stiffness/iters, the separate full-mesh sim stiffness/iters/solver,
friction/restitution/contact-skin, settle, frames, tail-frames -- are inherited verbatim
from proxy-asset-gen's scripts/eval_scenarios/moving_sphere.py, so the two viewers show the
same physical setup.  The skinning stack (decimation -> harmonic prolongation -> co-rotational
LBS) is ours, imported from eval_sim_harmonic.py.

The only intentional deviation from moving_sphere.py: the full fine-mesh sim (panel 3) runs
by DEFAULT here (it is one of the three requested panels), with --no-sim-full to skip it for
quick iteration; moving_sphere.py made it opt-in via --sim-full.

Run:
  python prototype/eval_moving_sphere.py
  python prototype/eval_moving_sphere.py --frames 240 --handles 128
  python prototype/eval_moving_sphere.py --relax 20                 # soft handles (smoother kinks)
  python prototype/eval_moving_sphere.py --tail-frames 60           # let the cloth settle on the stopped ball
  python prototype/eval_moving_sphere.py --cache out/_ball.npz      # cache the sims; reopen instantly
  python prototype/eval_moving_sphere.py --screenshot out/ball.png
  python prototype/eval_moving_sphere.py --smoke --frames 12 --n-settle 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplify_mechanics as sm  # noqa: E402
from viz_lbs import get_lbs, DEFAULT_MESH, PROLONG_THICKNESS  # noqa: E402
from eval_sim_harmonic import (  # noqa: E402  -- the harmonic skin + co-rotational LBS stack
    find_pinned_vertices,
    build_global_S,
    build_soft_S,
    ring_edges,
    estimate_handle_rotations,
    reconstruct_corotational,
    _rotation_angle,
    _mean_edge,
)

from kms.mesh import load_obj  # noqa: E402
import pbd  # noqa: E402


SPHERE_COLOR = (0.78, 0.65, 0.46)   # warm sand-dune tan (matches moving_sphere.py)


# --------------------------------------------------------------------------- #
#  Sphere trajectory (identical to moving_sphere.py, derived from the proxy bbox)
# --------------------------------------------------------------------------- #
def sphere_trajectory(V_p0, start_x_frac, end_x_frac, radius_frac):
    """Reproduce moving_sphere.py's bbox-derived path so the scenario matches.
    The sphere walks along x at a fixed (y, z) from start to end; `park` is an
    off-stage spot used during the settle phase.  Returns (start, end, radius,
    park, diag)."""
    bbox_min = V_p0.min(axis=0)
    bbox_max = V_p0.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    park_scaling_offset = 0.4
    half_x = park_scaling_offset * (bbox_max[0] - bbox_min[0])
    cy = float(park_scaling_offset * (bbox_max[1] + bbox_min[1]))
    z_offset = 0.2
    start = np.array([start_x_frac * half_x, cy, z_offset], dtype=np.float64)
    end = np.array([end_x_frac * half_x, cy, z_offset], dtype=np.float64)
    radius = radius_frac * diag
    park = start + np.array([10.0 * diag, 0.0, 0.0])
    return start, end, radius, park, diag


def make_center_at(start, end, T_move):
    """sphere center at logged frame t -- lerp start->end over [0, T_move), then
    HELD at `end` for any tail frames (t >= T_move).  Matches moving_sphere.py's
    per_frame clamp."""
    def center_at(t):
        t_clamped = min(t, T_move - 1)
        u = t_clamped / max(T_move - 1, 1)
        return (1.0 - u) * start + u * end
    return center_at


# --------------------------------------------------------------------------- #
#  PBD sim with a moving sphere collider.  kms.eval.run_sim builds the System but
#  adds no collider, so we mirror proxy-asset-gen's run_proxy_sim here: build the
#  system, add the sphere (parked off-stage during settle), then sweep it.
# --------------------------------------------------------------------------- #
def run_sim_with_sphere(V, F, pinned, radius, park, center_at, T, n_settle, *,
                        dt, iters, k_stretch, k_bend, k_damp, friction,
                        restitution, contact_skin, solver):
    """Settle the cloth (sphere parked far away), then log T frames while the
    sphere follows center_at(t).  Returns X (T, |V|, 3)."""
    mesh = pbd.build_mesh(V.astype(np.float64), F.astype(np.int64))
    sysm = pbd.System.from_mesh(mesh, density=1.0, gravity=(0.0, -9.81, 0.0))
    sysm.add_constraint(pbd.Stretch.from_mesh(mesh, k=k_stretch))
    sysm.add_constraint(pbd.Bend.from_mesh(mesh, k=k_bend))
    sysm.pin(np.asarray(pinned, np.int64).tolist())
    sphere = pbd.Sphere(center=park.copy(), radius=float(radius))
    sysm.add_collider(sphere)

    step_kw = dict(dt=dt, iters=iters, k_damp=k_damp, friction=friction,
                   restitution=restitution, contact_skin=contact_skin, solver=solver)
    for _ in range(n_settle):                      # sphere parked off-stage: clean hang
        sysm.step(**step_kw)

    X = np.zeros((T, V.shape[0], 3), dtype=np.float64)
    for t in range(T):
        sphere.center[:] = center_at(t)            # move the ball BEFORE the step
        sysm.step(**step_kw)
        X[t] = sysm.X
    return X


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- skinning stack (mirrors eval_sim_harmonic.py) ---
    p.add_argument("--mesh", default=DEFAULT_MESH)
    p.add_argument("--handles", type=int, default=128, help="target coarse vertex count")
    p.add_argument("--bending-weight", type=float, default=1.0,
                   help="decimation bending weight (must match the cached proxy)")
    p.add_argument("--thickness", type=float, default=PROLONG_THICKNESS,
                   help="shell thickness for the harmonic map's fine K")
    p.add_argument("--smooth", type=float, default=1.0,
                   help="bending multiplier for the PROLONGATION operator "
                        "(thickness*sqrt(smooth)); smooths the interior between handles")
    p.add_argument("--relax", type=float, default=0.0,
                   help="soft-handle relaxation (0 = exact interpolation). >0 approximates "
                        "the proxy at handles (lambda = median(diag K)/relax), relaxing the "
                        "handle kink into a smooth bump. Try 5-50.")
    p.add_argument("--pin-fraction", type=float, default=0.1,
                   help="top-y fraction pinned on BOTH the proxy and the fine mesh "
                        "(the garment hangs from its top)")

    # --- moving-sphere scenario (inherited verbatim from moving_sphere.py) ---
    p.add_argument("--frames", type=int, default=240,
                   help="logged frames during the moving phase (sphere lerps start->end)")
    p.add_argument("--tail-frames", type=int, default=0,
                   help="extra logged frames AFTER the sphere reaches `end`, held in place "
                        "(show the cloth settling on the stopped ball)")
    p.add_argument("--radius-frac", type=float, default=0.10,
                   help="sphere radius as a fraction of the proxy bbox diagonal")
    p.add_argument("--start-x-frac", type=float, default=1.5,
                   help="sphere start x as a fraction of half-x-extent (1.0 = +x bbox edge)")
    p.add_argument("--end-x-frac", type=float, default=-1.5,
                   help="sphere end x as a fraction of half-x-extent (-1.0 = -x bbox edge)")
    p.add_argument("--n-settle", type=int, default=120,
                   help="un-logged steps before the trajectory begins (cloth swings to a "
                        "hanging steady state under gravity+pinning before the ball arrives)")
    # proxy sim (matches moving_sphere.py / eval_sim_harmonic.py)
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--k-damp", type=float, default=0.05)
    p.add_argument("--k-stretch", type=float, default=0.99)
    p.add_argument("--k-bend", type=float, default=0.1)
    p.add_argument("--friction", type=float, default=0.4,
                   help="tangential friction at contact (0 = frictionless; ~0.4 = "
                        "draped_on_ball default)")
    p.add_argument("--restitution", type=float, default=0.0,
                   help="normal bounce on contact (0 = inelastic)")
    p.add_argument("--contact-skin", type=float, default=0.025,
                   help="safety margin between cloth and collider, world units")
    # full fine-mesh sim (panel 3) -- denser mesh wants more iters / stiffer constraints
    p.add_argument("--no-sim-full", action="store_true",
                   help="skip the direct fine-mesh PBD sim (panel 3); show only proxy + "
                        "corotational LBS. (moving_sphere.py made the full sim opt-in via "
                        "--sim-full; here it is on by default since it is a requested panel.)")
    p.add_argument("--full-iters", type=int, default=100,
                   help="solver iterations for the full-mesh sim (higher than --iters: "
                        "constraints take longer to propagate on denser edge graphs)")
    p.add_argument("--full-k-damp", type=float, default=0.05)
    p.add_argument("--full-k-stretch", type=float, default=0.999)
    p.add_argument("--full-k-bend", type=float, default=0.5)
    p.add_argument("--full-contact-skin", type=float, default=0.025,
                   help="contact skin for the full-mesh sim")
    p.add_argument("--full-solver", choices=["jacobi", "gauss-seidel"],
                   default="gauss-seidel",
                   help="PBD solver for the full-mesh sim (Gauss-Seidel converges ~2x "
                        "faster than Jacobi on dense meshes)")

    # --- caching / output ---
    p.add_argument("--cache", default=None,
                   help="path to a .npz cache. If it exists and shapes match, load the "
                        "sims instead of recomputing; otherwise simulate and write it. "
                        "Shape checks catch frame/mesh mismatches but NOT param/trajectory "
                        "changes -- delete the file or pass --recompute when those change.")
    p.add_argument("--recompute", action="store_true",
                   help="force re-simulation and overwrite --cache")
    p.add_argument("--screenshot", default=None,
                   help="save the max-deformation frame to this path and exit (no viewer)")
    p.add_argument("--no-viz", action="store_true", help="skip the polyscope viewer")
    p.add_argument("--smoke", action="store_true",
                   help="headless: short sim + shape/finiteness asserts, then exit")
    args = p.parse_args()

    if args.smoke:                                 # keep the smoke run cheap
        args.no_viz = True

    want_full = not args.no_sim_full

    print("=== moving sphere across the skirt: proxy | corotational LBS | full sim ===")
    mesh = load_obj(args.mesh)
    stem = Path(args.mesh).stem
    print(f"  fine mesh: {mesh.n_verts}v, {mesh.n_faces}f; target {args.handles} handles, "
          f"frames {args.frames}(+{args.tail_frames} tail)")

    # --- coarse proxy + the global harmonic map (cached) ---
    survivors, coarse, _maps = get_lbs(mesh, args.handles, stem,
                                       bending_weight=args.bending_weight)
    n_fine, n_coarse = mesh.n_verts, len(survivors)
    drift = np.linalg.norm(coarse.vertices - mesh.vertices[survivors], axis=1)
    edge = _mean_edge(mesh)
    print(f"  proxy {n_coarse}v; rest-to-anchor drift {drift.mean()/edge:.2f}x/"
          f"{drift.max()/edge:.2f}x edge (absorbed by co-rotational LBS)")

    # --- sphere trajectory (from the proxy bbox, like moving_sphere.py) ---
    start, end, radius, park, diag = sphere_trajectory(
        coarse.vertices, args.start_x_frac, args.end_x_frac, args.radius_frac)
    T_move = args.frames
    T = T_move + args.tail_frames
    center_at = make_center_at(start, end, T_move)
    sphere_centers = np.stack([center_at(t) for t in range(T)])
    print(f"  sphere: r={radius:.3f}  start={np.round(start,3)}  end={np.round(end,3)}  "
          f"T={T}")

    cache_path = Path(args.cache) if args.cache else None
    load_from_cache = (cache_path is not None and cache_path.exists()
                       and not args.recompute and not args.smoke)

    if load_from_cache:
        print(f"  loading cache: {cache_path}")
        z = np.load(cache_path, allow_pickle=False)
        X_coarse = np.ascontiguousarray(z["X_coarse"])
        V_corot = np.ascontiguousarray(z["V_corot"])
        X_full = np.ascontiguousarray(z["X_full"]) if "X_full" in z.files else None
        sphere_centers = np.ascontiguousarray(z["sphere_centers"])
        radius = float(z["sphere_radius"])

        def _check(name, got, expected):
            if got != expected:
                raise SystemExit(f"cache {name} shape {got} != expected {expected}; "
                                 f"delete {cache_path} or pass --recompute.")
        _check("X_coarse", X_coarse.shape, (T, n_coarse, 3))
        _check("V_corot", V_corot.shape, (T, n_fine, 3))
        _check("sphere_centers", sphere_centers.shape, (T, 3))
        if want_full:
            if X_full is None:
                raise SystemExit(f"full sim requested but {cache_path} has no X_full; "
                                 f"pass --recompute.")
            _check("X_full", X_full.shape, (T, n_fine, 3))
        print(f"    X_coarse={X_coarse.shape}  V_corot={V_corot.shape}"
              + (f"  X_full={X_full.shape}" if X_full is not None else ""))
    else:
        # --- harmonic prolongation S, scalar weights W ---
        prolong_thickness = args.thickness * np.sqrt(args.smooth)
        if args.relax > 0:
            t0 = time.time()
            S = build_soft_S(mesh, survivors, prolong_thickness, args.relax)
            print(f"  soft-handle prolongation: relax={args.relax:g} "
                  f"(approximating, not interpolating) in {time.time()-t0:.1f}s")
        else:
            S = build_global_S(mesh, survivors, stem, prolong_thickness, args.bending_weight)
        W = sm.prolongation_scalar_weights(S, n_fine, n_coarse)

        # --- panel 1: PBD-simulate the coarse proxy with the moving sphere ---
        pinned_proxy = find_pinned_vertices(coarse.vertices, args.pin_fraction)
        assert len(pinned_proxy) > 0, "no pinned vertices on the proxy top"
        print(f"\n  [1] proxy sim ({args.n_settle} settle + {T} logged, "
              f"{len(pinned_proxy)} pinned)...")
        t0 = time.time()
        X_coarse = run_sim_with_sphere(
            coarse.vertices, coarse.faces, pinned_proxy, radius, park, center_at, T,
            args.n_settle, dt=args.dt, iters=args.iters, k_stretch=args.k_stretch,
            k_bend=args.k_bend, k_damp=args.k_damp, friction=args.friction,
            restitution=args.restitution, contact_skin=args.contact_skin, solver="jacobi")
        print(f"      {time.time()-t0:.1f}s; proxy y=[{X_coarse[...,1].min():.3f}, "
              f"{X_coarse[...,1].max():.3f}]")

        # --- panel 2: drive the fine mesh via co-rotational LBS (no fine sim) ---
        print(f"  [2] corotational LBS drive...")
        t0 = time.time()
        src, dst = ring_edges(coarse.faces, n_coarse)
        R_seq = estimate_handle_rotations(coarse.vertices, src, dst, X_coarse)
        V_corot = reconstruct_corotational(W, coarse.vertices, mesh.vertices, X_coarse, R_seq)
        ang = _rotation_angle(R_seq)
        print(f"      {time.time()-t0:.1f}s; per-handle |R| mean {ang.mean():.1f}deg / "
              f"max {ang.max():.1f}deg; max |d_fine|="
              f"{np.linalg.norm(V_corot - mesh.vertices[None], axis=2).max():.3f}")

        # --- panel 3: simulate the fine mesh DIRECTLY with the same sphere ---
        X_full = None
        if want_full:
            pinned_fine = find_pinned_vertices(mesh.vertices, args.pin_fraction)
            assert len(pinned_fine) > 0, "no pinned vertices on the fine top"
            print(f"  [3] full fine-mesh sim ({args.n_settle} settle + {T} logged, "
                  f"{len(pinned_fine)} pinned, iters={args.full_iters}, "
                  f"solver={args.full_solver})...")
            t0 = time.time()
            X_full = run_sim_with_sphere(
                mesh.vertices, mesh.faces, pinned_fine, radius, park, center_at, T,
                args.n_settle, dt=args.dt, iters=args.full_iters,
                k_stretch=args.full_k_stretch, k_bend=args.full_k_bend,
                k_damp=args.full_k_damp, friction=args.friction,
                restitution=args.restitution, contact_skin=args.full_contact_skin,
                solver=args.full_solver)
            err = np.linalg.norm(V_corot - X_full, axis=2)
            print(f"      {time.time()-t0:.1f}s; full y=[{X_full[...,1].min():.3f}, "
                  f"{X_full[...,1].max():.3f}]")
            print(f"  corotational-vs-full recon error: mean {err.mean()/edge:.2f}x / "
                  f"max {err.max()/edge:.2f}x edge")

        if cache_path is not None and not args.smoke:
            save_kw = dict(X_coarse=X_coarse, V_corot=V_corot,
                           sphere_centers=sphere_centers,
                           sphere_radius=np.float64(radius))
            if X_full is not None:
                save_kw["X_full"] = X_full
            os.makedirs(cache_path.parent, exist_ok=True)
            np.savez(cache_path, **save_kw)
            print(f"  saved cache: {cache_path}")

    if args.smoke:
        print("\n--- Smoke checks ---")
        assert X_coarse.shape == (T, n_coarse, 3), X_coarse.shape
        assert V_corot.shape == (T, n_fine, 3), V_corot.shape
        assert np.isfinite(X_coarse).all() and np.isfinite(V_corot).all(), "non-finite"
        assert np.allclose(sphere_centers[0], start), "frame-0 sphere not at start"
        assert np.allclose(sphere_centers[T_move - 1], end), "last move-frame not at end"
        if want_full:
            assert X_full is not None and X_full.shape == (T, n_fine, 3), \
                (None if X_full is None else X_full.shape)
            assert np.isfinite(X_full).all(), "full sim non-finite"
        print(f"  X_coarse {X_coarse.shape}, V_corot {V_corot.shape}"
              + (f", X_full {X_full.shape}" if X_full is not None else "")
              + "; all finite; sphere endpoints OK")
        print("\nSMOKE: PASS")
        return

    if args.no_viz and not args.screenshot:
        return

    _render(mesh, coarse, X_coarse, V_corot, X_full, sphere_centers, radius,
            args.screenshot)


# --------------------------------------------------------------------------- #
#  Three-panel viewer (proxy | corotational LBS | full sim), sphere in each pane.
#  Layout follows proxy-asset-gen's eval_viz.show_eval (y-up, ascending +x columns
#  so they read left->right as panels 1,2,3); UI follows eval_sim_harmonic._render.
# --------------------------------------------------------------------------- #
def _render(mesh, coarse, X_coarse, V_corot, X_full, sphere_centers, radius, screenshot):
    import polyscope as ps
    import polyscope.imgui as psim

    Vf, Ff = mesh.vertices, mesh.faces
    T = X_coarse.shape[0]

    # (label, faces, per-frame positions, color, edge_width)
    panels = [("1: coarse proxy (PBD + sphere)", coarse.faces, X_coarse, [0.95, 0.60, 0.25], 1.0),
              ("2: fine -- corotational LBS",      Ff,           V_corot,  [0.80, 0.62, 0.95], 0.0)]
    if X_full is not None:
        panels.append(("3: fine -- full PBD sim",  Ff,           X_full,   [0.55, 0.80, 0.55], 0.0))

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    sp = 1.25 * float(np.linalg.norm(Vf.max(0) - Vf.min(0)))
    offs = [np.array([i * sp, 0.0, 0.0]) for i in range(len(panels))]

    meshes, spheres = [], []
    for i, (lab, F, seq, col, ew) in enumerate(panels):
        m = ps.register_surface_mesh(lab, seq[0] + offs[i], F, color=col, edge_width=ew)
        meshes.append((m, seq, offs[i]))
        s = ps.register_point_cloud(f"sphere [{lab}]",
                                    (sphere_centers[0] + offs[i]).reshape(1, 3),
                                    point_render_mode="sphere", color=SPHERE_COLOR)
        s.set_radius(float(radius), relative=False)
        spheres.append((s, offs[i]))

    def show_frame(t):
        for m, seq, o in meshes:
            m.update_vertex_positions(seq[t] + o)
        for s, o in spheres:
            s.update_point_positions((sphere_centers[t] + o).reshape(1, 3))

    # screenshot: the max fine-mesh deformation frame (the strongest indentation)
    if screenshot:
        defo = np.linalg.norm(V_corot - Vf[None], axis=2).mean(1)
        t = int(np.argmax(defo))
        show_frame(t)
        pts = np.vstack([seq[t] + o for _, seq, o in meshes])
        ext = pts.max(0) - pts.min(0)
        ps.set_window_size(int(760 * max(ext[0] / (ext[1] + 1e-30), 1.0)), 760)
        ps.set_view_projection_mode("orthographic")
        ps.reset_camera_to_home_view()
        ps.screenshot(screenshot, transparent_bg=False)
        print(f"  saved {screenshot} (max-deformation frame {t})")
        return

    state = {"frame": 0, "playing": True, "fps": 60.0, "last": 0.0}

    def cb():
        now = time.perf_counter()
        if psim.Button("pause" if state["playing"] else "play"):
            state["playing"] = not state["playing"]; state["last"] = now
        psim.SameLine(); _, state["fps"] = psim.SliderFloat("fps", state["fps"], 1.0, 120.0)
        advanced = False
        if state["playing"] and now - state["last"] >= 1.0 / max(state["fps"], 1e-3):
            state["frame"] = (state["frame"] + 1) % T; state["last"] = now; advanced = True
        changed, state["frame"] = psim.SliderInt("frame", state["frame"], 0, T - 1)
        if changed or advanced:
            show_frame(state["frame"])

    show_frame(0)
    ps.set_user_callback(cb)
    ps.reset_camera_to_home_view()
    print("\n  Polyscope open -- left to right: "
          + ", ".join(lab for lab, *_ in panels) + ".")
    print("  play/pause + frame slider in the panel.")
    ps.show()


if __name__ == "__main__":
    main()
