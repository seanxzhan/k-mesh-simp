"""Evaluate simplification methods via PBD simulation + LBS reconstruction.

Simplifies a mesh with each method, simulates each coarse proxy with PBD,
reconstructs the fine mesh via skinning weights, and shows results side-by-side.

Layout in Polyscope:
  Top row:    coarse proxy sim (one per method)
  Bottom row: fine mesh driven by skinning weights (one per method)

Run:
  python scripts/eval_sim.py --mesh data/9423122485_cleaned.obj
  python scripts/eval_sim.py --mesh data/9423122485_cleaned.obj --target 128 --frames 180
  python scripts/eval_sim.py --mesh data/spot.obj --target 64 --smoke
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from kms.mesh import load_obj, save_obj, face_areas, TriMesh
from kms.simplify_stiffness_quadric import simplify_stiffness_quadric
from kms.eval.sim_runner import run_sim
from kms.eval.lbs_reconstruct import lbs_reconstruct
from kms.eval.scenarios import DancingSway
from kms import colors


def find_pinned_vertices(V: np.ndarray, top_fraction: float = 0.1) -> np.ndarray:
    """Find vertices in the top fraction by y-coordinate (pin the top of the garment)."""
    y = V[:, 1]
    threshold = y.max() - top_fraction * (y.max() - y.min())
    return np.where(y >= threshold)[0].astype(np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", default="data/spot.obj", help="Input mesh OBJ")
    parser.add_argument("--target", type=int, default=64, help="Target coarse vertex count")
    parser.add_argument("--thickness", type=float, default=0.01, help="Shell thickness for stiffness")
    parser.add_argument("--frames", type=int, default=360, help="Simulation frames")
    parser.add_argument("--settle", type=int, default=120, help="Settle frames before logging")
    parser.add_argument("--cycles", type=float, default=4.0, help="Sway cycles")
    parser.add_argument("--amplitude", type=float, default=0.3, help="Sway amplitude (fraction of bbox)")
    parser.add_argument("--sharpness", type=float, default=1.5,
                        help="Waveform sharpness. 1.0=smooth sine, higher=jerky snaps between extremes")
    parser.add_argument("--pin-fraction", type=float, default=0.1, help="Top fraction of vertices to pin")
    # PBD sim parameters (matched to proxy-asset-gen dancing_sway.py)
    parser.add_argument("--dt", type=float, default=1.0/60.0)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--k-stretch", type=float, default=0.99)
    parser.add_argument("--k-bend", type=float, default=0.1)
    parser.add_argument("--k-damp", type=float, default=0.05)
    parser.add_argument("--friction", type=float, default=0.4)
    parser.add_argument("--restitution", type=float, default=0.0)
    parser.add_argument("--contact-skin", type=float, default=0.025)
    parser.add_argument("--output-dir", default="out", help="Output directory")
    parser.add_argument("--smoke", action="store_true", help="Headless verification")
    args = parser.parse_args()

    mesh = load_obj(args.mesh)
    target = args.target
    stem = Path(args.mesh).stem

    print("=== Simulation Evaluation ===")
    print(f"  Input: {mesh.n_verts}v, {mesh.n_faces}f")
    print(f"  Target: {target}v, frames: {args.frames}, scenario: dancing_sway\n")

    # --- Simplify with each method ---
    methods = {}

    # QEM (use stiffness quadric in "combined" mode with lambda=0 for pure QEM + proper weights)
    # Alternatively, use stiffness_quadric with a tiny lambda for near-QEM behavior
    t0 = time.time()
    result_qem, W_qem = simplify_stiffness_quadric(
        mesh, target_verts=target, mode="combined", lam=0.0,
        use_line_quadric=True, compute_skinning_weights=True, verbose=False,
    )
    dt_qem = time.time() - t0
    methods["QEM"] = (result_qem, W_qem, dt_qem)
    print(f"  QEM:      {result_qem.n_verts}v ({dt_qem:.2f}s)")

    # Combined (Approach 3)
    t0 = time.time()
    result_combined, W_combined = simplify_stiffness_quadric(
        mesh, target_verts=target, mode="combined", thickness=args.thickness,
        use_line_quadric=True, compute_skinning_weights=True, verbose=False,
    )
    dt_combined = time.time() - t0
    methods["Combined"] = (result_combined, W_combined, dt_combined)
    print(f"  Combined: {result_combined.n_verts}v ({dt_combined:.2f}s)")

    # --- Setup scenario ---
    scenario = DancingSway(
        n_frames=args.frames,
        cycles=args.cycles,
        amplitude_x=args.amplitude,
        sharpness=args.sharpness,
    )

    # --- Run simulation for each method ---
    print(f"\n  Simulating ({args.settle} settle + {args.frames} logged frames)...")

    sim_results = {}
    for method_name, (coarse_mesh, W, _) in methods.items():
        pinned = find_pinned_vertices(coarse_mesh.vertices, args.pin_fraction)
        if len(pinned) == 0:
            print(f"  WARNING: no pinned vertices for {method_name}, skipping")
            continue

        scenario.setup(coarse_mesh.vertices, pinned)

        t0 = time.time()
        X_coarse = run_sim(
            coarse_mesh.vertices, coarse_mesh.faces, pinned,
            per_frame=scenario.per_frame,
            n_frames=args.frames,
            n_settle=args.settle,
            dt=args.dt,
            iters=args.iters,
            k_stretch=args.k_stretch,
            k_bend=args.k_bend,
            k_damp=args.k_damp,
            friction=args.friction,
            restitution=args.restitution,
            contact_skin=args.contact_skin,
        )
        dt_sim = time.time() - t0

        # Reconstruct fine mesh
        V_recon = lbs_reconstruct(W, mesh.vertices, coarse_mesh.vertices, X_coarse)

        sim_results[method_name] = (X_coarse, V_recon, coarse_mesh, dt_sim)
        print(f"    {method_name}: sim={dt_sim:.2f}s, "
              f"proxy y=[{X_coarse[...,1].min():.3f}, {X_coarse[...,1].max():.3f}]")

    # --- Save outputs ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for method_name, (X_coarse, V_recon, coarse_mesh, _) in sim_results.items():
        tag = method_name.lower().replace(" ", "_")
        np.savez(
            str(out_dir / f"{stem}_sim_{tag}_{target}.npz"),
            X_coarse=X_coarse, V_recon=V_recon,
            V_coarse_rest=coarse_mesh.vertices, F_coarse=coarse_mesh.faces,
            V_fine_rest=mesh.vertices, F_fine=mesh.faces,
        )
    print(f"\n  Saved simulation data to {out_dir}/")

    if args.smoke:
        print("\n--- Smoke checks ---")
        for method_name, (X_coarse, V_recon, coarse_mesh, _) in sim_results.items():
            assert X_coarse.shape == (args.frames, coarse_mesh.n_verts, 3), f"{method_name} X shape"
            assert V_recon.shape == (args.frames, mesh.n_verts, 3), f"{method_name} V_recon shape"
            assert np.all(np.isfinite(X_coarse)), f"{method_name} X not finite"
            assert np.all(np.isfinite(V_recon)), f"{method_name} V_recon not finite"
        print("  All checks PASS")
        print("\nSMOKE: PASS")
    else:
        import polyscope as ps

        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        ps.set_front_dir("neg_z_front")

        bbox = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        spacing_x = bbox[0] * 1.5
        spacing_y = bbox[1] * 1.8

        method_names = list(sim_results.keys())

        # Register meshes for animation
        proxy_meshes = []
        fine_meshes = []

        for i, method_name in enumerate(method_names):
            X_coarse, V_recon, coarse_mesh, _ = sim_results[method_name]

            # Top row: coarse proxy
            verts = coarse_mesh.vertices.copy()
            verts[:, 0] += i * spacing_x
            ps_proxy = ps.register_surface_mesh(
                f"{method_name} proxy ({coarse_mesh.n_verts}v)",
                verts, coarse_mesh.faces,
                edge_width=1.0, color=colors.get_color_by_index(i),
            )
            proxy_meshes.append((ps_proxy, X_coarse, i))

            # Bottom row: reconstructed fine mesh
            verts_fine = mesh.vertices.copy()
            verts_fine[:, 0] += i * spacing_x
            verts_fine[:, 1] -= spacing_y
            ps_fine = ps.register_surface_mesh(
                f"{method_name} recon ({mesh.n_verts}v)",
                verts_fine, mesh.faces,
                edge_width=0.3, color=colors.get_color_by_index(i),
            )
            fine_meshes.append((ps_fine, V_recon, i))

        # Animation state with play/pause and frame control
        import polyscope.imgui as psim

        state = {
            "frame": 0,
            "playing": True,
            "fps": 60.0,
            "last_tick": 0.0,
        }

        def animation_callback():
            now = time.perf_counter()

            play_label = "pause" if state["playing"] else "play"
            if psim.Button(play_label):
                state["playing"] = not state["playing"]
                state["last_tick"] = now
            psim.SameLine()
            _, state["fps"] = psim.SliderFloat("fps", state["fps"], 1.0, 120.0)

            advanced = False
            if state["playing"] and args.frames > 0:
                dt = 1.0 / max(state["fps"], 1e-3)
                if now - state["last_tick"] >= dt:
                    state["frame"] = (state["frame"] + 1) % args.frames
                    state["last_tick"] = now
                    advanced = True

            changed_frame, state["frame"] = psim.SliderInt(
                "frame", state["frame"], 0, max(args.frames - 1, 0),
            )

            if changed_frame or advanced:
                t = state["frame"]
                for ps_mesh, X, col_idx in proxy_meshes:
                    verts = X[t].copy()
                    verts[:, 0] += col_idx * spacing_x
                    ps_mesh.update_vertex_positions(verts)

                for ps_mesh, V, col_idx in fine_meshes:
                    verts = V[t].copy()
                    verts[:, 0] += col_idx * spacing_x
                    verts[:, 1] -= spacing_y
                    ps_mesh.update_vertex_positions(verts)

        ps.set_user_callback(animation_callback)

        print(f"\n  Polyscope open — animating {args.frames} frames.")
        print("  Top row: coarse proxy simulation")
        print("  Bottom row: fine mesh reconstructed via skinning weights")
        for i, name in enumerate(method_names):
            _, _, cm, dt = sim_results[name]
            print(f"    Col {i+1}: {name} ({cm.n_verts}v proxy, sim={dt:.1f}s)")
        ps.show()


if __name__ == "__main__":
    main()
