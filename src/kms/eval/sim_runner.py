"""Generic PBD simulation runner for proxy mesh evaluation."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pbd


def run_sim(
    V: np.ndarray,
    F: np.ndarray,
    pinned: np.ndarray,
    per_frame: Callable[[int, pbd.System], None],
    n_frames: int = 120,
    n_settle: int = 60,
    dt: float = 1.0 / 60.0,
    iters: int = 15,
    k_stretch: float = 0.99,
    k_bend: float = 0.1,
    k_damp: float = 0.05,
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
    friction: float = 0.0,
    restitution: float = 0.0,
    contact_skin: float = 0.0,
    solver: str = "jacobi",
) -> np.ndarray:
    """Run PBD simulation on a mesh and return per-frame positions.

    Args:
        V: (N, 3) vertex positions (rest pose)
        F: (M, 3) face indices
        pinned: indices of pinned (kinematic) vertices
        per_frame: called as per_frame(t, sys) before each logged step.
                   Mutate sys.X[pinned] here for kinematic driving.
        n_frames: number of frames to log after settling
        n_settle: un-logged settling frames under gravity
        dt: timestep
        iters: PBD solver iterations per step
        k_stretch: stretch constraint stiffness
        k_bend: bend constraint stiffness
        k_damp: damping coefficient
        gravity: gravity vector
        friction: collision friction
        restitution: collision restitution
        contact_skin: collision skin thickness
        solver: "jacobi" or "gauss-seidel"

    Returns:
        X: (n_frames, N, 3) logged vertex positions
    """
    mesh = pbd.build_mesh(V.astype(np.float64), F.astype(np.int64))
    sys = pbd.System.from_mesh(mesh, density=1.0, gravity=gravity)
    sys.add_constraint(pbd.Stretch.from_mesh(mesh, k=k_stretch))
    sys.add_constraint(pbd.Bend.from_mesh(mesh, k=k_bend))
    sys.pin(pinned.tolist())

    step_kw = dict(
        dt=dt, iters=iters, k_damp=k_damp,
        friction=friction, restitution=restitution,
        contact_skin=contact_skin, solver=solver,
    )

    # Settle
    for _ in range(n_settle):
        sys.step(**step_kw)

    # Logged frames
    N = V.shape[0]
    X = np.zeros((n_frames, N, 3), dtype=np.float64)
    for t in range(n_frames):
        per_frame(t, sys)
        sys.step(**step_kw)
        X[t] = sys.X

    return X
