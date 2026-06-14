"""Simulation scenarios for proxy mesh evaluation.

Each scenario defines how to drive the pinned vertices and what obstacles exist.
Extensible: add a new class inheriting from Scenario.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Scenario(ABC):
    """Base class for evaluation scenarios."""

    name: str = "base"

    @abstractmethod
    def setup(self, V_rest: np.ndarray, pinned: np.ndarray) -> None:
        """Initialize scenario state from the rest mesh and pinned vertices."""

    @abstractmethod
    def per_frame(self, t: int, sys) -> None:
        """Mutate sys.X for kinematic vertices each frame."""

    def obstacles_at(self, t: int) -> list:
        """Return obstacle snapshots for visualization (default: none)."""
        return []


class DancingSway(Scenario):
    """Oscillate pinned vertices laterally, mimicking hip sway.

    The pinned vertices swing side-to-side in x on a sinusoidal trajectory.
    """

    name = "dancing_sway"

    def __init__(
        self,
        n_frames: int = 120,
        cycles: float = 2.0,
        amplitude_x: float = 0.3,
        amplitude_z: float = 0.0,
        phase_z: float = 0.25,
        sharpness: float = 4.0,
    ):
        self.n_frames = n_frames
        self.cycles = cycles
        self.amplitude_x = amplitude_x
        self.amplitude_z = amplitude_z
        self.phase_z = phase_z
        self.sharpness = sharpness

    def setup(self, V_rest: np.ndarray, pinned: np.ndarray) -> None:
        self.pinned = pinned
        self.pin_rest = V_rest[pinned].copy()

        bbox_min = V_rest.min(axis=0)
        bbox_max = V_rest.max(axis=0)
        self.amp_x = self.amplitude_x * (bbox_max[0] - bbox_min[0])
        self.amp_z = self.amplitude_z * (bbox_max[2] - bbox_min[2])
        self.omega = 2.0 * np.pi * self.cycles / max(self.n_frames - 1, 1)
        self.phase_z_rad = 2.0 * np.pi * self.phase_z

    def per_frame(self, t: int, sys) -> None:
        s = self.sharpness
        theta_x = self.omega * t
        theta_z = self.omega * t + self.phase_z_rad

        dx = self.amp_x * self._sharp_sin(theta_x, s)
        dz = self.amp_z * self._sharp_sin(theta_z, s)

        sys.X[self.pinned, 0] = self.pin_rest[:, 0] + dx
        sys.X[self.pinned, 2] = self.pin_rest[:, 2] + dz

    @staticmethod
    def _sharp_sin(theta: float, s: float) -> float:
        v = np.sin(theta)
        return float(np.sign(v) * np.abs(v) ** (1.0 / s))
