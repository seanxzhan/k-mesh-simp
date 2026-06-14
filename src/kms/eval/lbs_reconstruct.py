"""LBS reconstruction using sparse skinning weight matrix W."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def lbs_reconstruct(
    W: sparse.spmatrix,
    V_fine_rest: np.ndarray,
    V_coarse_rest: np.ndarray,
    X_coarse: np.ndarray,
) -> np.ndarray:
    """Reconstruct fine mesh positions from coarse simulation via linear blend.

    V_fine(t) = V_fine_rest + W @ (X_coarse(t) - V_coarse_rest)

    This is simplified LBS: each fine vertex moves by the weighted sum of
    its associated coarse vertices' displacements.

    Args:
        W: (n_fine, n_coarse) skinning weight matrix
        V_fine_rest: (n_fine, 3) fine mesh rest positions
        V_coarse_rest: (n_coarse, 3) coarse mesh rest positions
        X_coarse: (T, n_coarse, 3) coarse per-frame positions from simulation

    Returns:
        V_fine: (T, n_fine, 3) reconstructed fine mesh positions
    """
    T = X_coarse.shape[0]
    n_fine = V_fine_rest.shape[0]

    V_fine = np.zeros((T, n_fine, 3), dtype=np.float64)

    for t in range(T):
        displacement = X_coarse[t] - V_coarse_rest  # (n_coarse, 3)
        V_fine[t] = V_fine_rest + W @ displacement

    return V_fine
