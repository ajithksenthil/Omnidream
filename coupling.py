"""Mutual inductance and coupling compensation for the Omnidream TMS array.

Implements the coupled circuit model from ``deep_targeting_formulations.md``
§6: dipole-approximation mutual inductance, impedance matrix construction,
and voltage-command compensation.

All units are SI unless noted otherwise.
"""

from __future__ import annotations

import numpy as np

from config import CoilConfig

MU_0 = 4.0 * np.pi * 1e-7  # vacuum permeability (H/m)


# ---------------------------------------------------------------------------
# Mutual inductance estimation
# ---------------------------------------------------------------------------

def dipole_mutual_inductance(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    n_i: int,
    n_j: int,
    A_i: float,
    A_j: float,
) -> float:
    """Estimate mutual inductance using the dipole approximation.

    M_ij ≈ μ₀ · n_i · n_j · A_i · A_j / (4π · d_ij³)

    Parameters
    ----------
    pos_i, pos_j : ndarray (3,)
        Coil centre positions in **metres**.
    n_i, n_j : int
        Number of turns per coil.
    A_i, A_j : float
        Effective loop area per coil in **m²**.

    Returns
    -------
    M : float
        Mutual inductance in Henries.
    """
    d = np.linalg.norm(np.asarray(pos_i, float) - np.asarray(pos_j, float))
    if d < 1e-12:
        return 0.0
    return MU_0 * n_i * n_j * A_i * A_j / (4.0 * np.pi * d ** 3)


def build_inductance_matrix(
    positions_mm: np.ndarray,
    coil_cfg: CoilConfig | None = None,
) -> np.ndarray:
    """Build the N×N inductance matrix (self + mutual).

    Diagonal entries are the self-inductance ``L_self``.  Off-diagonal
    entries are the dipole-approximated mutual inductance ``M_ij``.

    Parameters
    ----------
    positions_mm : ndarray of shape (N, 3)
        Coil positions in **millimetres**.
    coil_cfg : CoilConfig, optional
        Provides turn count, loop area, and self-inductance.

    Returns
    -------
    L : ndarray of shape (N, N)
        Inductance matrix in Henries.
    """
    if coil_cfg is None:
        coil_cfg = CoilConfig()

    n_turns = coil_cfg.total_turns
    A_loop = coil_cfg.effective_loop_area_m2
    L_self = coil_cfg.self_inductance_h

    positions_m = positions_mm * 1e-3  # convert to metres
    N = len(positions_m)
    L = np.eye(N) * L_self

    for i in range(N):
        for j in range(i + 1, N):
            M_ij = dipole_mutual_inductance(
                positions_m[i], positions_m[j],
                n_turns, n_turns, A_loop, A_loop,
            )
            L[i, j] = M_ij
            L[j, i] = M_ij

    return L


# ---------------------------------------------------------------------------
# Impedance matrix
# ---------------------------------------------------------------------------

def build_impedance_matrix(
    L_matrix: np.ndarray,
    R_coil: float,
    freq: float,
) -> np.ndarray:
    """Build the complex impedance matrix at a given frequency.

    Z_ii = R + j·2πf·L_ii
    Z_ij = j·2πf·M_ij   (i ≠ j)

    Parameters
    ----------
    L_matrix : ndarray (N, N)
        Inductance matrix (Henries).
    R_coil : float
        Per-coil resistance (Ohms).  Applied uniformly.
    freq : float
        Operating frequency in Hz.

    Returns
    -------
    Z : ndarray (N, N), complex
        Impedance matrix.
    """
    N = L_matrix.shape[0]
    omega = 2.0 * np.pi * freq
    Z = 1j * omega * L_matrix
    Z += np.diag(np.full(N, R_coil))
    return Z


# ---------------------------------------------------------------------------
# Coupling compensation
# ---------------------------------------------------------------------------

def compensate_coupling(
    I_desired: np.ndarray,
    Z_matrix: np.ndarray,
) -> np.ndarray:
    """Compute voltage commands that produce the desired coil currents.

    V_command = Z · I_desired

    Parameters
    ----------
    I_desired : ndarray (N,), possibly complex
        Desired current phasor for each coil.
    Z_matrix : ndarray (N, N), complex
        Impedance matrix.

    Returns
    -------
    V_command : ndarray (N,), complex
        Voltage command for each channel.
    """
    return Z_matrix @ np.asarray(I_desired, complex)


def coupling_coefficient(M_ij: float, L_i: float, L_j: float) -> float:
    """Compute the coupling coefficient k = M / sqrt(L_i · L_j)."""
    denom = np.sqrt(abs(L_i * L_j))
    if denom < 1e-18:
        return 0.0
    return abs(M_ij) / denom
