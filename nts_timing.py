"""Neural Temporal Summation (NTS) timing computations for Omnidream.

Implements the NTS formulations from ``deep_targeting_formulations.md`` §4:
membrane decay kernel, peak membrane potential, optimal firing order, and
guard-time enforcement.
"""

from __future__ import annotations

import numpy as np

from config import NTSConfig, SafetyConfig


# ---------------------------------------------------------------------------
# Core kernel functions
# ---------------------------------------------------------------------------

def membrane_decay_weights(
    fire_times: np.ndarray,
    tau_m: float,
) -> np.ndarray:
    """Compute exponential decay weights for each pulse.

    weight_i = exp(−(t_last − t_i) / τ_m)

    Coils that fire later (closer to t_last) have weight closer to 1;
    earlier pulses have decayed more.

    Parameters
    ----------
    fire_times : ndarray of shape (N,)
        Firing time of each coil in seconds.
    tau_m : float
        Membrane time constant in seconds.

    Returns
    -------
    weights : ndarray of shape (N,)
    """
    t_last = np.max(fire_times)
    return np.exp(-(t_last - fire_times) / tau_m)


def compute_v_peak(
    amplitudes: np.ndarray,
    fire_times: np.ndarray,
    basis_matrix: np.ndarray,
    tau_m: float,
    q_pulse: float = 1.0,
) -> np.ndarray:
    """Compute peak membrane potential at every sample point.

    V_peak(r) = Q_pulse · Σ_i  α_i · |E_i(r)| · exp(−(t_N − t_i)/τ_m)

    Parameters
    ----------
    amplitudes : ndarray (N,)
        Pulse amplitude per coil.
    fire_times : ndarray (N,)
        Firing times in seconds.
    basis_matrix : ndarray (num_points, N)
        Precomputed |E_i(r)| at each sample point (non-negative).
    tau_m : float
        Membrane time constant in seconds.
    q_pulse : float
        Normalised pulse charge.

    Returns
    -------
    V_peak : ndarray (num_points,)
    """
    decay = membrane_decay_weights(fire_times, tau_m)
    effective = amplitudes * decay  # shape (N,)
    V_peak = q_pulse * np.abs(basis_matrix) @ effective
    return V_peak


# ---------------------------------------------------------------------------
# Optimal firing order
# ---------------------------------------------------------------------------

def optimal_firing_order(
    basis_matrix: np.ndarray,
    target_idx: int | np.ndarray,
) -> np.ndarray:
    """Return the firing order that maximises V_peak at the target.

    Weakest-contribution coils fire first (most decay), strongest fire last
    (least decay).  See ``deep_targeting_formulations.md`` §4.5.

    Parameters
    ----------
    basis_matrix : ndarray (num_points, N)
    target_idx : int or ndarray

    Returns
    -------
    order : ndarray of int (N,)
        Coil indices sorted from weakest to strongest at the target.
    """
    target_fields = np.abs(basis_matrix[target_idx])
    if target_fields.ndim > 1:
        target_fields = target_fields.mean(axis=0)
    return np.argsort(target_fields)  # ascending → weakest first


def assign_uniform_fire_times(
    order: np.ndarray,
    tau_window: float,
) -> np.ndarray:
    """Assign uniformly spaced firing times according to the given order.

    Parameters
    ----------
    order : ndarray of int (N,)
        Firing order (index of coil that fires 1st, 2nd, …).
    tau_window : float
        Total integration window in seconds.

    Returns
    -------
    fire_times : ndarray (N,)
        Firing time for each coil (indexed by original coil index).
    """
    n = len(order)
    if n <= 1:
        return np.zeros(n)
    uniform_slots = np.linspace(0.0, tau_window, n, endpoint=True)
    fire_times = np.empty(n)
    for slot_idx, coil_idx in enumerate(order):
        fire_times[coil_idx] = uniform_slots[slot_idx]
    return fire_times


# ---------------------------------------------------------------------------
# Guard-time enforcement
# ---------------------------------------------------------------------------

def enforce_guard_times(
    fire_times: np.ndarray,
    t_guard: float,
) -> np.ndarray:
    """Enforce minimum inter-pulse spacing.

    If two pulses are closer than ``t_guard``, the later one is pushed
    forward.  Operates on a copy.

    Parameters
    ----------
    fire_times : ndarray (N,)
    t_guard : float
        Minimum inter-pulse gap in seconds.

    Returns
    -------
    adjusted : ndarray (N,)
    """
    order = np.argsort(fire_times)
    t_sorted = fire_times[order].copy()
    for k in range(1, len(order)):
        if t_sorted[k] - t_sorted[k - 1] < t_guard:
            t_sorted[k] = t_sorted[k - 1] + t_guard
    out = np.empty_like(fire_times)
    out[order] = t_sorted
    return out


# ---------------------------------------------------------------------------
# Per-pulse safety
# ---------------------------------------------------------------------------

def compute_per_pulse_surface_max(
    amplitudes: np.ndarray,
    basis_matrix: np.ndarray,
    surface_indices: np.ndarray,
) -> np.ndarray:
    """Compute the maximum surface E-field from each coil's pulse alone.

    Parameters
    ----------
    amplitudes : ndarray (N,)
    basis_matrix : ndarray (num_points, N)
    surface_indices : ndarray of int

    Returns
    -------
    per_pulse_max : ndarray (N,)
        Maximum |α_i · E_i(r)| over all surface points for each coil.
    """
    surface_fields = np.abs(basis_matrix[surface_indices])  # (S, N)
    per_pulse_max = np.max(surface_fields * amplitudes[np.newaxis, :], axis=0)
    return per_pulse_max


# ---------------------------------------------------------------------------
# NTS fitness function
# ---------------------------------------------------------------------------

def nts_fitness(
    amplitudes: np.ndarray,
    fire_times: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: int | np.ndarray,
    surface_indices: np.ndarray,
    nts_cfg: NTSConfig | None = None,
    safety_cfg: SafetyConfig | None = None,
) -> float:
    """Compute the NTS scalarised cost (for minimisation — lower is better).

    J_NTS = −V_peak(target)
            + λ₁ · max(V_surface)
            + λ₂ · max(per_pulse_surface)
            + λ₃ · Σα²

    Parameters
    ----------
    amplitudes : ndarray (N,)
    fire_times : ndarray (N,)
    basis_matrix : ndarray (num_points, N)
    target_idx : int or ndarray
    surface_indices : ndarray of int
    nts_cfg : NTSConfig
    safety_cfg : SafetyConfig

    Returns
    -------
    cost : float
    """
    if nts_cfg is None:
        nts_cfg = NTSConfig()
    if safety_cfg is None:
        safety_cfg = SafetyConfig()

    # Enforce guard times
    t_adj = enforce_guard_times(fire_times, nts_cfg.t_guard_s)

    V_peak = compute_v_peak(
        amplitudes, t_adj, basis_matrix,
        tau_m=nts_cfg.tau_m_s, q_pulse=nts_cfg.q_pulse,
    )

    V_target = float(np.mean(V_peak[target_idx]))
    V_surface = V_peak[surface_indices]

    per_pulse_max = compute_per_pulse_surface_max(
        amplitudes, basis_matrix, surface_indices,
    )

    cost = -(
        V_target
        - nts_cfg.lambda_surface_max * float(np.max(V_surface))
        - nts_cfg.lambda_per_pulse * float(np.max(per_pulse_max))
        - nts_cfg.lambda_power * float(np.sum(amplitudes ** 2))
    )

    # Safety penalty: per-pulse surface E exceeds threshold
    if np.max(per_pulse_max) > safety_cfg.surface_e_threshold_vpm:
        cost += 1000.0

    return cost
