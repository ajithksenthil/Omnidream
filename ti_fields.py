"""Temporal Interference (TI) field computations for the Omnidream TMS array.

Implements the TI formulations from ``deep_targeting_formulations.md`` §3:
group amplitude decomposition, modulation depth, envelope bounds, and
SAR estimation for continuous-wave TI drive.
"""

from __future__ import annotations

import numpy as np

from config import SafetyConfig, TIConfig


def compute_group_amplitudes(
    amplitudes: np.ndarray,
    group: np.ndarray,
    basis_matrix: np.ndarray,
    freq1: float,
    freq2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the group amplitude fields A_1 and A_2 at every sample point.

    Parameters
    ----------
    amplitudes : ndarray of shape (N,)
        Per-coil amplitude weights.
    group : ndarray of shape (N,)
        Binary group assignment (0 or 1).
    basis_matrix : ndarray of shape (num_points, N)
        Pre-computed basis E-fields (V/m per A/s) for each coil.
    freq1, freq2 : float
        Carrier frequencies for group 0 and group 1 in Hz.

    Returns
    -------
    A1 : ndarray of shape (num_points,)
        Amplitude field from group 0 (units V/m).
    A2 : ndarray of shape (num_points,)
        Amplitude field from group 1 (units V/m).

    Notes
    -----
    E-field is proportional to dI/dt.  For sinusoidal drive I(t)=α·sin(2πft),
    dI/dt = α·2πf·cos(2πft), so the peak E-field contribution is
    α_i · E_i(r) · 2πf.
    """
    g = np.asarray(group, dtype=float)
    a = np.asarray(amplitudes, dtype=float)

    weights_g0 = a * (1.0 - g) * 2.0 * np.pi * freq1
    weights_g1 = a * g * 2.0 * np.pi * freq2

    A1 = basis_matrix @ weights_g0
    A2 = basis_matrix @ weights_g1
    return A1, A2


def compute_modulation_depth(
    A1: np.ndarray,
    A2: np.ndarray,
) -> np.ndarray:
    """Compute the TI modulation depth M(r) = 2·min(|A₁|, |A₂|).

    Parameters
    ----------
    A1, A2 : ndarray of shape (num_points,)
        Group amplitude fields.

    Returns
    -------
    M : ndarray of shape (num_points,)
        Modulation depth at each sample point.
    """
    return 2.0 * np.minimum(np.abs(A1), np.abs(A2))


def compute_envelope_bounds(
    A1: np.ndarray,
    A2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the envelope min/max bounds.

    E_min(r) = ||A₁| − |A₂||
    E_max(r) = |A₁| + |A₂|

    Returns
    -------
    E_min, E_max : ndarray of shape (num_points,)
    """
    abs_A1 = np.abs(A1)
    abs_A2 = np.abs(A2)
    E_min = np.abs(abs_A1 - abs_A2)
    E_max = abs_A1 + abs_A2
    return E_min, E_max


def compute_ti_sar(
    A1: np.ndarray,
    A2: np.ndarray,
    sigma: float | None = None,
    rho: float | None = None,
    safety_cfg: SafetyConfig | None = None,
) -> np.ndarray:
    """Estimate specific absorption rate for TI continuous drive.

    SAR(r) = σ · |E_rms(r)|² / ρ

    For two sinusoidal carriers at different frequencies, the time-averaged
    RMS E-field squared is 0.5·(A₁² + A₂²).

    Parameters
    ----------
    A1, A2 : ndarray of shape (num_points,)
        Group amplitude fields.
    sigma : float, optional
        Tissue conductivity (S/m).  Falls back to ``safety_cfg.sigma_gm_sm``.
    rho : float, optional
        Tissue density (kg/m³).  Falls back to ``safety_cfg.rho_tissue_kgm3``.
    safety_cfg : SafetyConfig, optional
        Provides default sigma and rho if not given directly.

    Returns
    -------
    SAR : ndarray of shape (num_points,)
        Estimated SAR in W/kg.
    """
    if safety_cfg is None:
        safety_cfg = SafetyConfig()
    if sigma is None:
        sigma = safety_cfg.sigma_gm_sm
    if rho is None:
        rho = safety_cfg.rho_tissue_kgm3

    E_rms_sq = 0.5 * (A1 ** 2 + A2 ** 2)
    return sigma * E_rms_sq / rho


def ti_fitness(
    amplitudes: np.ndarray,
    group: np.ndarray,
    freq_carrier: float,
    delta_freq: float,
    basis_matrix: np.ndarray,
    target_idx: int | np.ndarray,
    surface_indices: np.ndarray,
    ti_cfg: TIConfig | None = None,
    safety_cfg: SafetyConfig | None = None,
) -> float:
    """Compute the TI scalarised cost (for minimisation — lower is better).

    J_TI = −M(target) + λ₁·mean(M_surface) + λ₂·max(M_surface) + λ₃·Σα²

    Parameters
    ----------
    amplitudes : ndarray (N,)
    group : ndarray (N,)  — binary 0/1
    freq_carrier : float  — Hz
    delta_freq : float  — Hz
    basis_matrix : ndarray (num_points, N)
    target_idx : int or ndarray
    surface_indices : ndarray of int
    ti_cfg : TIConfig
    safety_cfg : SafetyConfig

    Returns
    -------
    cost : float  (lower is better; negate for maximisation)
    """
    if ti_cfg is None:
        ti_cfg = TIConfig()
    if safety_cfg is None:
        safety_cfg = SafetyConfig()

    f1 = freq_carrier
    f2 = freq_carrier + delta_freq

    A1, A2 = compute_group_amplitudes(amplitudes, group, basis_matrix, f1, f2)
    M = compute_modulation_depth(A1, A2)

    M_target = float(np.mean(M[target_idx]))
    M_surface = M[surface_indices]

    # SAR check
    SAR = compute_ti_sar(A1, A2, safety_cfg=safety_cfg)
    SAR_max = float(np.max(SAR[surface_indices])) if len(surface_indices) > 0 else 0.0

    cost = -(
        M_target
        - ti_cfg.lambda_surface_mean * float(np.mean(M_surface))
        - ti_cfg.lambda_surface_max * float(np.max(M_surface))
        - ti_cfg.lambda_power * float(np.sum(amplitudes ** 2))
    )

    # Safety penalty
    if SAR_max > safety_cfg.sar_limit_wkg:
        cost += 1000.0
    if len(M_surface) > 0 and np.max(M_surface) > M_target * 0.5:
        cost += 500.0

    return cost
