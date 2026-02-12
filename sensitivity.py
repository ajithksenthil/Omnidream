"""Sensitivity analysis for the Omnidream TMS array action space.

Computes the geometry of the optimisation landscape:
    - Jacobians (∂y/∂α) via central finite differences and analytical formulas
    - Hessians (∂²J/∂α²) of the scalar objective
    - Condition numbers (κ = σ_max/σ_min) characterising problem difficulty
    - Reachable sets (convex hull of achievable output pairs)
    - Pareto fronts (non-dominated depth-selectivity trade-off boundary)

This module depends on:
    - ti_fields.py (TI forward model)
    - nts_timing.py (NTS forward model)
    - control_framework.py (TMSPlant, GoalSpec, build_cost_function)
    - basis_fields.py (synthetic basis generation)

See ``Blueprints/control_theory.md`` §4 for the mathematical foundation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


# =====================================================================
# §1 — Jacobian Computation
# =====================================================================

def compute_jacobian_ti(
    amplitudes: np.ndarray,
    group: np.ndarray,
    freq1: float,
    freq2: float,
    basis_matrix: np.ndarray,
    target_idx: Any,
    surface_indices: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Compute Jacobian of TI outputs w.r.t. amplitudes via central differences.

    Returns
    -------
    dict with keys:
        "J_M_target"      : ndarray (N,) — ∂M_target/∂α_i
        "J_M_surface_max" : ndarray (N,) — ∂M_surface_max/∂α_i
        "J_SAR_max"       : ndarray (N,) — ∂SAR_max/∂α_i
        "J_full"          : ndarray (3, N) — stacked Jacobian rows
    """
    from ti_fields import compute_group_amplitudes, compute_modulation_depth, compute_ti_sar

    N = len(amplitudes)

    def _eval(a: np.ndarray) -> np.ndarray:
        A1, A2 = compute_group_amplitudes(a, group, basis_matrix, freq1, freq2)
        M = compute_modulation_depth(A1, A2)
        M_target = float(np.mean(M[target_idx]))
        M_surface = float(np.max(M[surface_indices])) if len(surface_indices) > 0 else 0.0
        SAR = compute_ti_sar(A1, A2)
        SAR_max = float(np.max(SAR[surface_indices])) if len(surface_indices) > 0 else 0.0
        return np.array([M_target, M_surface, SAR_max])

    J = np.zeros((3, N))
    for j in range(N):
        a_plus = amplitudes.copy()
        a_minus = amplitudes.copy()
        a_plus[j] += eps
        a_minus[j] -= eps
        a_minus[j] = max(a_minus[j], 0.0)
        y_plus = _eval(a_plus)
        y_minus = _eval(a_minus)
        J[:, j] = (y_plus - y_minus) / (a_plus[j] - a_minus[j] + 1e-30)

    return {
        "J_M_target": J[0],
        "J_M_surface_max": J[1],
        "J_SAR_max": J[2],
        "J_full": J,
    }


def compute_jacobian_nts(
    amplitudes: np.ndarray,
    fire_times: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    surface_indices: np.ndarray,
    tau_m: float = 3e-3,
    q_pulse: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Compute Jacobian of NTS outputs w.r.t. amplitudes AND fire_times.

    Returns
    -------
    dict with keys:
        "J_V_target_alpha"   : ndarray (N,) — ∂V_target/∂α_i
        "J_V_target_time"    : ndarray (N,) — ∂V_target/∂t_i
        "J_V_surface_alpha"  : ndarray (N,) — ∂V_surface_max/∂α_i
        "J_V_surface_time"   : ndarray (N,) — ∂V_surface_max/∂t_i
        "J_alpha"            : ndarray (2, N) — [V_target; V_surface] w.r.t. α
        "J_time"             : ndarray (2, N) — [V_target; V_surface] w.r.t. t
    """
    from nts_timing import compute_v_peak

    N = len(amplitudes)

    def _eval_a(a: np.ndarray) -> np.ndarray:
        V = compute_v_peak(a, fire_times, basis_matrix, tau_m, q_pulse)
        V_t = float(np.mean(V[target_idx]))
        V_s = float(np.max(V[surface_indices])) if len(surface_indices) > 0 else 0.0
        return np.array([V_t, V_s])

    def _eval_t(t: np.ndarray) -> np.ndarray:
        V = compute_v_peak(amplitudes, t, basis_matrix, tau_m, q_pulse)
        V_t = float(np.mean(V[target_idx]))
        V_s = float(np.max(V[surface_indices])) if len(surface_indices) > 0 else 0.0
        return np.array([V_t, V_s])

    # Jacobian w.r.t. amplitudes
    J_alpha = np.zeros((2, N))
    for j in range(N):
        a_plus = amplitudes.copy()
        a_minus = amplitudes.copy()
        a_plus[j] += eps
        a_minus[j] -= eps
        a_minus[j] = max(a_minus[j], 0.0)
        y_plus = _eval_a(a_plus)
        y_minus = _eval_a(a_minus)
        J_alpha[:, j] = (y_plus - y_minus) / (a_plus[j] - a_minus[j] + 1e-30)

    # Jacobian w.r.t. fire_times
    J_time = np.zeros((2, N))
    eps_t = 1e-7
    for j in range(N):
        t_plus = fire_times.copy()
        t_minus = fire_times.copy()
        t_plus[j] += eps_t
        t_minus[j] -= eps_t
        t_minus[j] = max(t_minus[j], 0.0)
        y_plus = _eval_t(t_plus)
        y_minus = _eval_t(t_minus)
        J_time[:, j] = (y_plus - y_minus) / (t_plus[j] - t_minus[j] + 1e-30)

    return {
        "J_V_target_alpha": J_alpha[0],
        "J_V_target_time": J_time[0],
        "J_V_surface_alpha": J_alpha[1],
        "J_V_surface_time": J_time[1],
        "J_alpha": J_alpha,
        "J_time": J_time,
    }


def compute_jacobian_analytical_nts(
    amplitudes: np.ndarray,
    fire_times: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    tau_m: float = 3e-3,
    q_pulse: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Analytical Jacobian for NTS at the target (no finite differences).

    From control_theory.md §2.2:
        ∂V_peak(r)/∂αᵢ = Q · |Eᵢ(r)| · exp(−(t_N − tᵢ)/τ_m)
        ∂V_peak(r)/∂tᵢ = Q · αᵢ · |Eᵢ(r)| · (1/τ_m) · exp(−(t_N − tᵢ)/τ_m)
    """
    t_last = np.max(fire_times)
    decay = np.exp(-(t_last - fire_times) / tau_m)

    # Target fields
    E_target = np.abs(basis_matrix[target_idx])
    if E_target.ndim > 1:
        E_target = E_target.mean(axis=0)

    J_alpha = q_pulse * E_target * decay
    J_time = q_pulse * amplitudes * E_target * decay / tau_m

    return {
        "J_V_target_alpha": J_alpha,
        "J_V_target_time": J_time,
    }


# =====================================================================
# §2 — Hessian Computation
# =====================================================================

def compute_hessian(
    amplitudes: np.ndarray,
    objective_fn: Callable[[np.ndarray], float],
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute the Hessian H_ij = ∂²J/∂α_i∂α_j via central finite differences.

    Parameters
    ----------
    amplitudes : ndarray (N,)
    objective_fn : callable (ndarray → float)
        Scalar objective function.
    eps : float
        Step size for finite differences.

    Returns
    -------
    H : ndarray (N, N)
        Symmetric Hessian matrix.
    """
    N = len(amplitudes)
    H = np.zeros((N, N))

    f0 = objective_fn(amplitudes)

    for i in range(N):
        for j in range(i, N):
            a_pp = amplitudes.copy()
            a_pm = amplitudes.copy()
            a_mp = amplitudes.copy()
            a_mm = amplitudes.copy()

            a_pp[i] += eps
            a_pp[j] += eps
            a_pm[i] += eps
            a_pm[j] -= eps
            a_mp[i] -= eps
            a_mp[j] += eps
            a_mm[i] -= eps
            a_mm[j] -= eps

            H[i, j] = (objective_fn(a_pp) - objective_fn(a_pm)
                        - objective_fn(a_mp) + objective_fn(a_mm)) / (4 * eps * eps)
            H[j, i] = H[i, j]

    return H


# =====================================================================
# §3 — Condition Number
# =====================================================================

def compute_condition_number(J: np.ndarray) -> float:
    """Compute κ = σ_max/σ_min from SVD of a Jacobian matrix.

    Parameters
    ----------
    J : ndarray (p, m)
        Jacobian matrix.

    Returns
    -------
    kappa : float
        Condition number (inf if singular).
    """
    sv = np.linalg.svd(J, compute_uv=False)
    sv_pos = sv[sv > 1e-15]
    if len(sv_pos) < 2:
        return float("inf")
    return float(sv_pos[0] / sv_pos[-1])


def compute_hessian_condition(H: np.ndarray) -> float:
    """Condition number of the Hessian (eigenvalue ratio)."""
    eigvals = np.linalg.eigvalsh(H)
    eigvals_pos = eigvals[eigvals > 1e-15]
    if len(eigvals_pos) < 2:
        return float("inf")
    return float(eigvals_pos[-1] / eigvals_pos[0])


def svd_analysis(J: np.ndarray) -> Dict[str, Any]:
    """Full SVD decomposition with interpretable output.

    Returns
    -------
    dict with:
        "U"               : left singular vectors (output directions)
        "S"               : singular values
        "Vt"              : right singular vectors (input directions)
        "condition_number" : κ
        "rank"            : numerical rank
        "explained_variance" : cumulative fraction of variance per component
    """
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    total_var = np.sum(S ** 2)
    cum_var = np.cumsum(S ** 2) / (total_var + 1e-30)

    return {
        "U": U,
        "S": S,
        "Vt": Vt,
        "condition_number": compute_condition_number(J),
        "rank": int(np.sum(S > 1e-10)),
        "explained_variance": cum_var,
    }


# =====================================================================
# §4 — Reachable Set
# =====================================================================

def compute_reachable_set_ti(
    basis_matrix: np.ndarray,
    group: np.ndarray,
    freq1: float,
    freq2: float,
    target_idx: Any,
    surface_indices: np.ndarray,
    alpha_max: float = 1.0,
    n_samples: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Compute the reachable set in (M_target, M_surface_max) space for TI.

    Uniformly samples the amplitude hypercube [0, α_max]^N and evaluates
    the TI forward model at each sample.

    Returns
    -------
    dict with:
        "M_target"      : ndarray (n_samples,) — target modulation depth
        "M_surface_max" : ndarray (n_samples,) — max surface modulation
        "amplitudes"    : ndarray (n_samples, N) — sampled amplitudes
        "hull_vertices" : ndarray (K, 2) — convex hull boundary points
    """
    from ti_fields import compute_group_amplitudes, compute_modulation_depth

    if rng is None:
        rng = np.random.default_rng(42)

    N = basis_matrix.shape[1]
    amplitudes_all = rng.uniform(0, alpha_max, size=(n_samples, N))

    M_targets = np.zeros(n_samples)
    M_surfaces = np.zeros(n_samples)

    for k in range(n_samples):
        a = amplitudes_all[k]
        A1, A2 = compute_group_amplitudes(a, group, basis_matrix, freq1, freq2)
        M = compute_modulation_depth(A1, A2)
        M_targets[k] = float(np.mean(M[target_idx]))
        M_surfaces[k] = float(np.max(M[surface_indices])) if len(surface_indices) > 0 else 0.0

    # Convex hull
    hull_vertices = _convex_hull_2d(M_targets, M_surfaces)

    return {
        "M_target": M_targets,
        "M_surface_max": M_surfaces,
        "amplitudes": amplitudes_all,
        "hull_vertices": hull_vertices,
    }


def compute_reachable_set_nts(
    basis_matrix: np.ndarray,
    target_idx: Any,
    surface_indices: np.ndarray,
    tau_m: float = 3e-3,
    q_pulse: float = 1.0,
    tau_window: float = 5e-3,
    alpha_max: float = 1.0,
    n_samples: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Compute the reachable set in (V_target, V_surface_max) space for NTS.

    Samples amplitude hypercube and uses optimal firing order for each sample.
    """
    from nts_timing import compute_v_peak, optimal_firing_order, assign_uniform_fire_times

    if rng is None:
        rng = np.random.default_rng(42)

    N = basis_matrix.shape[1]
    amplitudes_all = rng.uniform(0, alpha_max, size=(n_samples, N))

    V_targets = np.zeros(n_samples)
    V_surfaces = np.zeros(n_samples)

    # Shared firing order (depends on basis, not amplitudes)
    order = optimal_firing_order(basis_matrix, target_idx)
    fire_times = assign_uniform_fire_times(order, tau_window)

    for k in range(n_samples):
        V = compute_v_peak(amplitudes_all[k], fire_times, basis_matrix, tau_m, q_pulse)
        V_targets[k] = float(np.mean(V[target_idx]))
        V_surfaces[k] = float(np.max(V[surface_indices])) if len(surface_indices) > 0 else 0.0

    hull_vertices = _convex_hull_2d(V_targets, V_surfaces)

    return {
        "V_target": V_targets,
        "V_surface_max": V_surfaces,
        "amplitudes": amplitudes_all,
        "hull_vertices": hull_vertices,
    }


def _convex_hull_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the convex hull of 2D points.

    Returns the hull vertices as an (K, 2) array, ordered counter-clockwise.
    Falls back to the full point set if scipy is unavailable.
    """
    try:
        from scipy.spatial import ConvexHull
        points = np.column_stack([x, y])
        hull = ConvexHull(points)
        indices = hull.vertices
        return points[indices]
    except (ImportError, Exception):
        # Fallback: return the extreme points
        points = np.column_stack([x, y])
        # Simple extremes: min/max of x and y
        extremes = set()
        extremes.add(int(np.argmin(x)))
        extremes.add(int(np.argmax(x)))
        extremes.add(int(np.argmin(y)))
        extremes.add(int(np.argmax(y)))
        return points[list(extremes)]


# =====================================================================
# §5 — Pareto Front
# =====================================================================

def compute_pareto_front_ti(
    basis_matrix: np.ndarray,
    group: np.ndarray,
    freq1: float,
    freq2: float,
    target_idx: Any,
    surface_indices: np.ndarray,
    alpha_max: float = 1.0,
    n_weights: int = 50,
    optimizer: str = "scipy",
) -> Dict[str, np.ndarray]:
    """Compute the Pareto front for the depth-selectivity trade-off (TI).

    Solves for each λ ∈ [0, 1]:
        α*(λ) = argmin { λ·(-M_target) + (1-λ)·M_surface_max }

    Parameters
    ----------
    n_weights : int
        Number of λ values to sweep.
    optimizer : str
        "scipy" for L-BFGS-B, "random" for best-of-random samples.

    Returns
    -------
    dict with:
        "lambdas"       : ndarray (n_weights,)
        "M_target"      : ndarray (n_weights,)
        "M_surface_max" : ndarray (n_weights,)
        "amplitudes"    : ndarray (n_weights, N) — optimal amplitudes per λ
        "is_dominated"  : ndarray (n_weights,) bool — True if point is dominated
    """
    from ti_fields import compute_group_amplitudes, compute_modulation_depth

    N = basis_matrix.shape[1]
    lambdas = np.linspace(0.01, 0.99, n_weights)

    M_targets = np.zeros(n_weights)
    M_surfaces = np.zeros(n_weights)
    opt_amplitudes = np.zeros((n_weights, N))

    for idx, lam in enumerate(lambdas):
        def _cost(a: np.ndarray) -> float:
            a = np.clip(a, 0, alpha_max)
            A1, A2 = compute_group_amplitudes(a, group, basis_matrix, freq1, freq2)
            M = compute_modulation_depth(A1, A2)
            Mt = float(np.mean(M[target_idx]))
            Ms = float(np.max(M[surface_indices])) if len(surface_indices) > 0 else 0.0
            return lam * (-Mt) + (1 - lam) * Ms

        if optimizer == "scipy":
            try:
                from scipy.optimize import minimize
                x0 = np.ones(N) * alpha_max * 0.5
                bounds = [(0, alpha_max)] * N
                res = minimize(_cost, x0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 100, "ftol": 1e-8})
                a_opt = np.clip(res.x, 0, alpha_max)
            except ImportError:
                a_opt = _random_search(_cost, N, alpha_max, n_samples=200)
        else:
            a_opt = _random_search(_cost, N, alpha_max, n_samples=200)

        A1, A2 = compute_group_amplitudes(a_opt, group, basis_matrix, freq1, freq2)
        M = compute_modulation_depth(A1, A2)
        M_targets[idx] = float(np.mean(M[target_idx]))
        M_surfaces[idx] = float(np.max(M[surface_indices])) if len(surface_indices) > 0 else 0.0
        opt_amplitudes[idx] = a_opt

    # Mark dominated points
    is_dominated = _find_dominated(M_targets, M_surfaces)

    return {
        "lambdas": lambdas,
        "M_target": M_targets,
        "M_surface_max": M_surfaces,
        "amplitudes": opt_amplitudes,
        "is_dominated": is_dominated,
    }


def compute_pareto_front_nts(
    basis_matrix: np.ndarray,
    target_idx: Any,
    surface_indices: np.ndarray,
    tau_m: float = 3e-3,
    q_pulse: float = 1.0,
    tau_window: float = 5e-3,
    alpha_max: float = 1.0,
    n_weights: int = 50,
    optimizer: str = "scipy",
) -> Dict[str, np.ndarray]:
    """Compute the Pareto front for depth-selectivity trade-off (NTS)."""
    from nts_timing import compute_v_peak, optimal_firing_order, assign_uniform_fire_times

    N = basis_matrix.shape[1]
    lambdas = np.linspace(0.01, 0.99, n_weights)

    order = optimal_firing_order(basis_matrix, target_idx)
    fire_times = assign_uniform_fire_times(order, tau_window)

    V_targets = np.zeros(n_weights)
    V_surfaces = np.zeros(n_weights)
    opt_amplitudes = np.zeros((n_weights, N))

    for idx, lam in enumerate(lambdas):
        def _cost(a: np.ndarray) -> float:
            a = np.clip(a, 0, alpha_max)
            V = compute_v_peak(a, fire_times, basis_matrix, tau_m, q_pulse)
            Vt = float(np.mean(V[target_idx]))
            Vs = float(np.max(V[surface_indices])) if len(surface_indices) > 0 else 0.0
            return lam * (-Vt) + (1 - lam) * Vs

        if optimizer == "scipy":
            try:
                from scipy.optimize import minimize
                x0 = np.ones(N) * alpha_max * 0.5
                bounds = [(0, alpha_max)] * N
                res = minimize(_cost, x0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 100, "ftol": 1e-8})
                a_opt = np.clip(res.x, 0, alpha_max)
            except ImportError:
                a_opt = _random_search(_cost, N, alpha_max, n_samples=200)
        else:
            a_opt = _random_search(_cost, N, alpha_max, n_samples=200)

        V = compute_v_peak(a_opt, fire_times, basis_matrix, tau_m, q_pulse)
        V_targets[idx] = float(np.mean(V[target_idx]))
        V_surfaces[idx] = float(np.max(V[surface_indices])) if len(surface_indices) > 0 else 0.0
        opt_amplitudes[idx] = a_opt

    is_dominated = _find_dominated(V_targets, V_surfaces)

    return {
        "lambdas": lambdas,
        "V_target": V_targets,
        "V_surface_max": V_surfaces,
        "amplitudes": opt_amplitudes,
        "is_dominated": is_dominated,
    }


def _random_search(cost_fn: Callable, N: int, alpha_max: float,
                   n_samples: int = 200) -> np.ndarray:
    """Fallback random search optimiser."""
    rng = np.random.default_rng(42)
    best_cost = float("inf")
    best_a = np.ones(N) * alpha_max * 0.5
    for _ in range(n_samples):
        a = rng.uniform(0, alpha_max, N)
        c = cost_fn(a)
        if c < best_cost:
            best_cost = c
            best_a = a.copy()
    return best_a


def _find_dominated(y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """Find dominated points in a bi-objective problem.

    A point (y1_i, y2_i) is dominated if there exists another point
    (y1_j, y2_j) with y1_j >= y1_i AND y2_j <= y2_i (maximise y1, minimise y2).
    """
    n = len(y1)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if y1[j] >= y1[i] and y2[j] <= y2[i]:
                if y1[j] > y1[i] or y2[j] < y2[i]:
                    is_dominated[i] = True
                    break
    return is_dominated


# =====================================================================
# §6 — Comprehensive Sensitivity Report
# =====================================================================

def run_sensitivity_analysis(
    basis_matrix: np.ndarray,
    target_idx: Any,
    surface_indices: np.ndarray,
    config: Optional[Any] = None,
    mode: str = "TI",
    alpha_max: float = 1.0,
    n_reachable_samples: int = 500,
    n_pareto_weights: int = 50,
) -> Dict[str, Any]:
    """Run a comprehensive sensitivity analysis and return all results.

    This is the main entry point called by the pipeline.

    Parameters
    ----------
    basis_matrix : ndarray (num_points, N)
    target_idx : int or ndarray
    surface_indices : ndarray
    config : OmnidreamConfig, optional
    mode : str — "TI", "NTS", or "hybrid"
    alpha_max : float
    n_reachable_samples, n_pareto_weights : int

    Returns
    -------
    dict with comprehensive results including Jacobians, Hessians,
    condition numbers, reachable sets, and Pareto fronts.
    """
    from config import paper_baseline as _pb
    if config is None:
        config = _pb()

    N = basis_matrix.shape[1]
    results: Dict[str, Any] = {"mode": mode, "n_coils": N}

    # Reference amplitudes
    alpha_ref = np.ones(N) * alpha_max * 0.5

    if mode.upper() in ("TI", "HYBRID"):
        group = np.zeros(N)
        group[N // 2:] = 1.0
        f1 = config.ti.freq_carrier_hz
        f2 = f1 + config.ti.delta_freq_default_hz

        # Jacobian
        jac_ti = compute_jacobian_ti(alpha_ref, group, f1, f2,
                                      basis_matrix, target_idx, surface_indices)
        results["jacobian_ti"] = jac_ti
        results["condition_number_ti"] = compute_condition_number(jac_ti["J_full"])
        results["svd_ti"] = svd_analysis(jac_ti["J_full"])

        # Hessian of a simple TI objective
        from ti_fields import compute_group_amplitudes, compute_modulation_depth

        def _ti_obj(a):
            A1, A2 = compute_group_amplitudes(a, group, basis_matrix, f1, f2)
            M = compute_modulation_depth(A1, A2)
            return -float(np.mean(M[target_idx]))

        results["hessian_ti"] = compute_hessian(alpha_ref, _ti_obj)
        results["hessian_condition_ti"] = compute_hessian_condition(results["hessian_ti"])

        # Reachable set
        results["reachable_ti"] = compute_reachable_set_ti(
            basis_matrix, group, f1, f2, target_idx, surface_indices,
            alpha_max=alpha_max, n_samples=n_reachable_samples)

        # Pareto front
        results["pareto_ti"] = compute_pareto_front_ti(
            basis_matrix, group, f1, f2, target_idx, surface_indices,
            alpha_max=alpha_max, n_weights=n_pareto_weights)

    if mode.upper() in ("NTS", "HYBRID"):
        tau_m = config.nts.tau_m_s
        q_pulse = config.nts.q_pulse
        tau_window = config.nts.tau_window_s

        from nts_timing import optimal_firing_order, assign_uniform_fire_times
        order = optimal_firing_order(basis_matrix, target_idx)
        fire_times = assign_uniform_fire_times(order, tau_window)

        # Jacobian (finite-difference)
        jac_nts = compute_jacobian_nts(alpha_ref, fire_times, basis_matrix,
                                        target_idx, surface_indices,
                                        tau_m=tau_m, q_pulse=q_pulse)
        results["jacobian_nts"] = jac_nts
        results["condition_number_nts_alpha"] = compute_condition_number(jac_nts["J_alpha"])
        results["condition_number_nts_time"] = compute_condition_number(jac_nts["J_time"])

        # Analytical Jacobian (for validation)
        jac_nts_analytical = compute_jacobian_analytical_nts(
            alpha_ref, fire_times, basis_matrix, target_idx,
            tau_m=tau_m, q_pulse=q_pulse)
        results["jacobian_nts_analytical"] = jac_nts_analytical

        # Analytical vs finite-difference comparison
        fd_grad = jac_nts["J_V_target_alpha"]
        an_grad = jac_nts_analytical["J_V_target_alpha"]
        norm_fd = np.linalg.norm(fd_grad)
        if norm_fd > 1e-15:
            results["nts_jacobian_agreement"] = float(
                np.linalg.norm(fd_grad - an_grad) / norm_fd)
        else:
            results["nts_jacobian_agreement"] = 0.0

        # Hessian
        from nts_timing import compute_v_peak

        def _nts_obj(a):
            V = compute_v_peak(a, fire_times, basis_matrix, tau_m, q_pulse)
            return -float(np.mean(V[target_idx]))

        results["hessian_nts"] = compute_hessian(alpha_ref, _nts_obj)
        results["hessian_condition_nts"] = compute_hessian_condition(results["hessian_nts"])

        # Reachable set
        results["reachable_nts"] = compute_reachable_set_nts(
            basis_matrix, target_idx, surface_indices,
            tau_m=tau_m, q_pulse=q_pulse, tau_window=tau_window,
            alpha_max=alpha_max, n_samples=n_reachable_samples)

        # Pareto front
        results["pareto_nts"] = compute_pareto_front_nts(
            basis_matrix, target_idx, surface_indices,
            tau_m=tau_m, q_pulse=q_pulse, tau_window=tau_window,
            alpha_max=alpha_max, n_weights=n_pareto_weights)

    return results


def print_sensitivity_summary(results: Dict[str, Any]) -> str:
    """Format sensitivity analysis results as a human-readable summary."""
    lines = []
    lines.append(f"=== Sensitivity Analysis: {results['mode']} mode, "
                 f"{results['n_coils']} coils ===\n")

    if "condition_number_ti" in results:
        lines.append(f"TI Jacobian condition number:  κ = {results['condition_number_ti']:.2f}")
        svd = results.get("svd_ti", {})
        if "S" in svd:
            lines.append(f"  Singular values: {np.array2string(svd['S'][:5], precision=4)}")
            lines.append(f"  Effective rank:  {svd.get('rank', '?')}")
        lines.append(f"  Hessian condition number:    {results.get('hessian_condition_ti', '?'):.2f}")
        rt = results.get("reachable_ti", {})
        if "M_target" in rt:
            lines.append(f"  Reachable M_target range:    [{rt['M_target'].min():.4f}, {rt['M_target'].max():.4f}]")
            lines.append(f"  Reachable M_surface range:   [{rt['M_surface_max'].min():.4f}, {rt['M_surface_max'].max():.4f}]")
        pt = results.get("pareto_ti", {})
        if "M_target" in pt:
            non_dom = ~pt["is_dominated"]
            lines.append(f"  Pareto front points:         {int(np.sum(non_dom))} / {len(pt['is_dominated'])}")
        lines.append("")

    if "condition_number_nts_alpha" in results:
        lines.append(f"NTS Jacobian condition (α):    κ = {results['condition_number_nts_alpha']:.2f}")
        lines.append(f"NTS Jacobian condition (t):    κ = {results['condition_number_nts_time']:.2f}")
        lines.append(f"  FD vs analytical agreement:  {results.get('nts_jacobian_agreement', '?'):.2e}")
        lines.append(f"  Hessian condition number:    {results.get('hessian_condition_nts', '?'):.2f}")
        rn = results.get("reachable_nts", {})
        if "V_target" in rn:
            lines.append(f"  Reachable V_target range:    [{rn['V_target'].min():.4f}, {rn['V_target'].max():.4f}]")
            lines.append(f"  Reachable V_surface range:   [{rn['V_surface_max'].min():.4f}, {rn['V_surface_max'].max():.4f}]")
        pn = results.get("pareto_nts", {})
        if "V_target" in pn:
            non_dom = ~pn["is_dominated"]
            lines.append(f"  Pareto front points:         {int(np.sum(non_dom))} / {len(pn['is_dominated'])}")
        lines.append("")

    return "\n".join(lines)
