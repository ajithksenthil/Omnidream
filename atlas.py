"""Differential-geometric atlas over the perceptual outcome manifold.

Charts what TMS stimulation *does* (perceptual outcomes: Φ, sync R, energy,
MORL objectives), not just what amplitudes are used.  Multiple amplitude
configurations can produce the same perceptual state — a fiber bundle where
the atlas charts the base space (perceptual outcomes) and fibres represent
choice of how to get there.

Capabilities:
    - Evaluate the full perceptual state at any amplitude configuration
    - Build local charts (Jacobian, SVD, metric tensor, Hessian)
    - Compute transition maps between overlapping charts
    - Assemble a global atlas from WorldState objects
    - Analyse topology (connected components, effective dimension, boundaries)
    - Find geodesics between perceptual states
    - Densify coverage gaps adaptively

This module depends on:
    - control_framework.py (TMSPlant, forward_from_params, Jacobian)
    - cp_bridge.py (compute_total_energy, compute_collective_phi,
                     compute_sync_order_parameter, WorldState, CPBridgeConfig)
    - sensitivity.py (svd_analysis, compute_hessian, compute_condition_number)
    - trajectory.py (check_point_safety)
    - config.py (OmnidreamConfig, SafetyConfig, AtlasConfig)

See ``Blueprints/atlas_manifold.md`` (forthcoming) for the mathematical
foundation: pullback metric, chart transitions, Euler characteristic.
"""

from __future__ import annotations

import heapq
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# §1 — Data Structures
# =====================================================================

@dataclass
class PerceptualState:
    """Complete perceptual state at a single amplitude configuration.

    Attributes
    ----------
    phi : float
        Φ^{collective} — integrated information.
    sync_R : float
        Kuramoto synchronisation order parameter R ∈ [0, 1].
    energy : float
        E_total from the 4D NCA energy functional.
    energy_components : dict
        {E_nca, E_mf, E_arch, E_phi, E_couple, E_morl}.
    morl_objectives : dict
        {J_phi, J_arch, J_sync, J_task}.
    outputs : ndarray (5,)
        Plant outputs [target, surface, Vm, SAR, T_max].
    safety_margins : dict
        {sar, thermal, current, voltage} — positive = safe.
    """
    phi: float
    sync_R: float
    energy: float
    energy_components: Dict[str, float]
    morl_objectives: Dict[str, float]
    outputs: np.ndarray
    safety_margins: Dict[str, float]


@dataclass
class Chart:
    """Local coordinate chart on the perceptual manifold.

    Centred at a specific amplitude configuration, a chart stores the
    local linearisation (Jacobians), geometry (metric tensor, Hessian),
    and connectivity (neighbours, boundary status) of the manifold.

    Attributes
    ----------
    chart_id : int
    center_amplitudes : ndarray (N,)
        Amplitude configuration at the chart centre.
    center_state : PerceptualState
    J_output : ndarray (5, N)
        Plant Jacobian ∂y/∂α.
    J_perceptual : ndarray (P, N)
        Perceptual Jacobian ∂s/∂α where s = (Φ, R, E, J_phi, J_arch, J_sync, J_task).
    svd : dict
        {U, S, Vt, rank, condition_number, explained_variance}.
    metric_tensor : ndarray (N, N)
        g_{ij} = J_p^T @ W @ J_p  (pullback metric from perceptual space).
    hessian_energy : ndarray (N, N)
        ∂²E_total/∂α_i∂α_j.
    valid_radius : float
        Estimated radius where the linear approximation holds.
    neighbor_ids : list of int
        Adjacent chart IDs in the atlas graph.
    is_boundary : bool
        True if near a safety constraint boundary.
    effective_dim : int
        Local effective dimension from SVD rank.
    """
    chart_id: int
    center_amplitudes: np.ndarray
    center_state: PerceptualState
    J_output: np.ndarray
    J_perceptual: np.ndarray
    svd: Dict[str, Any]
    metric_tensor: np.ndarray
    hessian_energy: np.ndarray
    valid_radius: float
    neighbor_ids: List[int] = field(default_factory=list)
    is_boundary: bool = False
    effective_dim: int = 0


@dataclass
class TransitionMap:
    """Change-of-coordinates between two overlapping charts.

    Attributes
    ----------
    chart_a_id, chart_b_id : int
    overlap_center : ndarray (N,)
        Midpoint amplitudes between the two chart centres.
    jacobian_ab : ndarray (N, N)
        Approximate change-of-basis: ∂α_b/∂α_a ≈ J_b⁺ @ J_a.
    perceptual_distance : float
        ‖s_a − s_b‖_W in weighted perceptual space.
    energy_barrier : float
        max(E) along linear path − min(E_a, E_b).
    is_valid : bool
        True if all safety constraints satisfied along the path.
    """
    chart_a_id: int
    chart_b_id: int
    overlap_center: np.ndarray
    jacobian_ab: np.ndarray
    perceptual_distance: float
    energy_barrier: float
    is_valid: bool


@dataclass
class TopologyInfo:
    """Topological summary of the atlas.

    Attributes
    ----------
    n_components : int
        Number of connected components.
    component_ids : dict
        chart_id → component_id.
    effective_dimensions : dict
        chart_id → local effective dimension.
    mean_dimension : float
    boundary_chart_ids : list of int
    genus_estimate : int
        Estimated genus via Euler characteristic (0 = simply connected).
    """
    n_components: int
    component_ids: Dict[int, int]
    effective_dimensions: Dict[int, int]
    mean_dimension: float
    boundary_chart_ids: List[int]
    genus_estimate: int


@dataclass
class Atlas:
    """Global atlas of the perceptual outcome manifold.

    Attributes
    ----------
    charts : dict
        chart_id → Chart.
    transitions : dict
        (id_a, id_b) → TransitionMap.
    topology : TopologyInfo
    perceptual_bounds : dict
        dim_name → (min, max) across all charts.
    metadata : dict
        {mode, n_coils, build_time_s, ...}.
    """
    charts: Dict[int, Chart]
    transitions: Dict[Tuple[int, int], TransitionMap]
    topology: TopologyInfo
    perceptual_bounds: Dict[str, Tuple[float, float]]
    metadata: Dict[str, Any]


# =====================================================================
# §2 — Perceptual State Evaluation
# =====================================================================

# Default perceptual dimension weights (used for metric tensor and distance)
DEFAULT_PERCEPTUAL_WEIGHTS = {
    "phi": 2.0,
    "sync_R": 1.0,
    "energy": 0.5,
    "J_phi": 1.0,
    "J_arch": 0.5,
    "J_sync": 1.0,
    "J_task": 0.5,
}

_PERCEPTUAL_DIMS = ["phi", "sync_R", "energy", "J_phi", "J_arch", "J_sync", "J_task"]


def _weights_from_config(atlas_cfg: Any) -> Dict[str, float]:
    """Build a perceptual-weights dict from an AtlasConfig instance."""
    return {
        "phi": atlas_cfg.w_phi,
        "sync_R": atlas_cfg.w_sync_R,
        "energy": atlas_cfg.w_energy,
        "J_phi": atlas_cfg.w_J_phi,
        "J_arch": atlas_cfg.w_J_arch,
        "J_sync": atlas_cfg.w_J_sync,
        "J_task": atlas_cfg.w_J_task,
    }


def _resolve_weights(
    perceptual_weights: Optional[Dict[str, float]],
    atlas_config: Optional[Any],
) -> Dict[str, float]:
    """Resolve perceptual weights from explicit arg, atlas_config, or default."""
    if perceptual_weights is not None:
        return perceptual_weights
    if atlas_config is not None:
        return _weights_from_config(atlas_config)
    return DEFAULT_PERCEPTUAL_WEIGHTS


def _state_to_vector(state: PerceptualState) -> np.ndarray:
    """Extract perceptual state as a numeric vector (P,)."""
    return np.array([
        state.phi,
        state.sync_R,
        state.energy,
        state.morl_objectives.get("J_phi", 0.0),
        state.morl_objectives.get("J_arch", 0.0),
        state.morl_objectives.get("J_sync", 0.0),
        state.morl_objectives.get("J_task", 0.0),
    ])


def evaluate_perceptual_state(
    amplitudes: np.ndarray,
    plant: Any,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    eta_field: np.ndarray,
    group: np.ndarray,
    config: Any,
    goal: Any = None,
    cp_config: Any = None,
    **plant_kwargs: Any,
) -> PerceptualState:
    """Evaluate the full perceptual state at an amplitude configuration.

    Parameters
    ----------
    amplitudes : ndarray (N,)
    plant : TMSPlant
    te_matrix : ndarray (N, N)
    phases : ndarray (N,)
    basis_matrix : ndarray (M, N)
    target_idx : int or ndarray
    eta_field : ndarray (N,)
    group : ndarray (N,)
    config : OmnidreamConfig
    goal : GoalSpec, optional
    cp_config : CPBridgeConfig, optional

    Returns
    -------
    PerceptualState
    """
    from cp_bridge import (
        compute_total_energy,
        compute_sync_order_parameter,
    )
    from trajectory import check_point_safety

    # Forward model outputs
    outputs = plant.forward_from_params(amplitudes, **plant_kwargs)

    # Energy functional (includes phi, MORL objectives)
    energy_dict = compute_total_energy(
        plant, amplitudes, te_matrix, phases, basis_matrix, target_idx,
        eta_field, group, goal, cp_config, **plant_kwargs,
    )

    phi = energy_dict["phi_collective"]
    sync_R = compute_sync_order_parameter(phases)
    morl_objs = energy_dict.get("morl_objectives", {})

    # Energy components
    components = {
        k: energy_dict[k]
        for k in ("E_nca", "E_mf", "E_arch", "E_phi", "E_couple", "E_morl")
        if k in energy_dict
    }

    # Safety
    safety = check_point_safety(plant, amplitudes, config, **plant_kwargs)

    return PerceptualState(
        phi=phi,
        sync_R=sync_R,
        energy=energy_dict["E_total"],
        energy_components=components,
        morl_objectives=morl_objs,
        outputs=np.asarray(outputs, dtype=float),
        safety_margins=safety["margins"],
    )


# =====================================================================
# §3 — Perceptual Jacobian
# =====================================================================

def compute_perceptual_jacobian(
    amplitudes: np.ndarray,
    plant: Any,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    eta_field: np.ndarray,
    group: np.ndarray,
    config: Any,
    goal: Any = None,
    cp_config: Any = None,
    eps: float = 1e-5,
    atlas_config: Optional[Any] = None,
    **plant_kwargs: Any,
) -> np.ndarray:
    """Compute the perceptual Jacobian J_p = ∂s/∂α via central differences.

    s = [Φ, R, E, J_phi, J_arch, J_sync, J_task] (P=7 dimensions)

    Parameters
    ----------
    amplitudes : ndarray (N,)
    eps : float
        Step size for finite differences.  Overridden by
        ``atlas_config.finite_diff_eps`` when *eps* is at its default.
    atlas_config : AtlasConfig, optional
        If provided and *eps* is at its default, use
        ``atlas_config.finite_diff_eps``.

    Returns
    -------
    J_p : ndarray (P, N)
        P=7 perceptual dimensions, N = n_coils.
    """
    # Resolve eps from atlas_config when caller used the default sentinel
    if atlas_config is not None and eps == 1e-5:
        eps = atlas_config.finite_diff_eps

    N = len(amplitudes)
    P = len(_PERCEPTUAL_DIMS)
    J = np.zeros((P, N))

    common = dict(
        plant=plant, te_matrix=te_matrix, phases=phases,
        basis_matrix=basis_matrix, target_idx=target_idx,
        eta_field=eta_field, group=group, config=config,
        goal=goal, cp_config=cp_config, **plant_kwargs,
    )

    for j in range(N):
        a_plus = amplitudes.copy()
        a_minus = amplitudes.copy()
        a_plus[j] += eps
        a_minus[j] -= eps
        a_minus[j] = max(a_minus[j], 0.0)
        actual_step = a_plus[j] - a_minus[j]

        s_plus = evaluate_perceptual_state(a_plus, **common)
        s_minus = evaluate_perceptual_state(a_minus, **common)

        v_plus = _state_to_vector(s_plus)
        v_minus = _state_to_vector(s_minus)

        J[:, j] = (v_plus - v_minus) / (actual_step + 1e-30)

    return J


# =====================================================================
# §4 — Chart Construction
# =====================================================================

def build_chart(
    amplitudes: np.ndarray,
    plant: Any,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    eta_field: np.ndarray,
    group: np.ndarray,
    config: Any,
    chart_id: int = 0,
    goal: Any = None,
    cp_config: Any = None,
    perceptual_weights: Optional[Dict[str, float]] = None,
    eps: float = 1e-5,
    compute_hessian_flag: bool = True,
    atlas_config: Optional[Any] = None,
    **plant_kwargs: Any,
) -> Chart:
    """Build a local chart at the given amplitude configuration.

    Parameters
    ----------
    amplitudes : ndarray (N,)
    chart_id : int
    perceptual_weights : dict, optional
        Weights for the pullback metric.  Falls back to *atlas_config*
        weights, then ``DEFAULT_PERCEPTUAL_WEIGHTS``.
    eps : float
        Finite-difference step.  Falls back to *atlas_config*.
    compute_hessian_flag : bool
        If False, skip the expensive Hessian computation.
    atlas_config : AtlasConfig, optional
        Provides defaults for *perceptual_weights*, *eps*,
        *boundary_safety_threshold*, and valid-radius params.

    Returns
    -------
    Chart
    """
    from sensitivity import svd_analysis, compute_hessian

    N = len(amplitudes)

    # --- resolve from atlas_config ---
    if atlas_config is not None:
        if perceptual_weights is None:
            perceptual_weights = _weights_from_config(atlas_config)
        if eps == 1e-5:
            eps = atlas_config.finite_diff_eps
        boundary_threshold = atlas_config.boundary_safety_threshold
    else:
        boundary_threshold = 0.2

    if perceptual_weights is None:
        perceptual_weights = DEFAULT_PERCEPTUAL_WEIGHTS

    common = dict(
        plant=plant, te_matrix=te_matrix, phases=phases,
        basis_matrix=basis_matrix, target_idx=target_idx,
        eta_field=eta_field, group=group, config=config,
        goal=goal, cp_config=cp_config, **plant_kwargs,
    )

    # Centre state
    center_state = evaluate_perceptual_state(amplitudes, **common)

    # Plant Jacobian ∂y/∂α (5, N)
    J_output = plant.jacobian_output_wrt_amplitudes(amplitudes, **plant_kwargs)

    # Perceptual Jacobian ∂s/∂α (P, N)
    J_perceptual = compute_perceptual_jacobian(
        amplitudes, eps=eps, atlas_config=atlas_config, **common,
    )

    # SVD decomposition
    svd_result = svd_analysis(J_perceptual)

    # Pullback metric tensor: g = J_p^T @ W @ J_p
    P = J_perceptual.shape[0]
    w_diag = np.array([
        perceptual_weights.get(dim, 1.0) for dim in _PERCEPTUAL_DIMS
    ])
    W = np.diag(w_diag)
    metric_tensor = J_perceptual.T @ W @ J_perceptual

    # Energy Hessian ∂²E/∂α²
    if compute_hessian_flag:
        from cp_bridge import compute_total_energy as _cte

        def _energy_fn(a: np.ndarray) -> float:
            ed = _cte(
                plant, a, te_matrix, phases, basis_matrix, target_idx,
                eta_field, group, goal, cp_config, **plant_kwargs,
            )
            return ed["E_total"]

        hess = compute_hessian(amplitudes, _energy_fn, eps=max(eps, 1e-4))
    else:
        hess = np.zeros((N, N))

    # Valid radius estimate
    valid_radius = estimate_valid_radius(svd_result, atlas_config=atlas_config)

    # Effective dimension
    eff_dim = svd_result["rank"]

    # Boundary check
    margins = center_state.safety_margins
    is_boundary = any(v < boundary_threshold for v in margins.values())

    return Chart(
        chart_id=chart_id,
        center_amplitudes=amplitudes.copy(),
        center_state=center_state,
        J_output=J_output,
        J_perceptual=J_perceptual,
        svd=svd_result,
        metric_tensor=metric_tensor,
        hessian_energy=hess,
        valid_radius=valid_radius,
        neighbor_ids=[],
        is_boundary=is_boundary,
        effective_dim=eff_dim,
    )


def estimate_valid_radius(
    svd_result: Dict[str, Any],
    threshold: float = 0.1,
    atlas_config: Optional[Any] = None,
) -> float:
    """Estimate the radius where the linear chart approximation holds.

    Uses the ratio of the smallest to largest singular value to estimate
    how quickly the linearisation degrades.

    Parameters
    ----------
    svd_result : dict from svd_analysis
    threshold : float
        Acceptable relative error threshold.  Overridden by
        ``atlas_config.valid_radius_threshold`` when at default.
    atlas_config : AtlasConfig, optional
        Provides *valid_radius_threshold*, *valid_radius_min*,
        *valid_radius_max*.

    Returns
    -------
    radius : float
        Estimated valid radius in amplitude space.
    """
    if atlas_config is not None:
        if threshold == 0.1:
            threshold = atlas_config.valid_radius_threshold
        r_min = atlas_config.valid_radius_min
        r_max = atlas_config.valid_radius_max
    else:
        r_min = 0.001
        r_max = 10.0

    S = svd_result.get("S", np.array([1.0]))
    if len(S) == 0:
        return 0.01

    s_max = float(S[0])
    kappa = svd_result.get("condition_number", float("inf"))

    if kappa == float("inf") or kappa < 1.0:
        return 0.01

    # Heuristic: valid radius ∝ threshold / (σ_max * sqrt(κ))
    radius = threshold / (s_max * np.sqrt(kappa) + 1e-30)
    return float(np.clip(radius, r_min, r_max))


# =====================================================================
# §5 — Transition Maps
# =====================================================================

def compute_transition(
    chart_a: Chart,
    chart_b: Chart,
    plant: Any,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    eta_field: np.ndarray,
    group: np.ndarray,
    config: Any,
    goal: Any = None,
    cp_config: Any = None,
    perceptual_weights: Optional[Dict[str, float]] = None,
    n_barrier_samples: int = 10,
    atlas_config: Optional[Any] = None,
    **plant_kwargs: Any,
) -> TransitionMap:
    """Compute the transition map between two charts.

    Parameters
    ----------
    chart_a, chart_b : Chart
    n_barrier_samples : int
        Points to sample along the linear path for barrier estimation.
        Overridden by ``atlas_config.n_barrier_samples`` when at default.
    atlas_config : AtlasConfig, optional

    Returns
    -------
    TransitionMap
    """
    from trajectory import check_point_safety

    # Resolve from atlas_config
    if atlas_config is not None:
        if n_barrier_samples == 10:
            n_barrier_samples = atlas_config.n_barrier_samples

    perceptual_weights = _resolve_weights(perceptual_weights, atlas_config)

    # Midpoint
    overlap_center = 0.5 * (chart_a.center_amplitudes + chart_b.center_amplitudes)

    # Change-of-basis: J_ab = pinv(J_b) @ J_a
    try:
        J_b_pinv = np.linalg.pinv(chart_b.J_perceptual)
        jacobian_ab = J_b_pinv @ chart_a.J_perceptual
    except np.linalg.LinAlgError:
        N = len(chart_a.center_amplitudes)
        jacobian_ab = np.eye(N)

    # Perceptual distance
    perceptual_distance = compute_perceptual_distance(
        chart_a.center_state, chart_b.center_state, perceptual_weights,
    )

    # Energy barrier along linear path
    from cp_bridge import compute_total_energy as _cte

    energies = []
    is_valid = True
    for i in range(n_barrier_samples):
        s = i / max(n_barrier_samples - 1, 1)
        a_interp = (1.0 - s) * chart_a.center_amplitudes + s * chart_b.center_amplitudes
        ed = _cte(
            plant, a_interp, te_matrix, phases, basis_matrix, target_idx,
            eta_field, group, goal, cp_config, **plant_kwargs,
        )
        energies.append(ed["E_total"])

        # Safety check
        safety = check_point_safety(plant, a_interp, config, **plant_kwargs)
        if not safety["satisfied"]:
            is_valid = False

    E_min_endpoints = min(chart_a.center_state.energy, chart_b.center_state.energy)
    energy_barrier = max(energies) - E_min_endpoints if energies else 0.0
    energy_barrier = max(energy_barrier, 0.0)

    return TransitionMap(
        chart_a_id=chart_a.chart_id,
        chart_b_id=chart_b.chart_id,
        overlap_center=overlap_center,
        jacobian_ab=jacobian_ab,
        perceptual_distance=perceptual_distance,
        energy_barrier=energy_barrier,
        is_valid=is_valid,
    )


# =====================================================================
# §6 — Atlas Construction
# =====================================================================

def build_atlas(
    worlds: list,
    plant: Any,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    eta_field: np.ndarray,
    group: np.ndarray,
    config: Any,
    goal: Any = None,
    cp_config: Any = None,
    k_neighbors: int = 5,
    perceptual_weights: Optional[Dict[str, float]] = None,
    compute_hessians: bool = True,
    atlas_config: Optional[Any] = None,
    **plant_kwargs: Any,
) -> Atlas:
    """Build a global atlas from a set of WorldState objects.

    Parameters
    ----------
    worlds : list of WorldState
    k_neighbors : int
        Number of nearest neighbours for the chart graph.  Overridden by
        ``atlas_config.k_neighbors`` when at default (5).
    compute_hessians : bool
        If False, skip Hessian computation per chart (faster).
    atlas_config : AtlasConfig, optional
        Supplies defaults for *k_neighbors*, *perceptual_weights*,
        *compute_hessians*, and is forwarded to sub-functions.

    Returns
    -------
    Atlas
    """
    t_start = time.time()

    # Resolve from atlas_config
    if atlas_config is not None:
        if k_neighbors == 5:
            k_neighbors = atlas_config.k_neighbors

    perceptual_weights = _resolve_weights(perceptual_weights, atlas_config)

    common = dict(
        plant=plant, te_matrix=te_matrix, phases=phases,
        basis_matrix=basis_matrix, target_idx=target_idx,
        eta_field=eta_field, group=group, config=config,
        goal=goal, cp_config=cp_config, **plant_kwargs,
    )

    # Build charts
    charts: Dict[int, Chart] = {}
    for i, world in enumerate(worlds):
        chart = build_chart(
            amplitudes=world.amplitudes,
            chart_id=i,
            perceptual_weights=perceptual_weights,
            compute_hessian_flag=compute_hessians,
            atlas_config=atlas_config,
            **common,
        )
        charts[i] = chart

    if not charts:
        empty_topo = TopologyInfo(
            n_components=0, component_ids={}, effective_dimensions={},
            mean_dimension=0.0, boundary_chart_ids=[], genus_estimate=0,
        )
        return Atlas(
            charts={}, transitions={}, topology=empty_topo,
            perceptual_bounds={}, metadata={"n_charts": 0},
        )

    # Build k-NN graph in perceptual space
    ids = sorted(charts.keys())
    n = len(ids)
    state_vecs = np.array([_state_to_vector(charts[i].center_state) for i in ids])

    # Pairwise perceptual distances
    dist_matrix = np.zeros((n, n))
    w_arr = np.array([perceptual_weights.get(d, 1.0) for d in _PERCEPTUAL_DIMS])
    for i in range(n):
        for j in range(i + 1, n):
            diff = state_vecs[i] - state_vecs[j]
            d = float(np.sqrt(np.sum(w_arr * diff ** 2)))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Connect k nearest neighbours
    k = min(k_neighbors, max(1, n - 1))
    for i in range(n):
        sorted_indices = np.argsort(dist_matrix[i])
        neighbours = [int(j) for j in sorted_indices if j != i][:k]
        charts[ids[i]].neighbor_ids = [ids[j] for j in neighbours]

    # Compute transitions for connected pairs
    transitions: Dict[Tuple[int, int], TransitionMap] = {}
    seen_pairs = set()
    for cid, chart in charts.items():
        for nid in chart.neighbor_ids:
            pair = (min(cid, nid), max(cid, nid))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            trans = compute_transition(
                charts[pair[0]], charts[pair[1]],
                perceptual_weights=perceptual_weights,
                atlas_config=atlas_config,
                **common,
            )
            transitions[pair] = trans

    # Topology analysis
    topology = analyse_topology(charts, transitions)

    # Perceptual bounds
    perceptual_bounds: Dict[str, Tuple[float, float]] = {}
    for dim_idx, dim_name in enumerate(_PERCEPTUAL_DIMS):
        vals = [state_vecs[i][dim_idx] for i in range(n)]
        if vals:
            perceptual_bounds[dim_name] = (float(min(vals)), float(max(vals)))

    elapsed = time.time() - t_start

    metadata = {
        "n_charts": len(charts),
        "n_transitions": len(transitions),
        "n_worlds_input": len(worlds),
        "k_neighbors": k_neighbors,
        "build_time_s": elapsed,
    }

    return Atlas(
        charts=charts,
        transitions=transitions,
        topology=topology,
        perceptual_bounds=perceptual_bounds,
        metadata=metadata,
    )


# =====================================================================
# §7 — Topology Analysis
# =====================================================================

def analyse_topology(
    charts: Dict[int, Chart],
    transitions: Dict[Tuple[int, int], TransitionMap],
) -> TopologyInfo:
    """Analyse the topology of the chart graph.

    Parameters
    ----------
    charts : dict of Chart
    transitions : dict of TransitionMap

    Returns
    -------
    TopologyInfo
    """
    if not charts:
        return TopologyInfo(
            n_components=0, component_ids={}, effective_dimensions={},
            mean_dimension=0.0, boundary_chart_ids=[], genus_estimate=0,
        )

    chart_ids = list(charts.keys())

    # Build adjacency from transitions
    adj: Dict[int, set] = {cid: set() for cid in chart_ids}
    for (a_id, b_id) in transitions:
        adj[a_id].add(b_id)
        adj[b_id].add(a_id)

    # Also add from neighbor_ids (ensures connectivity even without transitions)
    for cid, chart in charts.items():
        for nid in chart.neighbor_ids:
            if nid in adj:
                adj[cid].add(nid)
                adj[nid].add(cid)

    # BFS for connected components
    visited = set()
    component_ids: Dict[int, int] = {}
    comp_idx = 0

    for start in chart_ids:
        if start in visited:
            continue
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component_ids[node] = comp_idx
            for nbr in adj.get(node, set()):
                if nbr not in visited:
                    queue.append(nbr)
        comp_idx += 1

    n_components = comp_idx

    # Effective dimensions
    effective_dimensions: Dict[int, int] = {
        cid: chart.effective_dim for cid, chart in charts.items()
    }
    dims = list(effective_dimensions.values())
    mean_dim = float(np.mean(dims)) if dims else 0.0

    # Boundary charts
    boundary_chart_ids = [cid for cid, chart in charts.items() if chart.is_boundary]

    # Euler characteristic estimate: χ = V - E + F
    # For a graph: χ ≈ V - E (ignoring faces)
    V = len(charts)
    E = len(transitions)
    # For each connected component, genus g where χ = 2 - 2g (closed surface)
    # For a graph embedded in a surface: χ = V - E (+ F for triangulations)
    # Rough estimate: genus ≈ max(0, (E - V + n_components) // 2)
    chi = V - E + n_components  # rough Euler characteristic
    genus_estimate = max(0, (1 - chi) // 2) if chi <= 1 else 0

    return TopologyInfo(
        n_components=n_components,
        component_ids=component_ids,
        effective_dimensions=effective_dimensions,
        mean_dimension=mean_dim,
        boundary_chart_ids=boundary_chart_ids,
        genus_estimate=genus_estimate,
    )


# =====================================================================
# §8 — Geodesic Computation
# =====================================================================

def find_geodesic(
    atlas: Atlas,
    state_a: PerceptualState,
    state_b: PerceptualState,
    plant: Any,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    eta_field: np.ndarray,
    group: np.ndarray,
    config: Any,
    goal: Any = None,
    cp_config: Any = None,
    n_steps: int = 100,
    perceptual_weights: Optional[Dict[str, float]] = None,
    atlas_config: Optional[Any] = None,
    **plant_kwargs: Any,
) -> Dict[str, Any]:
    """Find a geodesic path between two perceptual states.

    Uses Dijkstra through the chart graph (weighted by perceptual distance
    plus energy barrier), then refines with local interpolation.

    Parameters
    ----------
    atlas : Atlas
    state_a, state_b : PerceptualState
    n_steps : int
        Total number of points in the returned path.  Overridden by
        ``atlas_config.geodesic_n_steps`` when at default (100).
    atlas_config : AtlasConfig, optional

    Returns
    -------
    dict with:
        path_amplitudes : ndarray (n_steps, N)
        path_states : list of PerceptualState
        path_energies : ndarray (n_steps,)
        total_cost : float
        chart_sequence : list of int (chart IDs visited)
    """
    # Resolve from atlas_config
    if atlas_config is not None:
        if n_steps == 100:
            n_steps = atlas_config.geodesic_n_steps

    perceptual_weights = _resolve_weights(perceptual_weights, atlas_config)

    if not atlas.charts:
        return {
            "path_amplitudes": np.array([]),
            "path_states": [],
            "path_energies": np.array([]),
            "total_cost": float("inf"),
            "chart_sequence": [],
        }

    # Find nearest charts to start/end states
    start_id = find_nearest_chart(atlas, state_a, perceptual_weights)
    end_id = find_nearest_chart(atlas, state_b, perceptual_weights)

    # Dijkstra through chart graph
    chart_path = dijkstra_through_charts(atlas, start_id, end_id)

    if len(chart_path) < 2:
        # Single chart or no path — use direct interpolation
        chart = atlas.charts[chart_path[0]] if chart_path else list(atlas.charts.values())[0]
        amp_start = chart.center_amplitudes
        amp_end = chart.center_amplitudes
        if len(chart_path) == 1 and start_id != end_id:
            amp_start = atlas.charts[start_id].center_amplitudes
            amp_end = atlas.charts[end_id].center_amplitudes
    else:
        amp_start = atlas.charts[chart_path[0]].center_amplitudes
        amp_end = atlas.charts[chart_path[-1]].center_amplitudes

    # Build interpolated path through chart sequence
    segments = []
    if len(chart_path) >= 2:
        n_per_seg = max(2, n_steps // max(len(chart_path) - 1, 1))
        for i in range(len(chart_path) - 1):
            a = atlas.charts[chart_path[i]].center_amplitudes
            b = atlas.charts[chart_path[i + 1]].center_amplitudes
            seg = _interpolate_with_metric(a, b, n_per_seg, atlas.charts[chart_path[i]])
            if i > 0:
                seg = seg[1:]  # avoid duplicate endpoints
            segments.extend(seg)
    else:
        segments = [atlas.charts[chart_path[0]].center_amplitudes.copy()
                     for _ in range(n_steps)]

    # Resample to exactly n_steps
    path_amps = _resample_path(segments, n_steps)

    # Evaluate perceptual state at each point
    common = dict(
        plant=plant, te_matrix=te_matrix, phases=phases,
        basis_matrix=basis_matrix, target_idx=target_idx,
        eta_field=eta_field, group=group, config=config,
        goal=goal, cp_config=cp_config, **plant_kwargs,
    )
    path_states = []
    path_energies = np.zeros(n_steps)
    total_cost = 0.0

    for i in range(n_steps):
        state = evaluate_perceptual_state(path_amps[i], **common)
        path_states.append(state)
        path_energies[i] = state.energy

        if i > 0:
            total_cost += compute_perceptual_distance(
                path_states[i - 1], state, perceptual_weights,
            )

    return {
        "path_amplitudes": np.array(path_amps),
        "path_states": path_states,
        "path_energies": path_energies,
        "total_cost": total_cost,
        "chart_sequence": chart_path,
    }


def _interpolate_with_metric(
    a: np.ndarray,
    b: np.ndarray,
    n_steps: int,
    chart: Chart,
) -> List[np.ndarray]:
    """Metric-aware interpolation between two amplitude vectors.

    Uses the chart's metric tensor to determine the natural parameterisation.
    Falls back to linear if the metric is singular.
    """
    diff = b - a
    # Metric-weighted midpoint correction (first-order Christoffel-like)
    g = chart.metric_tensor
    try:
        g_inv = np.linalg.pinv(g)
        # Natural parameter: ds² = diff^T @ g @ diff
        path_len = float(np.sqrt(max(diff @ g @ diff, 0.0)))
    except (np.linalg.LinAlgError, ValueError):
        path_len = float(np.linalg.norm(diff))

    # Linear interpolation (first-order approximation to geodesic)
    path = []
    for i in range(n_steps):
        s = i / max(n_steps - 1, 1)
        alpha = (1.0 - s) * a + s * b
        path.append(alpha.copy())

    return path


def _resample_path(
    path: List[np.ndarray],
    n_target: int,
) -> List[np.ndarray]:
    """Resample a path to exactly n_target points using linear interpolation."""
    if not path:
        return []
    if len(path) == 1:
        return [path[0].copy() for _ in range(n_target)]

    # Compute cumulative arc length
    cum_len = [0.0]
    for i in range(1, len(path)):
        cum_len.append(cum_len[-1] + float(np.linalg.norm(path[i] - path[i - 1])))

    total_len = cum_len[-1]
    if total_len < 1e-30:
        return [path[0].copy() for _ in range(n_target)]

    # Uniform parameterisation
    result = []
    for k in range(n_target):
        target_len = (k / max(n_target - 1, 1)) * total_len
        # Find segment
        seg = 0
        for s in range(1, len(cum_len)):
            if cum_len[s] >= target_len:
                seg = s - 1
                break
        else:
            seg = len(path) - 2

        seg_start = cum_len[seg]
        seg_end = cum_len[seg + 1]
        seg_len = seg_end - seg_start

        if seg_len < 1e-30:
            t = 0.0
        else:
            t = (target_len - seg_start) / seg_len

        alpha = (1.0 - t) * path[seg] + t * path[seg + 1]
        result.append(alpha)

    return result


# =====================================================================
# §9 — Densification
# =====================================================================

def densify_atlas(
    atlas: Atlas,
    plant: Any,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    eta_field: np.ndarray,
    group: np.ndarray,
    config: Any,
    goal: Any = None,
    cp_config: Any = None,
    max_new: int = 20,
    perceptual_weights: Optional[Dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
    atlas_config: Optional[Any] = None,
    **plant_kwargs: Any,
) -> Atlas:
    """Add charts in coverage gaps to improve atlas resolution.

    Parameters
    ----------
    atlas : Atlas
    max_new : int
        Maximum number of new charts to add.  Overridden by
        ``atlas_config.max_new_charts`` when at default (20).
    rng : numpy Generator, optional
    atlas_config : AtlasConfig, optional
        Supplies defaults for *max_new*, *perturbation_scale*,
        *k_connect*, *barrier_samples*, *distance_threshold_quantile*.

    Returns
    -------
    Atlas (updated with new charts and transitions)
    """
    # Resolve from atlas_config
    if atlas_config is not None:
        if max_new == 20:
            max_new = atlas_config.max_new_charts
        perturbation_scale = atlas_config.densify_perturbation_scale
        k_connect = atlas_config.densify_k_connect
        barrier_samples = atlas_config.densify_barrier_samples
        dtq = atlas_config.distance_threshold_quantile
    else:
        perturbation_scale = 0.01
        k_connect = 3
        barrier_samples = 3
        dtq = 0.75

    perceptual_weights = _resolve_weights(perceptual_weights, atlas_config)

    if rng is None:
        rng = np.random.default_rng(42)

    if not atlas.charts:
        return atlas

    common = dict(
        plant=plant, te_matrix=te_matrix, phases=phases,
        basis_matrix=basis_matrix, target_idx=target_idx,
        eta_field=eta_field, group=group, config=config,
        goal=goal, cp_config=cp_config, **plant_kwargs,
    )

    # Identify gaps: pairs of connected charts with large perceptual distance
    gaps = identify_coverage_gaps(
        atlas, perceptual_weights,
        distance_threshold_quantile=dtq,
    )

    n_added = 0
    next_id = max(atlas.charts.keys()) + 1

    for gap_amps in gaps:
        if n_added >= max_new:
            break

        # Add small perturbation
        noise = rng.standard_normal(len(gap_amps)) * perturbation_scale
        new_amps = np.clip(gap_amps + noise, 0.0, None)

        chart = build_chart(
            amplitudes=new_amps,
            chart_id=next_id,
            perceptual_weights=perceptual_weights,
            compute_hessian_flag=False,  # skip for speed
            atlas_config=atlas_config,
            **common,
        )
        atlas.charts[next_id] = chart

        # Connect to nearest existing charts (exclude self)
        nearest_ids = _find_k_nearest_charts(
            atlas, chart.center_state, k=k_connect,
            perceptual_weights=perceptual_weights,
        )
        nearest_ids = [nid for nid in nearest_ids if nid != next_id]
        chart.neighbor_ids = nearest_ids
        for nid in nearest_ids:
            if nid in atlas.charts:
                atlas.charts[nid].neighbor_ids.append(next_id)

        # Compute transitions
        for nid in nearest_ids:
            if nid in atlas.charts:
                pair = (min(next_id, nid), max(next_id, nid))
                trans = compute_transition(
                    atlas.charts[pair[0]], atlas.charts[pair[1]],
                    perceptual_weights=perceptual_weights,
                    n_barrier_samples=barrier_samples,
                    atlas_config=atlas_config,
                    **common,
                )
                atlas.transitions[pair] = trans

        next_id += 1
        n_added += 1

    # Re-analyse topology
    atlas.topology = analyse_topology(atlas.charts, atlas.transitions)

    # Update metadata
    atlas.metadata["n_charts"] = len(atlas.charts)
    atlas.metadata["n_transitions"] = len(atlas.transitions)
    atlas.metadata["n_densified"] = n_added

    return atlas


def identify_coverage_gaps(
    atlas: Atlas,
    perceptual_weights: Optional[Dict[str, float]] = None,
    distance_threshold_quantile: float = 0.75,
    atlas_config: Optional[Any] = None,
) -> List[np.ndarray]:
    """Identify amplitude configurations in coverage gaps.

    Returns midpoints between distant connected chart pairs.

    Parameters
    ----------
    atlas : Atlas
    distance_threshold_quantile : float
        Fraction of perceptual distances above which a gap is declared.
        Overridden by ``atlas_config.distance_threshold_quantile``
        when at default (0.75).
    atlas_config : AtlasConfig, optional

    Returns
    -------
    gap_amplitudes : list of ndarray (N,)
    """
    if atlas_config is not None and distance_threshold_quantile == 0.75:
        distance_threshold_quantile = atlas_config.distance_threshold_quantile

    perceptual_weights = _resolve_weights(perceptual_weights, atlas_config)

    if not atlas.transitions:
        return []

    distances = [t.perceptual_distance for t in atlas.transitions.values()]
    if not distances:
        return []

    threshold = float(np.quantile(distances, distance_threshold_quantile))

    gaps = []
    for (a_id, b_id), trans in atlas.transitions.items():
        if trans.perceptual_distance > threshold:
            midpoint = 0.5 * (
                atlas.charts[a_id].center_amplitudes
                + atlas.charts[b_id].center_amplitudes
            )
            gaps.append(midpoint)

    return gaps


# =====================================================================
# §10 — Utility Functions
# =====================================================================

def compute_perceptual_distance(
    state_a: PerceptualState,
    state_b: PerceptualState,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute weighted L2 distance between two perceptual states.

    Parameters
    ----------
    state_a, state_b : PerceptualState
    weights : dict, optional
        Per-dimension weights (defaults to DEFAULT_PERCEPTUAL_WEIGHTS).

    Returns
    -------
    distance : float
    """
    if weights is None:
        weights = DEFAULT_PERCEPTUAL_WEIGHTS

    va = _state_to_vector(state_a)
    vb = _state_to_vector(state_b)
    w = np.array([weights.get(d, 1.0) for d in _PERCEPTUAL_DIMS])

    diff = va - vb
    return float(np.sqrt(np.sum(w * diff ** 2)))


def find_nearest_chart(
    atlas: Atlas,
    state: PerceptualState,
    perceptual_weights: Optional[Dict[str, float]] = None,
) -> int:
    """Find the chart whose centre is closest to a perceptual state.

    Returns
    -------
    chart_id : int
    """
    if perceptual_weights is None:
        perceptual_weights = DEFAULT_PERCEPTUAL_WEIGHTS

    best_id = -1
    best_dist = float("inf")

    for cid, chart in atlas.charts.items():
        d = compute_perceptual_distance(state, chart.center_state, perceptual_weights)
        if d < best_dist:
            best_dist = d
            best_id = cid

    return best_id


def _find_k_nearest_charts(
    atlas: Atlas,
    state: PerceptualState,
    k: int = 3,
    perceptual_weights: Optional[Dict[str, float]] = None,
) -> List[int]:
    """Find the k charts whose centres are closest to a perceptual state."""
    if perceptual_weights is None:
        perceptual_weights = DEFAULT_PERCEPTUAL_WEIGHTS

    dists = []
    for cid, chart in atlas.charts.items():
        d = compute_perceptual_distance(state, chart.center_state, perceptual_weights)
        dists.append((d, cid))

    dists.sort()
    return [cid for _, cid in dists[:k]]


def dijkstra_through_charts(
    atlas: Atlas,
    start_id: int,
    end_id: int,
) -> List[int]:
    """Find shortest path through the chart graph via Dijkstra.

    Edge cost = perceptual_distance + energy_barrier.

    Parameters
    ----------
    atlas : Atlas
    start_id, end_id : int

    Returns
    -------
    path : list of chart IDs
    """
    if start_id == end_id:
        return [start_id]

    if start_id not in atlas.charts or end_id not in atlas.charts:
        return [start_id]

    # Build adjacency with costs
    adj: Dict[int, List[Tuple[float, int]]] = {cid: [] for cid in atlas.charts}

    for (a_id, b_id), trans in atlas.transitions.items():
        cost = trans.perceptual_distance + trans.energy_barrier
        adj[a_id].append((cost, b_id))
        adj[b_id].append((cost, a_id))

    # Also use neighbor_ids for connectivity
    for cid, chart in atlas.charts.items():
        for nid in chart.neighbor_ids:
            if nid in atlas.charts:
                # Use perceptual distance as cost if no transition exists
                pair = (min(cid, nid), max(cid, nid))
                if pair not in atlas.transitions:
                    d = compute_perceptual_distance(
                        chart.center_state,
                        atlas.charts[nid].center_state,
                    )
                    adj[cid].append((d, nid))
                    adj[nid].append((d, cid))

    # Dijkstra
    dist = {cid: float("inf") for cid in atlas.charts}
    parent: Dict[int, int] = {}
    dist[start_id] = 0.0
    heap = [(0.0, start_id)]

    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        if node == end_id:
            break
        for cost, nbr in adj.get(node, []):
            new_d = d + cost
            if new_d < dist.get(nbr, float("inf")):
                dist[nbr] = new_d
                parent[nbr] = node
                heapq.heappush(heap, (new_d, nbr))

    # Reconstruct path
    if end_id not in parent and start_id != end_id:
        # No path found — return direct
        return [start_id, end_id]

    path = [end_id]
    cur = end_id
    while cur != start_id and cur in parent:
        cur = parent[cur]
        path.append(cur)
    path.reverse()

    return path


# =====================================================================
# §11 — Serialisation
# =====================================================================

def save_atlas(atlas: Atlas, path: Any) -> None:
    """Save atlas to a compressed .npz file.

    Parameters
    ----------
    atlas : Atlas
    path : str or Path
    """
    path = Path(path)
    save_dict: Dict[str, Any] = {}

    # Chart data
    chart_ids = sorted(atlas.charts.keys())
    n_charts = len(chart_ids)

    if n_charts > 0:
        first_chart = atlas.charts[chart_ids[0]]
        N = len(first_chart.center_amplitudes)
        P = first_chart.J_perceptual.shape[0]

        save_dict["chart_ids"] = np.array(chart_ids)
        save_dict["center_amplitudes"] = np.array(
            [atlas.charts[i].center_amplitudes for i in chart_ids])
        save_dict["energies"] = np.array(
            [atlas.charts[i].center_state.energy for i in chart_ids])
        save_dict["phis"] = np.array(
            [atlas.charts[i].center_state.phi for i in chart_ids])
        save_dict["sync_Rs"] = np.array(
            [atlas.charts[i].center_state.sync_R for i in chart_ids])
        save_dict["effective_dims"] = np.array(
            [atlas.charts[i].effective_dim for i in chart_ids])
        save_dict["valid_radii"] = np.array(
            [atlas.charts[i].valid_radius for i in chart_ids])
        save_dict["is_boundary"] = np.array(
            [atlas.charts[i].is_boundary for i in chart_ids])

    # Transition data
    if atlas.transitions:
        trans_keys = sorted(atlas.transitions.keys())
        save_dict["transition_pairs"] = np.array(trans_keys)
        save_dict["transition_distances"] = np.array(
            [atlas.transitions[k].perceptual_distance for k in trans_keys])
        save_dict["transition_barriers"] = np.array(
            [atlas.transitions[k].energy_barrier for k in trans_keys])
        save_dict["transition_valid"] = np.array(
            [atlas.transitions[k].is_valid for k in trans_keys])

    # Topology scalars
    save_dict["n_components"] = np.array([atlas.topology.n_components])
    save_dict["mean_dimension"] = np.array([atlas.topology.mean_dimension])
    save_dict["genus_estimate"] = np.array([atlas.topology.genus_estimate])

    # Perceptual bounds
    for dim, (lo, hi) in atlas.perceptual_bounds.items():
        save_dict[f"bounds_{dim}"] = np.array([lo, hi])

    np.savez_compressed(path, **save_dict)


def load_atlas_summary(path: Any) -> Dict[str, Any]:
    """Load atlas summary data from an .npz file.

    Returns a dict with key arrays (not the full Atlas object).
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    return dict(data)


# =====================================================================
# §12 — Summary
# =====================================================================

def print_atlas_summary(atlas: Atlas) -> str:
    """Format atlas as a human-readable summary.

    Parameters
    ----------
    atlas : Atlas

    Returns
    -------
    summary : str
    """
    lines = []
    lines.append(f"=== Perceptual Outcome Atlas ===\n")

    lines.append(f"Charts:              {len(atlas.charts)}")
    lines.append(f"Transitions:         {len(atlas.transitions)}")

    topo = atlas.topology
    lines.append(f"\nTopology:")
    lines.append(f"  Connected components: {topo.n_components}")
    lines.append(f"  Mean effective dim:   {topo.mean_dimension:.2f}")
    lines.append(f"  Boundary charts:      {len(topo.boundary_chart_ids)}")
    lines.append(f"  Genus estimate:       {topo.genus_estimate}")

    if topo.effective_dimensions:
        dims = list(topo.effective_dimensions.values())
        lines.append(f"  Dim range:            [{min(dims)}, {max(dims)}]")

    lines.append(f"\nPerceptual Bounds:")
    for dim, (lo, hi) in sorted(atlas.perceptual_bounds.items()):
        lines.append(f"  {dim:12s}: [{lo:10.4f}, {hi:10.4f}]")

    if atlas.transitions:
        dists = [t.perceptual_distance for t in atlas.transitions.values()]
        barriers = [t.energy_barrier for t in atlas.transitions.values()]
        valid_frac = sum(1 for t in atlas.transitions.values() if t.is_valid) / len(atlas.transitions)
        lines.append(f"\nTransition Statistics:")
        lines.append(f"  Mean perceptual dist: {np.mean(dists):.4f}")
        lines.append(f"  Max perceptual dist:  {np.max(dists):.4f}")
        lines.append(f"  Mean energy barrier:  {np.mean(barriers):.4f}")
        lines.append(f"  Max energy barrier:   {np.max(barriers):.4f}")
        lines.append(f"  Valid fraction:       {valid_frac:.2%}")

    if atlas.metadata:
        lines.append(f"\nMetadata:")
        for k, v in sorted(atlas.metadata.items()):
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")

    lines.append("")
    return "\n".join(lines)
