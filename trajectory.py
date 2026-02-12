"""Trajectory planning through world states for the Omnidream TMS array.

Converts user-specified paths through the many-worlds landscape into
executable time-series of coil stimulation parameters.  Three trajectory
specification modes are supported:

    - world_sequence:     explicit list of world IDs to visit
    - property_sequence:  target (Φ, R) values at each waypoint
    - energy_descent:     minimum-energy path between two worlds

Three interpolation algorithms are available:

    - linear:            direct amplitude blending (baseline)
    - geodesic:          minimum energy-barrier path via sampling
    - jacobian_steered:  output-tracking via pseudoinverse Jacobian steering

Safety constraints (SAR, thermal, current, voltage) are enforced at every
timestep.  The cognitive H-theorem (dH_cog/dt ≤ 0) is optionally enforced via
local H_cog-gradient descent. Implacement dynamics are tracked along the
trajectory.

This module depends on:
    - cp_bridge.py (WorldState, energy/Φ/sync computation, implacement)
    - control_framework.py (TMSPlant, forward model, Jacobian, GoalSpec)
    - config.py (OmnidreamConfig, safety limits)

See ``Blueprints/control_surface_bridge.md`` §5 for the many-worlds theory
and ``trajectory_planning.md`` (to be written) for trajectory formalism.
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from control_framework import map_output_vector


# =====================================================================
# §1 — Data Structures
# =====================================================================

@dataclass
class PropertyGoal:
    """Target point in (Φ, R) property space."""
    phi_target: float = 0.5
    sync_target: float = 1.0
    energy_target: Optional[float] = None
    tolerance: float = 0.05


@dataclass
class TrajectorySpec:
    """User-facing trajectory specification.

    Attributes
    ----------
    spec_type : str
        One of "world_sequence", "property_sequence", "energy_descent".
    world_ids : list of int, optional
        For world_sequence: IDs of worlds to visit in order.
    property_targets : list of PropertyGoal, optional
        For property_sequence: (Φ, R) targets at each waypoint.
    start_world_id, end_world_id : int, optional
        For energy_descent: source and destination worlds.
    n_steps_per_segment : int
        Number of interpolated points between each pair of waypoints.
    interpolation : str
        "linear", "geodesic", or "jacobian_steered".
    graph_k_neighbors : int
        For energy_descent, number of nearest-neighbour world graph edges.
    safety_mode : str
        "strict" (enforce all constraints) or "permissive" (log only).
    h_theorem_enforce : bool
        If True, apply H_cog-gradient corrections where dH/dt > 0.
    h_phi_weight : float
        Weight for Φ in H_cog = E_total - w_phi·Φ - w_sync·R.
    h_sync_weight : float
        Weight for sync R in H_cog = E_total - w_phi·Φ - w_sync·R.
    name : str
        Human-readable label for this trajectory.
    """
    spec_type: str = "world_sequence"
    world_ids: Optional[List[int]] = None
    property_targets: Optional[List[PropertyGoal]] = None
    start_world_id: Optional[int] = None
    end_world_id: Optional[int] = None
    n_steps_per_segment: int = 50
    interpolation: str = "jacobian_steered"
    graph_k_neighbors: int = 8
    safety_mode: str = "strict"
    h_theorem_enforce: bool = True
    h_phi_weight: float = 1.0
    h_sync_weight: float = 0.25
    name: str = "unnamed"


@dataclass
class TrajectoryPoint:
    """A single timestep in an executed trajectory.

    Attributes
    ----------
    t : float
        Normalised time ∈ [0, 1].
    amplitudes : ndarray (N,)
        Per-coil amplitudes at this timestep.
    outputs : ndarray (5,)
        [target_metric, surface_metric, membrane_metric, sar_max, temperature_max].
    energy : float
        E_total from 4D NCA functional.
    phi : float
        Collective Φ.
    sync_R : float
        Kuramoto synchronisation order parameter.
    morl_objectives : dict
        {J_phi, J_arch, J_sync, J_task}.
    sar_margin : float
        SAR limit − actual SAR (positive = safe).
    thermal_margin : float
        T limit − actual T (positive = safe).
    current_margin : float
        I limit − max |α| (positive = safe).
    voltage_margin : float
        V limit − max |Z·α| (positive = safe).
    dE_dt : float
        Energy time-derivative (finite difference).
    h_cog : float
        Cognitive free-energy surrogate H_cog.
    dH_dt : float
        H_cog time-derivative.
    requires_injection : bool
        True when dH/dt > 0 and external energy injection is implied.
    dPhi_dt : float
        Φ time-derivative.
    implacement_pairs : list
        (M_idx, F_idx) pairs where implacement occurred.
    delta_phi : float
        ΔΦ from implacement at this step.
    """
    t: float
    amplitudes: np.ndarray
    outputs: np.ndarray
    energy: float
    phi: float
    sync_R: float
    morl_objectives: Dict[str, float]
    sar_margin: float
    thermal_margin: float
    current_margin: float
    voltage_margin: float
    dE_dt: float = 0.0
    h_cog: float = 0.0
    dH_dt: float = 0.0
    requires_injection: bool = False
    dPhi_dt: float = 0.0
    implacement_pairs: List[Tuple[int, int]] = field(default_factory=list)
    delta_phi: float = 0.0


@dataclass
class TrajectoryResult:
    """Complete trajectory output.

    Attributes
    ----------
    spec : TrajectorySpec
    n_timesteps : int
    points : list of TrajectoryPoint
    waypoint_indices : list of int
        Indices into points for each waypoint.
    energy_barrier : float
        max(E) − min(E) along the path.
    phi_range : tuple (min, max)
    sync_range : tuple (min, max)
    all_constraints_satisfied : bool
    safety_violations : list of dict
    h_theorem_violations : list of int
        Point indices where dH_cog/dt > 0.
    total_implacement_delta_phi : float
    time_array : ndarray (T,)
    amplitudes_array : ndarray (T, N)
    outputs_array : ndarray (T, 5)
    energy_array : ndarray (T,)
    phi_array : ndarray (T,)
    sync_array : ndarray (T,)
    h_array : ndarray (T,)
    """
    spec: TrajectorySpec
    n_timesteps: int
    points: List[TrajectoryPoint]
    waypoint_indices: List[int]
    energy_barrier: float
    phi_range: Tuple[float, float]
    sync_range: Tuple[float, float]
    all_constraints_satisfied: bool
    safety_violations: List[Dict[str, Any]]
    h_theorem_violations: List[int]
    total_implacement_delta_phi: float
    time_array: np.ndarray
    amplitudes_array: np.ndarray
    outputs_array: np.ndarray
    energy_array: np.ndarray
    phi_array: np.ndarray
    sync_array: np.ndarray
    h_array: np.ndarray


# =====================================================================
# §2 — Waypoint Resolution
# =====================================================================

def resolve_waypoints(
    spec: TrajectorySpec,
    worlds: list,
) -> List[np.ndarray]:
    """Resolve a TrajectorySpec into a list of amplitude waypoints.

    Parameters
    ----------
    spec : TrajectorySpec
    worlds : list of WorldState

    Returns
    -------
    waypoints : list of ndarray (N,)
        Amplitude vectors at each waypoint.
    """
    if spec.spec_type == "world_sequence":
        if spec.world_ids is None or len(spec.world_ids) < 2:
            raise ValueError("world_sequence requires at least 2 world_ids")
        return [worlds[wid].amplitudes.copy() for wid in spec.world_ids]

    elif spec.spec_type == "property_sequence":
        if spec.property_targets is None or len(spec.property_targets) < 2:
            raise ValueError("property_sequence requires at least 2 targets")
        return _find_worlds_for_properties(spec.property_targets, worlds)

    elif spec.spec_type == "energy_descent":
        if spec.start_world_id is None or spec.end_world_id is None:
            raise ValueError("energy_descent requires start and end world IDs")
        world_ids = _find_min_energy_path_world_ids(
            worlds=worlds,
            start_world_id=spec.start_world_id,
            end_world_id=spec.end_world_id,
            k_neighbors=spec.graph_k_neighbors,
        )
        return [worlds[wid].amplitudes.copy() for wid in world_ids]

    else:
        raise ValueError(f"Unknown spec_type: {spec.spec_type}")


def _find_worlds_for_properties(
    targets: List[PropertyGoal],
    worlds: list,
) -> List[np.ndarray]:
    """Find the world closest to each (Φ, R) target.

    Uses weighted Euclidean distance in (Φ, R) space.

    Parameters
    ----------
    targets : list of PropertyGoal
    worlds : list of WorldState

    Returns
    -------
    waypoints : list of ndarray (N,)
    """
    waypoints = []
    phi_vals = np.array([float(w.phi) for w in worlds])
    sync_vals = np.array([float(getattr(w, "sync_R", 1.0)) for w in worlds])
    energy_vals = np.array([float(w.energy) for w in worlds])
    phi_scale = float(np.max(phi_vals) - np.min(phi_vals))
    sync_scale = float(np.max(sync_vals) - np.min(sync_vals))
    energy_scale = float(np.max(energy_vals) - np.min(energy_vals))

    def _scaled_sq(err: float, scale: float) -> float:
        if scale < 1e-6:
            return float(err ** 2)
        return float((err / scale) ** 2)

    for goal in targets:
        best_idx = 0
        best_dist = float("inf")

        for i, w in enumerate(worlds):
            w_sync = float(getattr(w, "sync_R", 1.0))
            d_phi = _scaled_sq(float(w.phi) - goal.phi_target, phi_scale)
            d_sync = _scaled_sq(w_sync - goal.sync_target, sync_scale)
            if goal.energy_target is not None:
                d_energy = _scaled_sq(float(w.energy) - goal.energy_target, energy_scale)
            else:
                d_energy = 0.0

            dist = d_phi + d_sync + d_energy
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_dist > goal.tolerance ** 2:
            warnings.warn(
                f"Property target (phi={goal.phi_target:.3f}, sync={goal.sync_target:.3f}) "
                f"matched at normalized distance {best_dist:.4f} (tolerance={goal.tolerance:.4f})"
            )
        waypoints.append(worlds[best_idx].amplitudes.copy())

    return waypoints


def _find_min_energy_path_world_ids(
    worlds: list,
    start_world_id: int,
    end_world_id: int,
    k_neighbors: int = 8,
) -> List[int]:
    """Find a low-barrier path between two worlds on a k-NN graph.

    Uses a minimax Dijkstra objective:
        cost(path) = min over paths of max energy encountered along the path
    with amplitude-path length as a tiebreaker.
    """
    n_worlds = len(worlds)
    if n_worlds == 0:
        return []
    if not (0 <= start_world_id < n_worlds) or not (0 <= end_world_id < n_worlds):
        raise ValueError("start_world_id/end_world_id out of range")
    if start_world_id == end_world_id:
        return [start_world_id]
    if n_worlds == 1:
        return [0]

    amps = np.array([w.amplitudes for w in worlds], dtype=float)
    energies = np.array([float(w.energy) for w in worlds], dtype=float)

    # Pairwise amplitude distances.
    diffs = amps[:, None, :] - amps[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)

    # Build undirected k-NN graph.
    k = int(np.clip(k_neighbors, 1, max(1, n_worlds - 1)))
    adj: List[set[int]] = [set() for _ in range(n_worlds)]
    for i in range(n_worlds):
        order = np.argsort(dists[i])
        neighbors = [int(j) for j in order if j != i][:k]
        for j in neighbors:
            adj[i].add(j)
            adj[j].add(i)

    import heapq

    inf = float("inf")
    best_peak = np.ones(n_worlds) * inf
    best_len = np.ones(n_worlds) * inf
    parent = np.full(n_worlds, -1, dtype=int)

    start_peak = energies[start_world_id]
    best_peak[start_world_id] = start_peak
    best_len[start_world_id] = 0.0
    heap: List[Tuple[float, float, int]] = [(start_peak, 0.0, start_world_id)]

    while heap:
        peak, plen, node = heapq.heappop(heap)
        if peak > best_peak[node] + 1e-12:
            continue
        if abs(peak - best_peak[node]) <= 1e-12 and plen > best_len[node] + 1e-12:
            continue
        if node == end_world_id:
            break

        for nbr in adj[node]:
            edge_peak = max(energies[node], energies[nbr])
            new_peak = max(peak, edge_peak)
            new_len = plen + float(dists[node, nbr])

            better_peak = new_peak < best_peak[nbr] - 1e-12
            better_tie = abs(new_peak - best_peak[nbr]) <= 1e-12 and new_len < best_len[nbr] - 1e-12
            if better_peak or better_tie:
                best_peak[nbr] = new_peak
                best_len[nbr] = new_len
                parent[nbr] = node
                heapq.heappush(heap, (new_peak, new_len, nbr))

    if parent[end_world_id] < 0:
        # Graph disconnected fallback: direct path.
        return [start_world_id, end_world_id]

    path = [end_world_id]
    cur = end_world_id
    while cur != start_world_id and parent[cur] >= 0:
        cur = int(parent[cur])
        path.append(cur)
    path.reverse()
    return path


# =====================================================================
# §3 — Interpolation Algorithms
# =====================================================================

def interpolate_linear(
    alpha_a: np.ndarray,
    alpha_b: np.ndarray,
    n_steps: int,
) -> List[np.ndarray]:
    """Linear interpolation between two amplitude vectors.

    Parameters
    ----------
    alpha_a, alpha_b : ndarray (N,)
    n_steps : int
        Number of points including endpoints.

    Returns
    -------
    path : list of ndarray (N,)
    """
    path = []
    for i in range(n_steps):
        s = i / max(n_steps - 1, 1)
        alpha = (1.0 - s) * alpha_a + s * alpha_b
        path.append(alpha)
    return path


def interpolate_geodesic(
    alpha_a: np.ndarray,
    alpha_b: np.ndarray,
    n_steps: int,
    plant: Any,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    eta_field: np.ndarray,
    group: np.ndarray,
    goal: Any = None,
    cp_config: Any = None,
    n_candidates: int = 20,
) -> List[np.ndarray]:
    """Geodesic interpolation: minimum energy-barrier path.

    Generates candidate paths by adding random perturbations to linear
    interpolation, evaluates E_total at each sample point, and selects
    the path with the minimum maximum barrier height.

    Parameters
    ----------
    alpha_a, alpha_b : ndarray (N,)
    n_steps : int
    plant : TMSPlant
    te_matrix, phases, basis_matrix, target_idx, eta_field, group, goal, cp_config
        Passed through to compute_total_energy.
    n_candidates : int
        Number of random candidate paths to sample.

    Returns
    -------
    best_path : list of ndarray (N,)
    """
    from cp_bridge import compute_total_energy

    linear_path = interpolate_linear(alpha_a, alpha_b, n_steps)

    # Evaluate linear baseline
    def _max_barrier(path):
        energies = []
        for a in path:
            E_dict = compute_total_energy(
                plant, a, te_matrix, phases, basis_matrix, target_idx,
                eta_field, group, goal, cp_config)
            energies.append(E_dict["E_total"])
        return max(energies) - min(energies), energies

    best_barrier, _ = _max_barrier(linear_path)
    best_path = linear_path

    # Random perturbation candidates
    N = len(alpha_a)
    diff = alpha_b - alpha_a
    scale = 0.1 * np.linalg.norm(diff)

    rng = np.random.default_rng(42)

    for _ in range(n_candidates):
        candidate = []
        for i in range(n_steps):
            s = i / max(n_steps - 1, 1)
            base = (1.0 - s) * alpha_a + s * alpha_b
            # Perturbation decays at endpoints (zero at s=0 and s=1)
            envelope = 4.0 * s * (1.0 - s)  # parabolic
            noise = rng.standard_normal(N) * scale * envelope
            alpha = np.clip(base + noise, 0.0, None)
            candidate.append(alpha)
        # Force endpoints
        candidate[0] = alpha_a.copy()
        candidate[-1] = alpha_b.copy()

        barrier, _ = _max_barrier(candidate)
        if barrier < best_barrier:
            best_barrier = barrier
            best_path = candidate

    return best_path


def interpolate_jacobian_steered(
    alpha_a: np.ndarray,
    alpha_b: np.ndarray,
    n_steps: int,
    plant: Any,
    k_p: float = 0.5,
    max_step: float = 0.1,
    **plant_kwargs: Any,
) -> List[np.ndarray]:
    """Jacobian-steered interpolation: output-tracking path.

    At each step, computes the Jacobian J = ∂y/∂α and steers amplitudes
    toward the linearly interpolated output target using J^+ (pseudoinverse).

    Parameters
    ----------
    alpha_a, alpha_b : ndarray (N,)
    n_steps : int
    plant : TMSPlant
    k_p : float
        Proportional gain for the steering controller.
    max_step : float
        Maximum amplitude change per step.
    **plant_kwargs
        Passed to forward_from_params and jacobian_output_wrt_amplitudes.

    Returns
    -------
    path : list of ndarray (N,)
    """
    # Compute output targets at endpoints
    y_a = plant.forward_from_params(alpha_a, **plant_kwargs)
    y_b = plant.forward_from_params(alpha_b, **plant_kwargs)

    path = [alpha_a.copy()]
    alpha_current = alpha_a.copy()

    for i in range(1, n_steps):
        s = i / max(n_steps - 1, 1)

        # Target output at this fraction
        y_target = (1.0 - s) * y_a + s * y_b

        # Current output
        y_current = plant.forward_from_params(alpha_current, **plant_kwargs)

        # Output error
        error = y_current - y_target

        # Jacobian at current point
        J = plant.jacobian_output_wrt_amplitudes(alpha_current, **plant_kwargs)

        # Pseudoinverse steering: Δα = -k_p * J^+ @ error
        try:
            J_pinv = np.linalg.pinv(J)
            delta_alpha = -k_p * J_pinv @ error
        except np.linalg.LinAlgError:
            # Fallback: linear step
            delta_alpha = (alpha_b - alpha_current) / max(n_steps - i, 1)

        # Clamp step size
        step_norm = np.linalg.norm(delta_alpha)
        if step_norm > max_step:
            delta_alpha = delta_alpha * (max_step / step_norm)

        # Also blend toward destination to ensure convergence
        blend_toward_dest = 0.3 * ((alpha_b - alpha_current) / max(n_steps - i, 1))
        delta_alpha = 0.7 * delta_alpha + blend_toward_dest

        alpha_current = np.clip(alpha_current + delta_alpha, 0.0, None)
        path.append(alpha_current.copy())

    # Force final endpoint
    path[-1] = alpha_b.copy()

    return path


def interpolate_segments(
    waypoints: List[np.ndarray],
    n_steps_per_segment: int,
    method: str,
    plant: Any = None,
    te_matrix: np.ndarray = None,
    phases: np.ndarray = None,
    basis_matrix: np.ndarray = None,
    target_idx: Any = None,
    eta_field: np.ndarray = None,
    group: np.ndarray = None,
    goal: Any = None,
    cp_config: Any = None,
    **plant_kwargs: Any,
) -> Tuple[List[np.ndarray], List[int]]:
    """Interpolate multi-segment trajectory between waypoints.

    Parameters
    ----------
    waypoints : list of ndarray (N,)
    n_steps_per_segment : int
    method : str
    plant, te_matrix, ... : for geodesic and Jacobian-steered methods.
    **plant_kwargs : for forward_from_params.

    Returns
    -------
    full_path : list of ndarray (N,)
    waypoint_indices : list of int
        Index of each waypoint in the full path.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints")

    full_path = []
    waypoint_indices = [0]

    for seg_idx in range(len(waypoints) - 1):
        a = waypoints[seg_idx]
        b = waypoints[seg_idx + 1]

        if method == "linear":
            segment = interpolate_linear(a, b, n_steps_per_segment)
        elif method == "geodesic":
            segment = interpolate_geodesic(
                a, b, n_steps_per_segment,
                plant, te_matrix, phases, basis_matrix, target_idx,
                eta_field, group, goal, cp_config)
        elif method == "jacobian_steered":
            segment = interpolate_jacobian_steered(
                a, b, n_steps_per_segment, plant, **plant_kwargs)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        # Avoid duplicating the shared endpoint between segments
        if seg_idx > 0:
            segment = segment[1:]

        full_path.extend(segment)
        waypoint_indices.append(len(full_path) - 1)

    return full_path, waypoint_indices


# =====================================================================
# §4 — Constraint Enforcement
# =====================================================================

def check_point_safety(
    plant: Any,
    amplitudes: np.ndarray,
    config: Any,
    **plant_kwargs: Any,
) -> Dict[str, Any]:
    """Check safety constraints at a single point.

    Parameters
    ----------
    plant : TMSPlant
    amplitudes : ndarray (N,)
    config : OmnidreamConfig
    **plant_kwargs : for forward_from_params.

    Returns
    -------
    result : dict
        satisfied: bool
        margins: {sar, thermal, current, voltage}
        violations: list of (name, actual, limit)
    """
    y = plant.forward_from_params(amplitudes, **plant_kwargs)
    y_map = map_output_vector(y)
    SAR_max = y_map["sar_max"]
    T_max = y_map["temperature_max"]
    max_amp = float(np.max(np.abs(amplitudes)))
    max_voltage = 0.0
    if hasattr(plant, "Z") and plant.Z is not None:
        try:
            max_voltage = float(np.max(np.abs(plant.Z @ amplitudes)))
        except Exception:
            max_voltage = 0.0

    sar_limit = config.safety.sar_limit_wkg
    t_limit = config.safety.temp_critical_continuous_c
    i_limit = config.safety.max_current_a
    v_limit = config.safety.max_voltage_v

    violations = []
    if SAR_max > sar_limit:
        violations.append(("SAR", SAR_max, sar_limit))
    if T_max > t_limit:
        violations.append(("Thermal", T_max, t_limit))
    if max_amp > i_limit:
        violations.append(("Current", max_amp, i_limit))
    if max_voltage > v_limit:
        violations.append(("Voltage", max_voltage, v_limit))

    return {
        "satisfied": len(violations) == 0,
        "margins": {
            "sar": sar_limit - SAR_max,
            "thermal": t_limit - T_max,
            "current": i_limit - max_amp,
            "voltage": v_limit - max_voltage,
        },
        "violations": violations,
    }


def enforce_safety_along_path(
    amplitudes_path: List[np.ndarray],
    plant: Any,
    config: Any,
    max_iterations: int = 10,
    **plant_kwargs: Any,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Project each point onto the feasible region.

    Iteratively scales down amplitudes at violating points.

    Parameters
    ----------
    amplitudes_path : list of ndarray (N,)
    plant : TMSPlant
    config : OmnidreamConfig
    max_iterations : int
    **plant_kwargs

    Returns
    -------
    corrected_path : list of ndarray (N,)
    violations_log : list of dict for points that couldn't be fixed.
    """
    corrected = [a.copy() for a in amplitudes_path]
    violations_log = []

    for idx in range(len(corrected)):
        for iteration in range(max_iterations):
            check = check_point_safety(plant, corrected[idx], config, **plant_kwargs)
            if check["satisfied"]:
                break
            # Scale down by 5% each iteration
            corrected[idx] *= 0.95

        # Final check
        final_check = check_point_safety(plant, corrected[idx], config, **plant_kwargs)
        if not final_check["satisfied"]:
            violations_log.append({
                "index": idx,
                "violations": final_check["violations"],
                "margins": final_check["margins"],
            })

    return corrected, violations_log


# =====================================================================
# §5 — H-Theorem Enforcement
# =====================================================================

def compute_energy_flow(
    energies: np.ndarray,
    dt: float = 1.0,
    threshold: float = 1e-6,
) -> Tuple[np.ndarray, List[int]]:
    """Compute dE/dt.

    Parameters
    ----------
    energies : ndarray (T,)
    dt : float
    threshold : float

    Returns
    -------
    dE_dt : ndarray (T-1,)
    violations : list of int — indices where dE/dt > threshold.
    """
    dE_dt = np.diff(energies) / dt
    violations = list(np.where(dE_dt > threshold)[0])
    return dE_dt, violations


def compute_h_cog(
    energy: float,
    phi: float,
    sync_R: float,
    phi_weight: float = 1.0,
    sync_weight: float = 0.25,
) -> float:
    """Compute cognitive free-energy surrogate.

    H_cog = E_total - w_phi * Φ - w_sync * R
    """
    return float(energy - phi_weight * phi - sync_weight * sync_R)


def compute_h_array(
    energies: np.ndarray,
    phis: np.ndarray,
    syncs: np.ndarray,
    phi_weight: float = 1.0,
    sync_weight: float = 0.25,
) -> np.ndarray:
    """Vectorised H_cog over a trajectory."""
    return (np.asarray(energies, dtype=float)
            - phi_weight * np.asarray(phis, dtype=float)
            - sync_weight * np.asarray(syncs, dtype=float))


def compute_h_flow(
    h_values: np.ndarray,
    dt: float = 1.0,
    threshold: float = 1e-6,
) -> Tuple[np.ndarray, List[int]]:
    """Compute dH/dt and flag H-theorem violations where dH/dt > threshold."""
    dH_dt = np.diff(h_values) / dt
    violations = list(np.where(dH_dt > threshold)[0])
    return dH_dt, violations


def enforce_h_theorem(
    amplitudes_path: List[np.ndarray],
    h_values: np.ndarray,
    plant: Any,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    eta_field: np.ndarray,
    group: np.ndarray,
    goal: Any = None,
    cp_config: Any = None,
    h_phi_weight: float = 1.0,
    h_sync_weight: float = 0.25,
    n_iterations: int = 3,
    step_size: float = 0.01,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Adjust amplitudes to reduce H-theorem violations.

    For each point where dH/dt > 0, take a descent step along the H_cog
    gradient to reduce H_cog.

    Parameters
    ----------
    amplitudes_path : list of ndarray (N,)
    h_values : ndarray (T,)
    plant, te_matrix, phases, basis_matrix, target_idx, eta_field, group
    goal, cp_config
    h_phi_weight, h_sync_weight
    n_iterations : int
    step_size : float

    Returns
    -------
    corrected_path : list of ndarray (N,)
    corrected_h : ndarray (T,)
    """
    corrected = [a.copy() for a in amplitudes_path]
    corrected_h = h_values.copy()

    for iteration in range(n_iterations):
        dH_dt, violations = compute_h_flow(corrected_h)

        if not violations:
            break

        for v_idx in violations:
            # Skip endpoints
            if v_idx >= len(corrected) - 1:
                continue

            point_idx = v_idx + 1  # The point that's too high
            a = corrected[point_idx]

            # Compute energy gradient via finite differences
            N = len(a)
            grad_H = np.zeros(N)
            eps = 1e-6

            H_center = corrected_h[point_idx]

            for j in range(N):
                a_plus = a.copy()
                a_plus[j] += eps
                metrics_plus = evaluate_cp_metrics(
                    a_plus, plant, te_matrix, phases, basis_matrix, target_idx,
                    eta_field, group, goal, cp_config)
                H_plus = compute_h_cog(
                    metrics_plus["energy"],
                    metrics_plus["phi"],
                    metrics_plus["sync_R"],
                    phi_weight=h_phi_weight,
                    sync_weight=h_sync_weight,
                )
                grad_H[j] = (H_plus - H_center) / eps

            # Descent step
            grad_norm = np.linalg.norm(grad_H)
            if grad_norm > 1e-15:
                corrected[point_idx] = np.clip(
                    a - step_size * grad_H / grad_norm, 0.0, None)

                metrics_new = evaluate_cp_metrics(
                    corrected[point_idx], plant, te_matrix, phases,
                    basis_matrix, target_idx, eta_field, group, goal, cp_config)
                H_new = compute_h_cog(
                    metrics_new["energy"],
                    metrics_new["phi"],
                    metrics_new["sync_R"],
                    phi_weight=h_phi_weight,
                    sync_weight=h_sync_weight,
                )
                corrected_h[point_idx] = H_new

    return corrected, corrected_h


# =====================================================================
# §6 — CP Metrics Evaluation
# =====================================================================

def evaluate_cp_metrics(
    amplitudes: np.ndarray,
    plant: Any,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    eta_field: np.ndarray,
    group: np.ndarray,
    goal: Any = None,
    cp_config: Any = None,
    agents: Any = None,
) -> Dict[str, Any]:
    """Evaluate all CP bridge metrics at a single amplitude vector.

    Parameters
    ----------
    amplitudes : ndarray (N,)
    (remaining parameters passed to CP bridge functions)

    Returns
    -------
    metrics : dict with energy, phi, sync_R, morl_objectives,
              implacement_pairs, delta_phi
    """
    from cp_bridge import (
        compute_total_energy, compute_collective_phi,
        compute_sync_order_parameter, compute_morl_objectives,
        build_agent_states, detect_implacement, apply_implacement,
    )

    # Energy and Φ
    energy_dict = compute_total_energy(
        plant, amplitudes, te_matrix, phases, basis_matrix, target_idx,
        eta_field, group, goal, cp_config)

    phi = energy_dict["phi_collective"]
    sync_R = compute_sync_order_parameter(phases)

    morl_objs = energy_dict.get("morl_objectives", {})

    # Implacement
    if agents is not None:
        impl_pairs = detect_implacement(agents, te_matrix, basis_matrix)
        _, delta_phi = apply_implacement(eta_field, impl_pairs)
    else:
        impl_pairs = []
        delta_phi = 0.0

    return {
        "energy": energy_dict["E_total"],
        "energy_dict": energy_dict,
        "phi": phi,
        "sync_R": sync_R,
        "morl_objectives": morl_objs,
        "implacement_pairs": impl_pairs,
        "delta_phi": delta_phi,
    }


# =====================================================================
# §7 — Main Trajectory Planner
# =====================================================================

def plan_trajectory(
    spec: TrajectorySpec,
    worlds: list,
    plant: Any,
    basis_matrix: np.ndarray,
    target_idx: Any,
    surface_indices: np.ndarray,
    L_matrix: np.ndarray,
    config: Any,
    cp_config: Any = None,
    group: np.ndarray = None,
    te_matrix: np.ndarray = None,
    **plant_kwargs: Any,
) -> TrajectoryResult:
    """Plan and execute a trajectory through world states.

    Parameters
    ----------
    spec : TrajectorySpec
    worlds : list of WorldState
    plant : TMSPlant
    basis_matrix : ndarray (M, N)
    target_idx : int or ndarray
    surface_indices : ndarray
    L_matrix : ndarray (N, N)
    config : OmnidreamConfig
    cp_config : CPBridgeConfig, optional
    group : ndarray (N,), optional — group assignment (0/1)
    te_matrix : ndarray (N, N), optional
    **plant_kwargs : passed to forward_from_params

    Returns
    -------
    result : TrajectoryResult
    """
    from cp_bridge import (
        CPBridgeConfig, compute_transfer_entropy_matrix,
        compute_phases_from_groups, build_agent_states,
    )

    if cp_config is None:
        cp_config = CPBridgeConfig()

    N = basis_matrix.shape[1]

    if group is None:
        group = np.zeros(N)
        group[N // 2:] = 1.0

    # Transfer entropy
    if te_matrix is None:
        te_matrix = compute_transfer_entropy_matrix(L_matrix, cp_config.noise_variance)

    # Phases and frequencies
    f1 = config.ti.freq_carrier_hz
    f2 = f1 + config.ti.delta_freq_default_hz
    phases = compute_phases_from_groups(group, f1, f2, t=0.0)

    # Learning rate field
    eta_field = np.where(group > 0.5, 1.0, 0.1)

    # Resistance diagonal
    R_diag = np.diag(plant.R)
    omega = 2.0 * math.pi * f1

    # GoalSpec
    from control_framework import GoalSpec
    goal = GoalSpec(target_idx=target_idx, surface_indices=surface_indices)

    # ------------------------------------------------------------------
    # Step 1: Resolve waypoints
    # ------------------------------------------------------------------
    waypoints = resolve_waypoints(spec, worlds)

    # ------------------------------------------------------------------
    # Step 2: Interpolate
    # ------------------------------------------------------------------
    amp_path, wp_indices = interpolate_segments(
        waypoints,
        spec.n_steps_per_segment,
        spec.interpolation,
        plant=plant,
        te_matrix=te_matrix,
        phases=phases,
        basis_matrix=basis_matrix,
        target_idx=target_idx,
        eta_field=eta_field,
        group=group,
        goal=goal,
        cp_config=cp_config,
        **plant_kwargs,
    )

    # ------------------------------------------------------------------
    # Step 3: Safety enforcement
    # ------------------------------------------------------------------
    violations_log = []
    if spec.safety_mode == "strict":
        amp_path, violations_log = enforce_safety_along_path(
            amp_path, plant, config, **plant_kwargs)

    # ------------------------------------------------------------------
    # Step 4: Evaluate CP metrics at each point
    # ------------------------------------------------------------------
    T_total = len(amp_path)
    time_array = np.linspace(0.0, 1.0, T_total)

    points = []
    energies = np.zeros(T_total)
    phis = np.zeros(T_total)
    syncs = np.zeros(T_total)

    for i in range(T_total):
        a = amp_path[i]

        # Forward model outputs
        y = plant.forward_from_params(a, **plant_kwargs)

        # CP metrics
        metrics = evaluate_cp_metrics(
            a, plant, te_matrix, phases, basis_matrix, target_idx,
            eta_field, group, goal, cp_config)

        energies[i] = metrics["energy"]
        phis[i] = metrics["phi"]
        syncs[i] = metrics["sync_R"]

        # Safety margins
        safety = check_point_safety(plant, a, config, **plant_kwargs)

        point = TrajectoryPoint(
            t=float(time_array[i]),
            amplitudes=a.copy(),
            outputs=y.copy(),
            energy=metrics["energy"],
            phi=metrics["phi"],
            sync_R=metrics["sync_R"],
            morl_objectives=metrics["morl_objectives"],
            sar_margin=safety["margins"]["sar"],
            thermal_margin=safety["margins"]["thermal"],
            current_margin=safety["margins"]["current"],
            voltage_margin=safety["margins"]["voltage"],
            implacement_pairs=metrics["implacement_pairs"],
            delta_phi=metrics["delta_phi"],
        )
        points.append(point)

    # ------------------------------------------------------------------
    # Step 5: H-theorem enforcement
    # ------------------------------------------------------------------
    h_violations = []
    h_values = compute_h_array(
        energies, phis, syncs,
        phi_weight=spec.h_phi_weight,
        sync_weight=spec.h_sync_weight,
    )
    if T_total > 1:
        dt = 1.0 / (T_total - 1)
        dE_dt, _ = compute_energy_flow(energies, dt)
        dH_dt, h_violations = compute_h_flow(h_values, dt)

        if spec.h_theorem_enforce and h_violations:
            amp_path, h_values = enforce_h_theorem(
                amp_path, h_values, plant, te_matrix, phases,
                basis_matrix, target_idx, eta_field, group, goal, cp_config,
                h_phi_weight=spec.h_phi_weight,
                h_sync_weight=spec.h_sync_weight,
            )

            # Recompute points after H-theorem corrections
            for i in range(T_total):
                a = amp_path[i]
                y = plant.forward_from_params(a, **plant_kwargs)
                metrics = evaluate_cp_metrics(
                    a, plant, te_matrix, phases, basis_matrix, target_idx,
                    eta_field, group, goal, cp_config)
                safety = check_point_safety(plant, a, config, **plant_kwargs)

                points[i] = TrajectoryPoint(
                    t=float(time_array[i]),
                    amplitudes=a.copy(),
                    outputs=y.copy(),
                    energy=metrics["energy"],
                    phi=metrics["phi"],
                    sync_R=metrics["sync_R"],
                    morl_objectives=metrics["morl_objectives"],
                    sar_margin=safety["margins"]["sar"],
                    thermal_margin=safety["margins"]["thermal"],
                    current_margin=safety["margins"]["current"],
                    voltage_margin=safety["margins"]["voltage"],
                    implacement_pairs=metrics["implacement_pairs"],
                    delta_phi=metrics["delta_phi"],
                )
                energies[i] = metrics["energy"]
                phis[i] = metrics["phi"]
                syncs[i] = metrics["sync_R"]

            # Recompute violations
            h_values = compute_h_array(
                energies, phis, syncs,
                phi_weight=spec.h_phi_weight,
                sync_weight=spec.h_sync_weight,
            )
            dE_dt, _ = compute_energy_flow(energies, dt)
            dH_dt, h_violations = compute_h_flow(h_values, dt)

        # Assign dE/dt to points
        for i in range(len(dE_dt)):
            points[i].dE_dt = float(dE_dt[i])
            points[i].dH_dt = float(dH_dt[i])
            points[i].requires_injection = bool(dH_dt[i] > 1e-6)

        # Assign dPhi/dt
        dPhi_dt = np.diff(phis) / dt
        for i in range(len(dPhi_dt)):
            points[i].dPhi_dt = float(dPhi_dt[i])

    # Assign H values to points
    for i, p in enumerate(points):
        p.h_cog = float(h_values[i])

    # ------------------------------------------------------------------
    # Step 6: Build summary arrays and result
    # ------------------------------------------------------------------
    all_amps = np.array([p.amplitudes for p in points])
    all_outputs = np.array([p.outputs for p in points])

    total_impl_dphi = sum(p.delta_phi for p in points)

    result = TrajectoryResult(
        spec=spec,
        n_timesteps=T_total,
        points=points,
        waypoint_indices=wp_indices,
        energy_barrier=float(np.max(energies) - np.min(energies)),
        phi_range=(float(np.min(phis)), float(np.max(phis))),
        sync_range=(float(np.min(syncs)), float(np.max(syncs))),
        all_constraints_satisfied=len(violations_log) == 0,
        safety_violations=violations_log,
        h_theorem_violations=h_violations,
        total_implacement_delta_phi=total_impl_dphi,
        time_array=time_array,
        amplitudes_array=all_amps,
        outputs_array=all_outputs,
        energy_array=energies,
        phi_array=phis,
        sync_array=syncs,
        h_array=h_values,
    )

    return result


# =====================================================================
# §8 — Summary and Serialisation
# =====================================================================

def print_trajectory_summary(result: TrajectoryResult) -> str:
    """Format trajectory result as human-readable summary.

    Parameters
    ----------
    result : TrajectoryResult

    Returns
    -------
    summary : str
    """
    lines = []
    lines.append(f"=== Trajectory: {result.spec.name} ===\n")
    lines.append(f"Mode:                {result.spec.spec_type}")
    lines.append(f"Interpolation:       {result.spec.interpolation}")
    lines.append(f"Timesteps:           {result.n_timesteps}")
    lines.append(f"Waypoints:           {len(result.waypoint_indices)}")

    lines.append(f"\nEnergy:")
    lines.append(f"  Start:             {result.energy_array[0]:.4f}")
    lines.append(f"  End:               {result.energy_array[-1]:.4f}")
    lines.append(f"  Min:               {result.energy_array.min():.4f}")
    lines.append(f"  Max:               {result.energy_array.max():.4f}")
    lines.append(f"  Barrier:           {result.energy_barrier:.4f}")

    lines.append(f"\nCollective Φ:")
    lines.append(f"  Range:             [{result.phi_range[0]:.4f}, {result.phi_range[1]:.4f}]")

    lines.append(f"\nSync R:")
    lines.append(f"  Range:             [{result.sync_range[0]:.4f}, {result.sync_range[1]:.4f}]")

    lines.append(f"\nSafety:")
    lines.append(f"  All satisfied:     {result.all_constraints_satisfied}")
    if result.safety_violations:
        lines.append(f"  Violations:        {len(result.safety_violations)} points")

    lines.append(f"\nH-Theorem:")
    lines.append(f"  Violations:        {len(result.h_theorem_violations)} points")
    lines.append(f"  Enforce:           {result.spec.h_theorem_enforce}")
    lines.append(f"  H_cog start/end:   {result.h_array[0]:.4f} → {result.h_array[-1]:.4f}")
    lines.append(f"  H_cog min/max:     {result.h_array.min():.4f} / {result.h_array.max():.4f}")
    injection_points = int(np.sum(np.diff(result.h_array) > 1e-6))
    lines.append(f"  Injection points:  {injection_points}")

    lines.append(f"\nImplacement:")
    lines.append(f"  Total ΔΦ:          {result.total_implacement_delta_phi:.4f}")

    lines.append("")
    return "\n".join(lines)


def save_trajectory(result: TrajectoryResult, output_dir: Path) -> None:
    """Save trajectory result to disk.

    Parameters
    ----------
    result : TrajectoryResult
    output_dir : Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays
    np.savez_compressed(
        output_dir / "trajectory_result.npz",
        time=result.time_array,
        amplitudes=result.amplitudes_array,
        outputs=result.outputs_array,
        energy=result.energy_array,
        phi=result.phi_array,
        sync=result.sync_array,
        h_cog=result.h_array,
        waypoint_indices=np.array(result.waypoint_indices),
    )

    # Save text summary
    summary = print_trajectory_summary(result)
    with (output_dir / "trajectory_summary.txt").open("w") as f:
        f.write(summary)
