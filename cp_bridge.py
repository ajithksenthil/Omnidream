"""Computational Psychodynamics bridge for the Omnidream TMS array.

Maps the TMS control surface onto the CP/DAI/4D-NCA/MORL framework:

    - Each coil → a DAI agent (MB¹) performing Active Inference
    - Mutual inductance → transfer entropy (directed information flow)
    - Helmet array → group Markov blanket (MB²) with MORL objectives
    - Full system → 4D NCA energy functional E_total
    - Pareto front → many-worlds landscape of achievable configurations

This module depends on:
    - config.py (parameters)
    - coupling.py (inductance matrix)
    - control_framework.py (TMSPlant, GoalSpec, build_cost_function)
    - ti_fields.py (TI modulation depth)
    - sensitivity.py (Pareto front)
    - basis_fields.py (synthetic basis generation)

See ``Blueprints/control_surface_bridge.md`` for the full derivation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import OmnidreamConfig, paper_baseline
from control_framework import map_output_vector


# =====================================================================
# §1 — Data Structures
# =====================================================================

@dataclass
class AgentState:
    """MB¹ agent state for a single TMS coil.

    Attributes
    ----------
    coil_idx : int
        Index into the N-coil array.
    amplitude : float
        α_i — the active state (how hard the coil drives).
    current : float
        I_i — the internal state (actual current flowing).
    back_emf : float
        Sensory state: back-EMF induced by coupled coils (V).
    free_energy : float
        F^i — the agent's variational free energy.
    eta : float
        Learning rate / plasticity parameter.
    mf_label : str
        "M" (masculine / Group 0) or "F" (feminine / Group 1).
    phi_individual : float
        Φ^i — individual integration contribution.
    """
    coil_idx: int
    amplitude: float
    current: float
    back_emf: float
    free_energy: float
    eta: float
    mf_label: str
    phi_individual: float = 0.0


@dataclass
class GroupState:
    """MB² group state for the helmet array.

    Attributes
    ----------
    agents : list of AgentState
    collective_phi : float
        Φ^{collective} — integrated information of the array.
    sync_order_parameter : float
        R — Kuramoto sync parameter in [0, 1].
    group_free_energy : float
        F² — the group's variational free energy.
    morl_objectives : dict
        {J_phi, J_arch, J_sync, J_task} mapped from TMS outputs.
    """
    agents: List[AgentState]
    collective_phi: float
    sync_order_parameter: float
    group_free_energy: float
    morl_objectives: Dict[str, float]


@dataclass
class WorldState:
    """A single 'world' — a Pareto-optimal TMS configuration.

    Attributes
    ----------
    amplitudes : ndarray (N,)
    outputs : ndarray (5,) — [target_metric, surface_metric, membrane_metric, sar_max, temperature_max]
    energy : float — E_total at this configuration
    phi : float — Φ^{collective} at this configuration
    probability : float — Boltzmann weight
    world_id : int
    sync_R : float — weighted synchronisation order parameter in [0, 1]
    """
    amplitudes: np.ndarray
    outputs: np.ndarray
    energy: float
    phi: float
    probability: float
    world_id: int
    sync_R: float = 1.0


@dataclass
class CPBridgeConfig:
    """Configuration for the CP bridge mapping.

    Energy functional weights corresponding to the 4D NCA terms.
    """
    lambda_nca: float = 1.0
    lambda_mf: float = 0.5
    lambda_arch: float = 1.0
    lambda_phi: float = 2.0
    lambda_couple: float = 0.5
    lambda_morl: float = 1.0
    temperature: float = 1.0       # Boltzmann temperature for world probabilities
    phi_min: float = 0.01          # Consciousness threshold
    eta_decay_rate: float = 0.1    # Implacement decay for η
    noise_variance: float = 1e-6   # Noise floor for transfer entropy


# =====================================================================
# §2 — Transfer Entropy from Mutual Inductance
# =====================================================================

def compute_transfer_entropy_matrix(
    L_matrix: np.ndarray,
    noise_variance: float = 1e-6,
) -> np.ndarray:
    """Map mutual inductance matrix to transfer entropy matrix.

    From control_surface_bridge.md §2.1:
        T_{i→j} = |M_{ij}|² / (L_i · σ²_noise)

    Parameters
    ----------
    L_matrix : ndarray (N, N)
        Inductance matrix with self-inductances on diagonal and
        mutual inductances off-diagonal.
    noise_variance : float
        Noise floor σ² for normalisation.

    Returns
    -------
    T : ndarray (N, N)
        Transfer entropy matrix.  T[i, j] = T_{i→j}.
        Diagonal is zero.
    """
    N = L_matrix.shape[0]
    T = np.zeros((N, N))

    diag = np.diag(L_matrix)  # self-inductances L_i

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            M_ij = L_matrix[i, j]  # mutual inductance
            L_i = max(diag[i], 1e-15)  # prevent division by zero
            T[i, j] = abs(M_ij) ** 2 / (L_i * noise_variance)

    return T


def compute_attachment_matrix(
    te_matrix: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """Binary attachment: Attach(i, j) iff T_{i→j} > threshold.

    Parameters
    ----------
    te_matrix : ndarray (N, N)
        Transfer entropy matrix.
    threshold : float

    Returns
    -------
    A : ndarray (N, N) of bool
    """
    return te_matrix > threshold


def compute_mutual_attachment(te_matrix: np.ndarray) -> np.ndarray:
    """Symmetric mutual attachment: min(T_{i→j}, T_{j→i}).

    Parameters
    ----------
    te_matrix : ndarray (N, N)

    Returns
    -------
    MA : ndarray (N, N)
        Symmetric matrix.
    """
    return np.minimum(te_matrix, te_matrix.T)


# =====================================================================
# §3 — Collective Φ Computation
# =====================================================================

def compute_individual_phi(
    basis_matrix: np.ndarray,
    amplitudes: np.ndarray,
    target_idx: Any,
) -> np.ndarray:
    """Compute per-coil integration contribution Φ^i.

    Φ^i = α_i · |E_i(target)| / Σ_j α_j |E_j(target)|

    This measures what fraction of the total target field is contributed
    by coil i, weighted by its amplitude.

    Parameters
    ----------
    basis_matrix : ndarray (M, N)
    amplitudes : ndarray (N,)
    target_idx : int or ndarray

    Returns
    -------
    phi_i : ndarray (N,)
    """
    # Target field strength per coil
    E_target = np.abs(basis_matrix[target_idx])
    if E_target.ndim > 1:
        E_target = E_target.mean(axis=0)  # average over target region

    weighted = np.abs(amplitudes) * E_target
    total = np.sum(weighted)

    if total < 1e-30:
        return np.zeros(len(amplitudes))

    return weighted / total


def compute_sync_coupling_phi(
    te_matrix: np.ndarray,
    phases: np.ndarray,
) -> float:
    """Compute the synchronisation-dependent coupling Φ.

    Φ_{sync} = Σ_{i<j} min(T_{i→j}, T_{j→i}) · cos(φ_i - φ_j)

    Parameters
    ----------
    te_matrix : ndarray (N, N)
    phases : ndarray (N,)

    Returns
    -------
    phi_sync : float
    """
    N = len(phases)
    phi_sync = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            ma = min(te_matrix[i, j], te_matrix[j, i])
            phi_sync += ma * math.cos(phases[i] - phases[j])

    return phi_sync


def compute_collective_phi(
    basis_matrix: np.ndarray,
    amplitudes: np.ndarray,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    target_idx: Any,
) -> float:
    """Compute collective integrated information Φ^{collective}.

    Φ^{coll} = Σ_i Φ^i + Σ_{i<j} Φ^{ij}_{sync}

    Parameters
    ----------
    basis_matrix : ndarray (M, N)
    amplitudes : ndarray (N,)
    te_matrix : ndarray (N, N)
    phases : ndarray (N,)
    target_idx : int or ndarray

    Returns
    -------
    phi_coll : float
    """
    phi_individual = compute_individual_phi(basis_matrix, amplitudes, target_idx)
    phi_ind_sum = float(np.sum(phi_individual))
    phi_sync = compute_sync_coupling_phi(te_matrix, phases)

    return phi_ind_sum + phi_sync


def compute_phases_from_groups(
    group_assignment: np.ndarray,
    freq1: float,
    freq2: float,
    t: float = 0.0,
) -> np.ndarray:
    """Compute M-F wave phases from TI group assignment.

    Group 0 coils oscillate at f₁, Group 1 at f₂.
    Phase φ_i = 2π · f_{g(i)} · t.

    Parameters
    ----------
    group_assignment : ndarray (N,) — 0.0 or 1.0
    freq1, freq2 : float — carrier frequencies (Hz)
    t : float — time point (s)

    Returns
    -------
    phases : ndarray (N,)
    """
    freqs = np.where(group_assignment > 0.5, freq2, freq1)
    return 2.0 * math.pi * freqs * t


# =====================================================================
# §4 — Agent-Level Free Energy
# =====================================================================

def compute_agent_free_energy(
    coil_idx: int,
    amplitudes: np.ndarray,
    L_matrix: np.ndarray,
    R_diag: np.ndarray,
    omega: float,
) -> float:
    """Compute variational free energy for a single coil-agent.

    F^i = ½ L_i I_i² + Σ_{j≠i} M_{ij} I_i I_j + ½ R_i I_i² / ω

    Parameters
    ----------
    coil_idx : int
    amplitudes : ndarray (N,) — used as currents (quasi-static)
    L_matrix : ndarray (N, N)
    R_diag : ndarray (N,) — diagonal resistances
    omega : float — angular frequency 2πf

    Returns
    -------
    F_i : float
    """
    I_i = amplitudes[coil_idx]
    L_i = L_matrix[coil_idx, coil_idx]

    # Self-inductance (stored energy)
    E_stored = 0.5 * L_i * I_i ** 2

    # Mutual coupling energy
    E_coupled = 0.0
    for j in range(len(amplitudes)):
        if j != coil_idx:
            E_coupled += L_matrix[coil_idx, j] * I_i * amplitudes[j]

    # Dissipation (per cycle)
    R_i = R_diag[coil_idx]
    omega_safe = max(omega, 1e-10)
    E_dissipated = 0.5 * R_i * I_i ** 2 / omega_safe

    return E_stored + E_coupled + E_dissipated


def compute_back_emf(
    coil_idx: int,
    amplitudes: np.ndarray,
    L_matrix: np.ndarray,
    omega: float,
) -> float:
    """Compute back-EMF on coil i from all other coils.

    V_back_i = ω · Σ_{j≠i} M_{ij} I_j

    Parameters
    ----------
    coil_idx : int
    amplitudes : ndarray (N,)
    L_matrix : ndarray (N, N)
    omega : float

    Returns
    -------
    V_back : float
    """
    v = 0.0
    for j in range(len(amplitudes)):
        if j != coil_idx:
            v += L_matrix[coil_idx, j] * amplitudes[j]
    return omega * abs(v)


def build_agent_states(
    amplitudes: np.ndarray,
    L_matrix: np.ndarray,
    R_diag: np.ndarray,
    group: np.ndarray,
    omega: float,
    basis_matrix: Optional[np.ndarray] = None,
    target_idx: Any = 0,
    eta_field: Optional[np.ndarray] = None,
) -> List[AgentState]:
    """Construct AgentState for each coil.

    Parameters
    ----------
    amplitudes : ndarray (N,)
    L_matrix : ndarray (N, N)
    R_diag : ndarray (N,)
    group : ndarray (N,) — 0.0 or 1.0
    omega : float
    basis_matrix : ndarray (M, N), optional
    target_idx : int or ndarray, optional
    eta_field : ndarray (N,), optional — learning rates

    Returns
    -------
    agents : list of AgentState
    """
    N = len(amplitudes)

    if eta_field is None:
        eta_field = np.where(group > 0.5, 1.0, 0.1)  # F=1.0, M=0.1

    phi_ind = np.zeros(N)
    if basis_matrix is not None:
        phi_ind = compute_individual_phi(basis_matrix, amplitudes, target_idx)

    agents = []
    for i in range(N):
        agents.append(AgentState(
            coil_idx=i,
            amplitude=float(amplitudes[i]),
            current=float(amplitudes[i]),  # quasi-static: I ≈ α
            back_emf=compute_back_emf(i, amplitudes, L_matrix, omega),
            free_energy=compute_agent_free_energy(i, amplitudes, L_matrix, R_diag, omega),
            eta=float(eta_field[i]),
            mf_label="F" if group[i] > 0.5 else "M",
            phi_individual=float(phi_ind[i]),
        ))

    return agents


# =====================================================================
# §5 — Group Free Energy and MORL
# =====================================================================

def compute_sync_order_parameter(
    phases: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Kuramoto synchronisation order parameter R.

    R = |Σ_i exp(i·φ_i)| / N

    R = 1: perfect synchronisation
    R = 0: random phases

    Parameters
    ----------
    phases : ndarray (N,)
    weights : ndarray (N,), optional
        Optional non-negative weights. If provided, computes a weighted
        order parameter R = |Σ w_i exp(iφ_i)| / Σ w_i.

    Returns
    -------
    R : float in [0, 1]
    """
    N = len(phases)
    if N == 0:
        return 0.0
    z = np.exp(1j * phases)
    if weights is None:
        return float(np.abs(np.mean(z)))
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != N:
        raise ValueError(f"weights length {w.size} must match phases length {N}")
    w = np.maximum(w, 0.0)
    w_sum = float(np.sum(w))
    if w_sum < 1e-15:
        return float(np.abs(np.mean(z)))
    return float(np.abs(np.sum(w * z) / w_sum))


def compute_morl_objectives(
    plant: Any,
    amplitudes: np.ndarray,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    goal: Optional[Any] = None,
    cp_config: Optional[CPBridgeConfig] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """Compute MORL objective vector J = [J_phi, J_arch, J_sync, J_task].

    Parameters
    ----------
    plant : TMSPlant
    amplitudes : ndarray (N,)
    te_matrix : ndarray (N, N)
    phases : ndarray (N,)
    basis_matrix : ndarray (M, N)
    target_idx : int or ndarray
    goal : GoalSpec, optional
    cp_config : CPBridgeConfig, optional

    Returns
    -------
    objectives : dict
    """
    # J_phi: collective Φ (higher = better, so negate for minimisation)
    J_phi = compute_collective_phi(basis_matrix, amplitudes, te_matrix, phases, target_idx)

    # J_arch: distance to goal archetype (lower = better)
    y = plant.forward_from_params(amplitudes, **kwargs)
    y_map = map_output_vector(y)
    target_metric = y_map["target_metric"]
    surface_metric = y_map["surface_metric"]
    if goal is not None:
        J_arch = (target_metric - goal.modulation_target) ** 2 + (surface_metric - goal.modulation_surface_max) ** 2
    else:
        J_arch = surface_metric ** 2  # minimise surface stimulation

    # J_sync: synchronisation order parameter (higher = better)
    J_sync = compute_sync_order_parameter(phases)

    # J_task: task performance = target modulation
    J_task = target_metric  # M_target or V_target

    return {
        "J_phi": J_phi,
        "J_arch": J_arch,
        "J_sync": J_sync,
        "J_task": J_task,
    }


def scalarize_morl(
    objectives: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Scalarise MORL objectives: r_MORL = Σ_k w_k · J_k.

    Parameters
    ----------
    objectives : dict with J_phi, J_arch, J_sync, J_task
    weights : dict, optional (defaults to equal weights)

    Returns
    -------
    r_morl : float
    """
    if weights is None:
        weights = {"J_phi": 1.0, "J_arch": -1.0, "J_sync": 1.0, "J_task": 1.0}

    return sum(weights.get(k, 0.0) * v for k, v in objectives.items())


def compute_group_free_energy(
    plant: Any,
    amplitudes: np.ndarray,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    agents: Optional[List[AgentState]] = None,
    goal: Optional[Any] = None,
    cp_config: Optional[CPBridgeConfig] = None,
    **kwargs: Any,
) -> float:
    """Compute group free energy F².

    F² = cost(α) - λ_Φ · Φ^{coll} + λ_sync · (1 - R) + Σ_i λ¹ · F¹_i

    Parameters
    ----------
    plant : TMSPlant
    amplitudes : ndarray (N,)
    te_matrix : ndarray (N, N)
    phases : ndarray (N,)
    basis_matrix : ndarray (M, N)
    target_idx : int or ndarray
    agents : list of AgentState, optional
    goal : GoalSpec, optional
    cp_config : CPBridgeConfig, optional

    Returns
    -------
    F2 : float
    """
    if cp_config is None:
        cp_config = CPBridgeConfig()

    # Cost function term (goal alignment)
    from control_framework import build_cost_function, StimulationMode, GoalSpec as GS
    if goal is None:
        from control_framework import SystemGoal
        goal = GS()
    cost_fn = build_cost_function(plant, goal)
    cost_term = cost_fn(amplitudes, **kwargs)

    # Φ term
    phi_coll = compute_collective_phi(basis_matrix, amplitudes, te_matrix, phases, target_idx)

    # Sync term
    R = compute_sync_order_parameter(phases)

    # Individual free energies
    agent_fe_sum = 0.0
    if agents is not None:
        agent_fe_sum = sum(a.free_energy for a in agents)

    F2 = (cost_term
          - cp_config.lambda_phi * phi_coll
          + cp_config.lambda_mf * (1.0 - R)
          + 0.01 * agent_fe_sum)

    return F2


# =====================================================================
# §6 — 4D NCA Energy Functional
# =====================================================================

def compute_nca_energy(
    basis_matrix: np.ndarray,
    amplitudes: np.ndarray,
    target_idx: Any,
) -> float:
    """E_NCA: local consistency (basis field superposition residual).

    E_NCA = ||Σ α_i E_i(target) - max(Σ α_i E_i(target))||²

    Since we don't have an explicit E_goal, we measure the residual between
    the actual target field and its own maximum (normalised consistency).

    Parameters
    ----------
    basis_matrix : ndarray (M, N)
    amplitudes : ndarray (N,)
    target_idx : int or ndarray

    Returns
    -------
    E_nca : float ≥ 0
    """
    # Total field at all points
    E_total = basis_matrix @ amplitudes

    # Target field
    E_target_val = float(np.mean(E_total[target_idx]))
    E_max = float(np.max(np.abs(E_total))) if len(E_total) > 0 else 1.0

    if E_max < 1e-30:
        return 0.0

    # Normalised consistency: how concentrated is the field at the target?
    # Perfect focality → E_nca = 0
    residual = 1.0 - (E_target_val / E_max)
    return max(residual ** 2, 0.0)


def compute_mf_energy(
    eta_field: np.ndarray,
    group: np.ndarray,
    lambda_wave: float = 1.0,
) -> float:
    """E_MF: M-F wave structure energy.

    E_MF = Var(η) + λ_wave · ||η - η_target||²

    where η_target = 0.1 for M (Group 0), 1.0 for F (Group 1).

    Parameters
    ----------
    eta_field : ndarray (N,)
    group : ndarray (N,) — 0.0 or 1.0
    lambda_wave : float

    Returns
    -------
    E_mf : float ≥ 0
    """
    eta_target = np.where(group > 0.5, 1.0, 0.1)
    variance_term = float(np.var(eta_field))
    alignment_term = float(np.mean((eta_field - eta_target) ** 2))

    return variance_term + lambda_wave * alignment_term


def compute_arch_energy(
    plant: Any,
    amplitudes: np.ndarray,
    goal: Optional[Any] = None,
    **kwargs: Any,
) -> float:
    """E_arch: archetypal alignment energy.

    E_arch = ||y - y_goal||² (distance to goal configuration)

    Parameters
    ----------
    plant : TMSPlant
    amplitudes : ndarray (N,)
    goal : GoalSpec, optional

    Returns
    -------
    E_arch : float ≥ 0
    """
    y = plant.forward_from_params(amplitudes, **kwargs)
    y_map = map_output_vector(y)
    target_metric = y_map["target_metric"]
    surface_metric = y_map["surface_metric"]

    if goal is not None:
        target_val = goal.modulation_target
        surface_val = goal.modulation_surface_max
    else:
        target_val = 1.0
        surface_val = 0.0

    return float((target_metric - target_val) ** 2 + (surface_metric - surface_val) ** 2)


def compute_phi_energy(collective_phi: float, lambda_phi: float = 2.0) -> float:
    """E_Φ: integration energy (negative — we want to maximise Φ).

    E_Φ = -λ_Φ · Φ^{coll}

    Parameters
    ----------
    collective_phi : float
    lambda_phi : float

    Returns
    -------
    E_phi : float ≤ 0 (when Φ > 0)
    """
    return -lambda_phi * collective_phi


def compute_coupling_energy(
    te_matrix: np.ndarray,
    phases: np.ndarray,
) -> float:
    """E_couple: inter-agent coupling energy.

    E_couple = -Σ_{i<j} T_{ij} · T_{ji} · cos(Δφ_{ij})

    Parameters
    ----------
    te_matrix : ndarray (N, N)
    phases : ndarray (N,)

    Returns
    -------
    E_couple : float ≤ 0 (when coupling is coherent)
    """
    N = len(phases)
    E = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            T_ij = te_matrix[i, j]
            T_ji = te_matrix[j, i]
            cos_dphi = math.cos(phases[i] - phases[j])
            E -= T_ij * T_ji * cos_dphi

    return E


def compute_morl_energy(
    objectives: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    lambda_morl: float = 1.0,
) -> float:
    """E_MORL: multi-objective term.

    E_MORL = -λ · Σ_k w_k · J_k

    Parameters
    ----------
    objectives : dict
    weights : dict, optional
    lambda_morl : float

    Returns
    -------
    E_morl : float
    """
    r = scalarize_morl(objectives, weights)
    return -lambda_morl * r


def compute_total_energy(
    plant: Any,
    amplitudes: np.ndarray,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    eta_field: np.ndarray,
    group: np.ndarray,
    goal: Optional[Any] = None,
    cp_config: Optional[CPBridgeConfig] = None,
    morl_weights: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """Compute the full 4D NCA energy functional.

    E_total = E_NCA + E_MF + E_arch + E_Φ + E_couple + E_MORL

    Parameters
    ----------
    plant : TMSPlant
    amplitudes : ndarray (N,)
    te_matrix : ndarray (N, N)
    phases : ndarray (N,)
    basis_matrix : ndarray (M, N)
    target_idx : int or ndarray
    eta_field : ndarray (N,)
    group : ndarray (N,)
    goal : GoalSpec, optional
    cp_config : CPBridgeConfig, optional
    morl_weights : dict, optional

    Returns
    -------
    energies : dict with E_nca, E_mf, E_arch, E_phi, E_couple, E_morl, E_total
    """
    if cp_config is None:
        cp_config = CPBridgeConfig()

    # Individual terms
    E_nca = cp_config.lambda_nca * compute_nca_energy(basis_matrix, amplitudes, target_idx)
    E_mf = cp_config.lambda_mf * compute_mf_energy(eta_field, group)
    E_arch = cp_config.lambda_arch * compute_arch_energy(plant, amplitudes, goal, **kwargs)

    phi_coll = compute_collective_phi(basis_matrix, amplitudes, te_matrix, phases, target_idx)
    E_phi = compute_phi_energy(phi_coll, cp_config.lambda_phi)

    E_couple = cp_config.lambda_couple * compute_coupling_energy(te_matrix, phases)

    objectives = compute_morl_objectives(
        plant, amplitudes, te_matrix, phases, basis_matrix, target_idx, goal, cp_config, **kwargs)
    E_morl = compute_morl_energy(objectives, morl_weights, cp_config.lambda_morl)

    E_total = E_nca + E_mf + E_arch + E_phi + E_couple + E_morl

    return {
        "E_nca": E_nca,
        "E_mf": E_mf,
        "E_arch": E_arch,
        "E_phi": E_phi,
        "E_couple": E_couple,
        "E_morl": E_morl,
        "E_total": E_total,
        "phi_collective": phi_coll,
        "morl_objectives": objectives,
    }


# =====================================================================
# §7 — Many-Worlds Computation
# =====================================================================

def compute_world_probabilities(
    energies: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Boltzmann distribution over worlds.

    P(world_k) = exp(-E_k / T) / Z

    Parameters
    ----------
    energies : ndarray (K,)
    temperature : float

    Returns
    -------
    probs : ndarray (K,) summing to 1.0
    """
    if temperature <= 0:
        # T → 0: all probability on minimum energy
        probs = np.zeros_like(energies)
        probs[np.argmin(energies)] = 1.0
        return probs

    # Shift for numerical stability
    E_min = np.min(energies)
    log_probs = -(energies - E_min) / temperature
    # Softmax
    log_probs -= np.max(log_probs)  # additional stability
    probs = np.exp(log_probs)
    Z = np.sum(probs)
    if Z < 1e-30:
        return np.ones_like(probs) / len(probs)
    return probs / Z


def pareto_to_worlds(
    pareto_results: Dict[str, np.ndarray],
    plant: Any,
    te_matrix: np.ndarray,
    phases: np.ndarray,
    basis_matrix: np.ndarray,
    target_idx: Any,
    eta_field: np.ndarray,
    group: np.ndarray,
    goal: Optional[Any] = None,
    cp_config: Optional[CPBridgeConfig] = None,
    mode: str = "TI",
    **kwargs: Any,
) -> List[WorldState]:
    """Convert Pareto front points to WorldState objects.

    Each non-dominated point becomes a 'world' with energy, Φ, and
    Boltzmann probability.

    Parameters
    ----------
    pareto_results : dict from sensitivity.compute_pareto_front_ti/nts
    plant : TMSPlant
    te_matrix : ndarray (N, N)
    phases : ndarray (N,)
    basis_matrix : ndarray (M, N)
    target_idx : int or ndarray
    eta_field : ndarray (N,)
    group : ndarray (N,)
    goal : GoalSpec, optional
    cp_config : CPBridgeConfig, optional
    mode : str

    Returns
    -------
    worlds : list of WorldState
    """
    if cp_config is None:
        cp_config = CPBridgeConfig()

    amplitudes_all = pareto_results["amplitudes"]
    is_dominated = pareto_results.get("is_dominated", np.zeros(len(amplitudes_all), dtype=bool))

    # Get target/surface arrays
    if mode.upper() == "TI":
        targets = pareto_results.get("M_target", np.zeros(len(amplitudes_all)))
        surfaces = pareto_results.get("M_surface_max", np.zeros(len(amplitudes_all)))
    else:
        targets = pareto_results.get("V_target", np.zeros(len(amplitudes_all)))
        surfaces = pareto_results.get("V_surface_max", np.zeros(len(amplitudes_all)))

    # Only non-dominated points become worlds
    nd_mask = ~is_dominated
    nd_indices = np.where(nd_mask)[0]

    # Compute energy for each non-dominated point
    energies = []
    phis = []
    syncs = []
    outputs_list = []

    for idx in nd_indices:
        a = amplitudes_all[idx]
        E_dict = compute_total_energy(
            plant, a, te_matrix, phases, basis_matrix, target_idx,
            eta_field, group, goal, cp_config, **kwargs)
        energies.append(E_dict["E_total"])
        phis.append(E_dict["phi_collective"])
        # Weighted synchrony reflects amplitude-biased phase coherence.
        syncs.append(compute_sync_order_parameter(phases, weights=np.abs(a)))
        # Store full plant outputs for this world to preserve executable
        # state information across the full 5-D output contract.
        outputs_list.append(np.asarray(plant.forward_from_params(a, **kwargs), dtype=float))

    if len(energies) == 0:
        return []

    energies_arr = np.array(energies)
    probs = compute_world_probabilities(energies_arr, cp_config.temperature)

    worlds = []
    for k, idx in enumerate(nd_indices):
        worlds.append(WorldState(
            amplitudes=amplitudes_all[idx].copy(),
            outputs=outputs_list[k],
            energy=energies[k],
            phi=phis[k],
            sync_R=float(syncs[k]),
            probability=float(probs[k]),
            world_id=k,
        ))

    return worlds


def compute_world_coherence(worlds: List[WorldState]) -> np.ndarray:
    """Coherence of each world = its Φ value.

    Parameters
    ----------
    worlds : list of WorldState

    Returns
    -------
    coherence : ndarray (K,)
    """
    if not worlds:
        return np.array([])
    return np.array([w.phi for w in worlds])


# =====================================================================
# §8 — Implacement Dynamics
# =====================================================================

def detect_implacement(
    agents: List[AgentState],
    te_matrix: np.ndarray,
    basis_matrix: np.ndarray,
    epsilon: float = 0.5,
    theta: float = 0.1,
) -> List[Tuple[int, int]]:
    """Detect implacement events between M and F coils.

    Implacement(M_i → F_j) ≡ [d(E_i, E_j) < ε] ∧ [T_{M→F} > θ]

    Parameters
    ----------
    agents : list of AgentState
    te_matrix : ndarray (N, N)
    basis_matrix : ndarray (M, N)
    epsilon : float — field overlap threshold
    theta : float — transfer entropy threshold

    Returns
    -------
    pairs : list of (M_idx, F_idx) tuples
    """
    pairs = []

    for a_m in agents:
        if a_m.mf_label != "M":
            continue
        for a_f in agents:
            if a_f.mf_label != "F":
                continue

            i, j = a_m.coil_idx, a_f.coil_idx

            # Field overlap (normalised dot product of basis columns)
            e_i = basis_matrix[:, i]
            e_j = basis_matrix[:, j]
            norm_i = np.linalg.norm(e_i)
            norm_j = np.linalg.norm(e_j)
            if norm_i < 1e-15 or norm_j < 1e-15:
                continue
            overlap = float(np.dot(e_i, e_j) / (norm_i * norm_j))

            # Check conditions
            if overlap > (1.0 - epsilon) and te_matrix[i, j] > theta:
                pairs.append((i, j))

    return pairs


def apply_implacement(
    eta_field: np.ndarray,
    implacement_pairs: List[Tuple[int, int]],
    decay_rate: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """Apply implacement dynamics: reduce η for implaced F nodes.

    Returns updated η field and total ΔΦ.

    Parameters
    ----------
    eta_field : ndarray (N,)
    implacement_pairs : list of (M_idx, F_idx)
    decay_rate : float

    Returns
    -------
    eta_new : ndarray (N,)
    delta_phi : float — total Φ growth
    """
    eta_new = eta_field.copy()
    delta_phi = 0.0

    for m_idx, f_idx in implacement_pairs:
        eta_m = eta_field[m_idx]
        eta_f = eta_field[f_idx]

        if eta_m > 0 and eta_f > eta_m:
            # Implacement: reduce F's plasticity
            eta_new[f_idx] *= (1.0 - decay_rate)

            # ΔΦ = T_{M→F} · |1 - η_F/η_M| · κ  (simplified: κ=1)
            delta_phi += abs(1.0 - eta_f / eta_m)

    return eta_new, delta_phi


# =====================================================================
# §9 — Comprehensive Analysis Entry Point
# =====================================================================

def run_cp_bridge_analysis(
    basis_matrix: np.ndarray,
    target_idx: Any,
    surface_indices: np.ndarray,
    config: Optional[OmnidreamConfig] = None,
    mode: str = "TI",
    pareto_results: Optional[Dict[str, np.ndarray]] = None,
    positions_mm: Optional[np.ndarray] = None,
    cp_config: Optional[CPBridgeConfig] = None,
) -> Dict[str, Any]:
    """Run the full CP bridge analysis.

    Parameters
    ----------
    basis_matrix : ndarray (M, N)
    target_idx : int or ndarray
    surface_indices : ndarray
    config : OmnidreamConfig, optional
    mode : str — "TI" or "NTS"
    pareto_results : dict, optional (from sensitivity analysis)
    positions_mm : ndarray (N, 3), optional
    cp_config : CPBridgeConfig, optional

    Returns
    -------
    results : dict with comprehensive CP bridge metrics
    """
    from control_framework import TMSPlant, StimulationMode, GoalSpec

    if config is None:
        config = paper_baseline()
    if cp_config is None:
        cp_config = CPBridgeConfig()

    N = basis_matrix.shape[1]

    # Build plant
    stim_mode = StimulationMode.TI if mode.upper() == "TI" else StimulationMode.NTS
    plant = TMSPlant(basis_matrix, target_idx, surface_indices,
                     config=config, mode=stim_mode, positions_mm=positions_mm)

    # Reference amplitudes
    amplitudes = np.ones(N) * 0.5

    # Group assignment
    group = np.zeros(N)
    group[N // 2:] = 1.0

    # Frequencies
    f1 = config.ti.freq_carrier_hz
    f2 = f1 + config.ti.delta_freq_default_hz

    # Inductance and transfer entropy
    L_matrix = plant.L
    R_diag = np.diag(plant.R)
    omega = 2.0 * math.pi * f1

    te_matrix = compute_transfer_entropy_matrix(L_matrix, cp_config.noise_variance)
    attachment = compute_attachment_matrix(te_matrix)
    mutual_attach = compute_mutual_attachment(te_matrix)

    # Phases
    phases = compute_phases_from_groups(group, f1, f2, t=0.0)

    # Learning rate field
    eta_field = np.where(group > 0.5, 1.0, 0.1)

    # Agent states
    agents = build_agent_states(amplitudes, L_matrix, R_diag, group, omega,
                                basis_matrix, target_idx, eta_field)

    # Collective Φ
    phi_coll = compute_collective_phi(basis_matrix, amplitudes, te_matrix, phases, target_idx)

    # Sync order parameter
    R_sync = compute_sync_order_parameter(phases)

    # MORL objectives
    goal = GoalSpec(target_idx=target_idx, surface_indices=surface_indices)
    objectives = compute_morl_objectives(
        plant, amplitudes, te_matrix, phases, basis_matrix, target_idx, goal, cp_config)

    # Group free energy
    F2 = compute_group_free_energy(
        plant, amplitudes, te_matrix, phases, basis_matrix, target_idx,
        agents, goal, cp_config)

    # 4D NCA energy
    energy_dict = compute_total_energy(
        plant, amplitudes, te_matrix, phases, basis_matrix, target_idx,
        eta_field, group, goal, cp_config)

    # Implacement detection
    impl_pairs = detect_implacement(agents, te_matrix, basis_matrix)
    eta_new, delta_phi = apply_implacement(eta_field, impl_pairs, cp_config.eta_decay_rate)

    # Many-worlds (if Pareto results available)
    worlds = []
    if pareto_results is not None:
        worlds = pareto_to_worlds(
            pareto_results, plant, te_matrix, phases, basis_matrix, target_idx,
            eta_field, group, goal, cp_config, mode=mode)

    # Compile results
    results = {
        "mode": mode,
        "n_coils": N,
        # Transfer entropy
        "te_matrix": te_matrix,
        "attachment_matrix": attachment,
        "mutual_attachment": mutual_attach,
        # Agent states
        "agents": agents,
        "agent_free_energies": np.array([a.free_energy for a in agents]),
        # Collective metrics
        "collective_phi": phi_coll,
        "sync_order_parameter": R_sync,
        "group_free_energy": F2,
        # MORL
        "morl_objectives": objectives,
        # Energy functional
        "energy": energy_dict,
        # M-F dynamics
        "eta_field": eta_field,
        "eta_field_post_implacement": eta_new,
        "implacement_pairs": impl_pairs,
        "delta_phi_implacement": delta_phi,
        # Many worlds
        "worlds": worlds,
        "n_worlds": len(worlds),
    }

    if worlds:
        results["world_probabilities"] = np.array([w.probability for w in worlds])
        results["world_coherences"] = compute_world_coherence(worlds)
        results["world_energies"] = np.array([w.energy for w in worlds])

    return results


def print_cp_bridge_summary(results: Dict[str, Any]) -> str:
    """Format CP bridge results as a human-readable summary.

    Parameters
    ----------
    results : dict from run_cp_bridge_analysis

    Returns
    -------
    summary : str
    """
    lines = []
    lines.append(f"=== CP Bridge Analysis: {results['mode']} mode, "
                 f"{results['n_coils']} coils ===\n")

    # Collective metrics
    lines.append(f"Collective Φ:              {results['collective_phi']:.4f}")
    lines.append(f"Sync order parameter R:    {results['sync_order_parameter']:.4f}")
    lines.append(f"Group free energy F²:      {results['group_free_energy']:.4f}")

    # MORL
    obj = results["morl_objectives"]
    lines.append(f"\nMORL Objectives:")
    for k, v in obj.items():
        lines.append(f"  {k}: {v:.4f}")

    # Energy
    E = results["energy"]
    lines.append(f"\n4D NCA Energy Functional:")
    for k in ["E_nca", "E_mf", "E_arch", "E_phi", "E_couple", "E_morl"]:
        lines.append(f"  {k}: {E[k]:.4f}")
    lines.append(f"  E_total: {E['E_total']:.4f}")

    # Transfer entropy
    te = results["te_matrix"]
    lines.append(f"\nTransfer Entropy:")
    lines.append(f"  Mean (off-diagonal): {np.mean(te[te > 0]):.4e}" if np.any(te > 0) else "  Mean: 0")
    lines.append(f"  Max:                 {np.max(te):.4e}")
    n_attached = int(np.sum(results["attachment_matrix"]))
    lines.append(f"  Attached pairs:      {n_attached}")

    # Agent free energies
    fe = results["agent_free_energies"]
    lines.append(f"\nAgent Free Energies:")
    lines.append(f"  Mean: {np.mean(fe):.4f}   Min: {np.min(fe):.4f}   Max: {np.max(fe):.4f}")

    # Implacement
    lines.append(f"\nImplacement:")
    lines.append(f"  Pairs detected:      {len(results['implacement_pairs'])}")
    lines.append(f"  ΔΦ from implacement: {results['delta_phi_implacement']:.4f}")

    # Many worlds
    lines.append(f"\nMany Worlds:")
    lines.append(f"  Total worlds:        {results['n_worlds']}")
    if results["n_worlds"] > 0:
        probs = results["world_probabilities"]
        cohs = results["world_coherences"]
        lines.append(f"  Top-3 Φ values:      {np.sort(cohs)[-3:][::-1]}")
        lines.append(f"  Prob range:          [{probs.min():.4f}, {probs.max():.4f}]")
        lines.append(f"  Energy range:        [{results['world_energies'].min():.4f}, "
                     f"{results['world_energies'].max():.4f}]")

    lines.append("")
    return "\n".join(lines)
