"""Abstract control framework and TMS-specific plant model for Omnidream.

Implements the control-theoretic interfaces derived in
``Blueprints/control_theory.md``:

    - PlantModel ABC — physics-agnostic forward model with linearization
    - TMSPlant — concrete plant for the Omnidream TMS coil array
    - Controller ABC — action selection interface
    - GoalSpec — objective specification decoupled from physics

The key design principle is **separation of physics from control**: the same
Controller + GoalSpec can drive any PlantModel (TMS, FUS, tDCS, optogenetics)
by swapping only the plant implementation.

This module depends on:
    - config.py (parameters)
    - em_theory.py (inductance, SAR, neural response)
    - ti_fields.py (TI forward model)
    - nts_timing.py (NTS forward model)
    - coupling.py (impedance matrix)
    - basis_fields.py (synthetic basis generation)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config import OmnidreamConfig, paper_baseline


# =====================================================================
# §1 — Enumerations and Data Structures
# =====================================================================

class SystemGoal(Enum):
    """High-level objectives for neural stimulation systems.

    These are modality-agnostic goals that any PlantModel should support.
    """
    FOCAL_DEPTH = auto()           # Maximise stimulus at deep target
    SPATIAL_SELECTIVITY = auto()   # Minimise off-target stimulation
    POWER_EFFICIENCY = auto()      # Minimise total input power
    SPEED = auto()                 # Minimise time-to-effect
    THERMAL_HEADROOM = auto()      # Maximise distance from thermal limits


class StimulationMode(Enum):
    """Supported stimulation paradigms."""
    TI = auto()      # Temporal Interference
    NTS = auto()     # Neural Temporal Summation
    HYBRID = auto()  # Combined TI + NTS


STANDARD_OUTPUT_LABELS = [
    "target_metric",
    "surface_metric",
    "membrane_metric",
    "sar_max",
    "temperature_max",
]


LEGACY_OUTPUT_LABELS = ["M_target", "M_surface_max", "V_m_peak", "SAR_max", "T_max"]


@dataclass
class PlantDimensions:
    """Dimensionality of the state-space model.

    For N coils the canonical dimensions are:
        n_state = 2N + 2  (currents + membrane potential + temperatures + modulation)
        m_input = N        (voltage commands)
        p_output = 5       (M_target, M_surface_max, V_m_peak, SAR_max, T_max)
    """
    n_state: int
    m_input: int
    p_output: int
    n_coils: int

    @classmethod
    def from_n_coils(cls, n_coils: int) -> PlantDimensions:
        """Construct canonical dimensions for an N-coil array."""
        return cls(
            n_state=2 * n_coils + 2,
            m_input=n_coils,
            p_output=5,
            n_coils=n_coils,
        )


@dataclass
class GoalSpec:
    """Specification of what the stimulation system should achieve.

    Decoupled from physics: the same GoalSpec works for TMS, FUS, tDCS, etc.
    The optimizer translates goals into plant-specific actions.

    Parameters
    ----------
    target_coords : ndarray (3,) or None
        Target location in mm (if applicable).
    target_idx : int or ndarray
        Index/indices into the basis matrix for the target region.
    surface_indices : ndarray
        Indices of surface/off-target sample points.
    modulation_target : float
        Desired stimulus magnitude at target (V/m or normalised).
    modulation_surface_max : float
        Maximum tolerable surface stimulation.
    power_budget_w : float
        Total power budget in watts.
    weights : dict mapping SystemGoal → float
        Relative importance of each objective (higher = more important).
    constraints : dict
        Hard constraints: {"SAR_max": 3.2, "T_max": 41.0, "I_max": 5.0, ...}
    """
    target_coords: Optional[np.ndarray] = None
    target_idx: Any = 0
    surface_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    modulation_target: float = 1.0
    modulation_surface_max: float = 0.5
    power_budget_w: float = 10.0
    weights: Dict[SystemGoal, float] = field(default_factory=lambda: {
        SystemGoal.FOCAL_DEPTH: 1.0,
        SystemGoal.SPATIAL_SELECTIVITY: 0.5,
        SystemGoal.POWER_EFFICIENCY: 0.1,
    })
    constraints: Dict[str, float] = field(default_factory=lambda: {
        "SAR_max": 3.2,
        "T_max": 41.0,
        "I_max": 5.0,
        "V_max": 60.0,
    })


def map_output_vector(y: np.ndarray) -> Dict[str, float]:
    """Map a 5-vector plant output to a standardized + backward-compatible dict.

    Canonical keys:
        target_metric, surface_metric, membrane_metric, sar_max, temperature_max

    Legacy aliases are also included to avoid breaking existing code paths:
        M_target/V_target, M_surface_max/V_surface_max, V_m_peak, SAR_max, T_max
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if y_arr.size != 5:
        raise ValueError(f"Expected output vector of length 5, got shape {y_arr.shape}")

    target_metric = float(y_arr[0])
    surface_metric = float(y_arr[1])
    membrane_metric = float(y_arr[2])
    sar_max = float(y_arr[3])
    temperature_max = float(y_arr[4])

    return {
        "target_metric": target_metric,
        "surface_metric": surface_metric,
        "membrane_metric": membrane_metric,
        "sar_max": sar_max,
        "temperature_max": temperature_max,
        # Backward-compatible aliases
        "M_target": target_metric,
        "V_target": target_metric,
        "M_surface_max": surface_metric,
        "V_surface_max": surface_metric,
        "V_m_peak": membrane_metric,
        "SAR_max": sar_max,
        "T_max": temperature_max,
    }


# =====================================================================
# §2 — Abstract Plant Model
# =====================================================================

class PlantModel(ABC):
    """Abstract base class for a physical stimulation plant.

    Any stimulation modality (TMS, FUS, tDCS, optogenetics) implements
    this interface.  The Controller and optimisation infrastructure
    (GA, SAC, MPC) interact only through these methods.

    State-space notation follows control_theory.md §1:
        ẋ = f(x, u, θ)
        y = h(x, u)
    """

    @abstractmethod
    def dims(self) -> PlantDimensions:
        """Return the plant dimensionality."""
        ...

    @abstractmethod
    def forward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute outputs y = h(x, u).

        Parameters
        ----------
        x : ndarray (n_state,)
            Current state vector.
        u : ndarray (m_input,)
            Input/control vector.

        Returns
        -------
        y : ndarray (p_output,)
            Output vector [M_target, M_surface_max, V_m_peak, SAR_max, T_max].
        """
        ...

    @abstractmethod
    def forward_from_params(self, amplitudes: np.ndarray,
                            **kwargs: Any) -> np.ndarray:
        """Compute outputs directly from optimisation parameters.

        This is a convenience method for optimisers that work in parameter
        space (amplitudes, timings, groups) rather than state space.

        Parameters
        ----------
        amplitudes : ndarray (N,)
            Per-coil amplitude weights.
        **kwargs
            Mode-specific parameters (group, freq1, freq2, fire_times, ...).

        Returns
        -------
        y : ndarray (p_output,)
        """
        ...

    @abstractmethod
    def linearize(self, x0: np.ndarray, u0: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Linearize the plant at operating point (x0, u0).

        Returns the state-space matrices (A, B, C, D) such that:
            Δẋ ≈ A·Δx + B·Δu
            Δy ≈ C·Δx + D·Δu

        Parameters
        ----------
        x0 : ndarray (n_state,)
        u0 : ndarray (m_input,)

        Returns
        -------
        A : ndarray (n_state, n_state)
        B : ndarray (n_state, m_input)
        C : ndarray (p_output, n_state)
        D : ndarray (p_output, m_input)
        """
        ...

    @abstractmethod
    def jacobian_output_wrt_input(self, x: np.ndarray, u: np.ndarray
                                   ) -> np.ndarray:
        """Compute J = ∂y/∂u at the current state.

        Parameters
        ----------
        x : ndarray (n_state,)
        u : ndarray (m_input,)

        Returns
        -------
        J : ndarray (p_output, m_input)
        """
        ...

    def controllability_matrix(self, A: np.ndarray, B: np.ndarray
                               ) -> np.ndarray:
        """Compute the controllability matrix 𝒞 = [B, AB, A²B, …, Aⁿ⁻¹B].

        Parameters
        ----------
        A : ndarray (n, n)
        B : ndarray (n, m)

        Returns
        -------
        C_ctrl : ndarray (n, n*m)
        """
        n = A.shape[0]
        m = B.shape[1]
        C_ctrl = np.zeros((n, n * m))
        AB_k = B.copy()
        for k in range(n):
            C_ctrl[:, k * m:(k + 1) * m] = AB_k
            AB_k = A @ AB_k
        return C_ctrl

    def observability_matrix(self, A: np.ndarray, C: np.ndarray
                             ) -> np.ndarray:
        """Compute the observability matrix 𝒪 = [C; CA; CA²; …; CAⁿ⁻¹].

        Parameters
        ----------
        A : ndarray (n, n)
        C : ndarray (p, n)

        Returns
        -------
        O_obs : ndarray (n*p, n)
        """
        n = A.shape[0]
        p = C.shape[0]
        O_obs = np.zeros((n * p, n))
        CA_k = C.copy()
        for k in range(n):
            O_obs[k * p:(k + 1) * p, :] = CA_k
            CA_k = CA_k @ A
        return O_obs

    def controllability_rank(self, A: np.ndarray, B: np.ndarray,
                             tol: float = 1e-10) -> int:
        """Rank of the controllability matrix."""
        C_ctrl = self.controllability_matrix(A, B)
        return int(np.linalg.matrix_rank(C_ctrl, tol=tol))

    def observability_rank(self, A: np.ndarray, C: np.ndarray,
                           tol: float = 1e-10) -> int:
        """Rank of the observability matrix."""
        O_obs = self.observability_matrix(A, C)
        return int(np.linalg.matrix_rank(O_obs, tol=tol))

    def output_controllability_matrix(self, A: np.ndarray, B: np.ndarray,
                                      C: np.ndarray) -> np.ndarray:
        """Output controllability matrix 𝒞_out = C · 𝒞."""
        return C @ self.controllability_matrix(A, B)

    def output_controllability_rank(self, A: np.ndarray, B: np.ndarray,
                                    C: np.ndarray, tol: float = 1e-10) -> int:
        """Rank of the output controllability matrix."""
        C_out = self.output_controllability_matrix(A, B, C)
        return int(np.linalg.matrix_rank(C_out, tol=tol))


# =====================================================================
# §3 — TMS Plant Model
# =====================================================================

class TMSPlant(PlantModel):
    """Concrete plant model for the Omnidream TMS coil array.

    Integrates the existing physics modules (ti_fields, nts_timing,
    coupling, em_theory) behind the PlantModel interface.

    State vector (2N+2):
        x = [I(N), V_m(1), T(N), M_target(1)]

    Input (N):
        u = [V_command(N)]

    Output (5):
        y = [M_target, M_surface_max, V_m_peak, SAR_max, T_coil_max]
    """

    def __init__(
        self,
        basis_matrix: np.ndarray,
        target_idx: Any,
        surface_indices: np.ndarray,
        config: Optional[OmnidreamConfig] = None,
        mode: StimulationMode = StimulationMode.TI,
        positions_mm: Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        basis_matrix : ndarray (num_points, N)
            Pre-computed basis E-fields.
        target_idx : int or ndarray
            Target sample point index/indices.
        surface_indices : ndarray
            Surface/off-target sample point indices.
        config : OmnidreamConfig, optional
            System configuration (defaults to paper_baseline).
        mode : StimulationMode
            Active stimulation mode.
        positions_mm : ndarray (N, 3), optional
            Coil positions for inductance computation.
        """
        self.basis_matrix = np.asarray(basis_matrix, dtype=float)
        self.target_idx = target_idx
        self.surface_indices = np.asarray(surface_indices, dtype=int)
        self.config = config or paper_baseline()
        self.mode = mode
        self.n_coils = self.basis_matrix.shape[1]

        # Build inductance and impedance matrices
        self._build_circuit_model(positions_mm)

        # Cache em_theory objects
        from em_theory import NeuralResponse, SARThermal
        self.neural = NeuralResponse(tau_m_s=self.config.nts.tau_m_s)
        self.sar_calc = SARThermal

    def _build_circuit_model(self, positions_mm: Optional[np.ndarray]) -> None:
        """Build L, R, Z matrices from coil geometry."""
        N = self.n_coils
        coil = self.config.coil

        # Resistance matrix (diagonal)
        self.R = np.eye(N) * coil.coil_resistance_ohm

        # Inductance matrix
        if positions_mm is not None:
            from coupling import build_inductance_matrix
            self.L = build_inductance_matrix(positions_mm, coil)
        else:
            # Default: diagonal (no mutual coupling)
            self.L = np.eye(N) * coil.self_inductance_h

        # Impedance matrix at carrier frequency
        freq = self.config.ti.freq_carrier_hz
        from coupling import build_impedance_matrix
        self.Z = build_impedance_matrix(self.L, coil.coil_resistance_ohm, freq)

        # L inverse (used in state-space A, B matrices)
        try:
            self.L_inv = np.linalg.inv(self.L)
        except np.linalg.LinAlgError:
            self.L_inv = np.linalg.pinv(self.L)

        # Thermal parameters (simplified)
        self.C_thermal = 0.5 * N  # J/K per coil (approximate for small coils)
        self.h_conv = 0.01        # W/K convective heat transfer
        self.T_ambient = 37.0     # body temperature °C

    def dims(self) -> PlantDimensions:
        return PlantDimensions.from_n_coils(self.n_coils)

    # -----------------------------------------------------------------
    # Forward model
    # -----------------------------------------------------------------

    def forward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute outputs from state and input.

        Uses the quasi-static approximation: I = Z⁻¹V (instantaneous
        circuit response), then computes fields from currents.
        """
        N = self.n_coils
        d = self.dims()

        # Extract state components
        I_coils = x[:N]
        V_m = x[N]
        T_coils = x[N + 1:2 * N + 1]

        # Compute outputs based on mode
        if self.mode == StimulationMode.TI:
            y = self._forward_ti(I_coils, V_m, T_coils)
        elif self.mode == StimulationMode.NTS:
            y = self._forward_nts(I_coils, V_m, T_coils)
        else:
            # Hybrid: compute both and take the max at target
            y_ti = self._forward_ti(I_coils, V_m, T_coils)
            y_nts = self._forward_nts(I_coils, V_m, T_coils)
            y = np.array([
                max(y_ti[0], y_nts[0]),         # M_target (best of both)
                max(y_ti[1], y_nts[1]),         # M_surface_max (worst case)
                max(y_ti[2], y_nts[2]),         # V_m_peak
                max(y_ti[3], y_nts[3]),         # SAR_max (worst case)
                max(y_ti[4], y_nts[4]),         # T_max (worst case)
            ])
        return y

    def _forward_ti(self, I_coils: np.ndarray, V_m: float,
                    T_coils: np.ndarray) -> np.ndarray:
        """TI-specific forward model."""
        from ti_fields import compute_group_amplitudes, compute_modulation_depth

        # Default group assignment and frequencies
        group = self._default_group()
        f1 = self.config.ti.freq_carrier_hz
        f2 = f1 + self.config.ti.delta_freq_default_hz

        # Use absolute currents as amplitudes
        amplitudes = np.abs(I_coils)

        A1, A2 = compute_group_amplitudes(amplitudes, group, self.basis_matrix, f1, f2)
        M = compute_modulation_depth(A1, A2)

        M_target = float(np.mean(M[self.target_idx]))
        M_surface_max = float(np.max(M[self.surface_indices])) if len(self.surface_indices) > 0 else 0.0

        # SAR
        SAR = self.sar_calc.sar_ti(A1[self.surface_indices], A2[self.surface_indices],
                                    sigma_sm=self.config.safety.sigma_gm_sm,
                                    rho_kgm3=self.config.safety.rho_tissue_kgm3)
        SAR_max = float(np.max(SAR)) if len(SAR) > 0 else 0.0

        T_max = float(np.max(T_coils)) if len(T_coils) > 0 else self.T_ambient

        return np.array([M_target, M_surface_max, V_m, SAR_max, T_max])

    def _forward_nts(self, I_coils: np.ndarray, V_m: float,
                     T_coils: np.ndarray) -> np.ndarray:
        """NTS-specific forward model."""
        from nts_timing import compute_v_peak, optimal_firing_order, assign_uniform_fire_times

        amplitudes = np.abs(I_coils)

        # Default firing order: optimal by target field strength
        order = optimal_firing_order(self.basis_matrix, self.target_idx)
        fire_times = assign_uniform_fire_times(order, self.config.nts.tau_window_s)

        V_peak = compute_v_peak(amplitudes, fire_times, self.basis_matrix,
                                tau_m=self.config.nts.tau_m_s,
                                q_pulse=self.config.nts.q_pulse)

        V_target = float(np.mean(V_peak[self.target_idx]))
        V_surface_max = float(np.max(V_peak[self.surface_indices])) if len(self.surface_indices) > 0 else 0.0

        # SAR for pulsed mode (duty-cycle-adjusted)
        E_rms = amplitudes * np.mean(np.abs(self.basis_matrix), axis=0)
        SAR = self.sar_calc.sar_pointwise(
            E_rms, sigma_sm=self.config.safety.sigma_gm_sm,
            rho_kgm3=self.config.safety.rho_tissue_kgm3)
        SAR_max = float(np.max(SAR)) * self.config.safety.max_duty_cycle_pulsed

        T_max = float(np.max(T_coils)) if len(T_coils) > 0 else self.T_ambient

        return np.array([V_target, V_surface_max, V_m, SAR_max, T_max])

    def _default_group(self) -> np.ndarray:
        """Default 50/50 group assignment for TI."""
        group = np.zeros(self.n_coils)
        group[self.n_coils // 2:] = 1.0
        return group

    # -----------------------------------------------------------------
    # Parameter-space forward (for optimisers)
    # -----------------------------------------------------------------

    def forward_from_params(self, amplitudes: np.ndarray,
                            **kwargs: Any) -> np.ndarray:
        """Compute outputs from optimisation parameters.

        For TI: kwargs should include group, freq1, freq2.
        For NTS: kwargs should include fire_times.
        """
        if self.mode == StimulationMode.TI:
            return self._forward_ti_params(amplitudes, **kwargs)
        elif self.mode == StimulationMode.NTS:
            return self._forward_nts_params(amplitudes, **kwargs)
        else:
            y_ti = self._forward_ti_params(amplitudes, **kwargs)
            y_nts = self._forward_nts_params(amplitudes, **kwargs)
            return np.array([
                max(y_ti[0], y_nts[0]),
                max(y_ti[1], y_nts[1]),
                max(y_ti[2], y_nts[2]),
                max(y_ti[3], y_nts[3]),
                max(y_ti[4], y_nts[4]),
            ])

    def _forward_ti_params(self, amplitudes: np.ndarray, **kwargs: Any) -> np.ndarray:
        """TI forward from optimisation parameters."""
        from ti_fields import compute_group_amplitudes, compute_modulation_depth

        group = kwargs.get("group", self._default_group())
        f1 = kwargs.get("freq1", self.config.ti.freq_carrier_hz)
        f2 = kwargs.get("freq2", f1 + self.config.ti.delta_freq_default_hz)

        A1, A2 = compute_group_amplitudes(amplitudes, group, self.basis_matrix, f1, f2)
        M = compute_modulation_depth(A1, A2)

        M_target = float(np.mean(M[self.target_idx]))
        M_surface_max = float(np.max(M[self.surface_indices])) if len(self.surface_indices) > 0 else 0.0

        SAR = self.sar_calc.sar_ti(A1[self.surface_indices], A2[self.surface_indices],
                                    sigma_sm=self.config.safety.sigma_gm_sm,
                                    rho_kgm3=self.config.safety.rho_tissue_kgm3)
        SAR_max = float(np.max(SAR)) if len(SAR) > 0 else 0.0

        V_m = M_target * self.neural.transfer_function_magnitude(
            abs(f2 - f1))  # beat envelope filtered by membrane

        T_max = self.T_ambient  # steady-state approximation
        from em_theory import SARThermal
        if SAR_max > 0:
            T_max += SARThermal.pennes_steady_state_delta_T(SAR_max)

        return np.array([M_target, M_surface_max, V_m, SAR_max, T_max])

    def _forward_nts_params(self, amplitudes: np.ndarray, **kwargs: Any) -> np.ndarray:
        """NTS forward from optimisation parameters."""
        from nts_timing import compute_v_peak

        fire_times = kwargs.get("fire_times", None)
        if fire_times is None:
            from nts_timing import optimal_firing_order, assign_uniform_fire_times
            order = optimal_firing_order(self.basis_matrix, self.target_idx)
            fire_times = assign_uniform_fire_times(order, self.config.nts.tau_window_s)

        V_peak = compute_v_peak(amplitudes, fire_times, self.basis_matrix,
                                tau_m=self.config.nts.tau_m_s,
                                q_pulse=self.config.nts.q_pulse)

        V_target = float(np.mean(V_peak[self.target_idx]))
        V_surface_max = float(np.max(V_peak[self.surface_indices])) if len(self.surface_indices) > 0 else 0.0
        V_m = V_target  # membrane potential equals peak for NTS

        E_rms = amplitudes * np.mean(np.abs(self.basis_matrix), axis=0)
        SAR = self.sar_calc.sar_pointwise(
            E_rms, sigma_sm=self.config.safety.sigma_gm_sm,
            rho_kgm3=self.config.safety.rho_tissue_kgm3)
        SAR_max = float(np.max(SAR)) * self.config.safety.max_duty_cycle_pulsed

        T_max = self.T_ambient
        from em_theory import SARThermal
        if SAR_max > 0:
            T_max += SARThermal.pennes_steady_state_delta_T(SAR_max)

        return np.array([V_target, V_surface_max, V_m, SAR_max, T_max])

    # -----------------------------------------------------------------
    # Linearization (finite-difference Jacobians)
    # -----------------------------------------------------------------

    def linearize(self, x0: np.ndarray, u0: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Linearize via analytical + finite-difference approach.

        Returns (A, B, C, D) at operating point (x0, u0).

        A is constructed analytically from the time-scale separated dynamics:
            - Circuit block: -L⁻¹ R
            - Neural block: -1/τ_m
            - Thermal block: -h_conv/C_th

        B, C, D use finite differences around the operating point.
        """
        N = self.n_coils
        n = 2 * N + 2
        m = N
        p = 5

        # --- A matrix (analytical) ---
        A = np.zeros((n, n))

        # Circuit block (N×N): dI/dt = L⁻¹(V - RI) → ∂/∂I = -L⁻¹ R
        A[:N, :N] = -self.L_inv @ self.R

        # Neural block (1×1): dV_m/dt = (-V_m + g(I))/τ_m → ∂/∂V_m = -1/τ_m
        tau_m = self.config.nts.tau_m_s
        A[N, N] = -1.0 / tau_m

        # Neural-circuit coupling (1×N): ∂(dV_m/dt)/∂I
        # Approximate via finite difference
        eps_I = 1e-6
        I0 = x0[:N]
        y0 = self.forward(x0, u0)
        for i in range(N):
            x_pert = x0.copy()
            x_pert[i] += eps_I
            y_pert = self.forward(x_pert, u0)
            # V_m response to current change
            A[N, i] = (y_pert[2] - y0[2]) / (eps_I * tau_m)

        # Thermal block (N×N): dT/dt = (I²R - h(T-T_a))/C_th → ∂/∂T = -h/C
        for i in range(N):
            A[N + 1 + i, N + 1 + i] = -self.h_conv / self.C_thermal

        # Thermal-circuit coupling: ∂(dT/dt)/∂I = 2IR/C_th
        for i in range(N):
            A[N + 1 + i, i] = 2 * I0[i] * self.R[i, i] / self.C_thermal

        # --- B matrix (analytical) ---
        B = np.zeros((n, m))
        B[:N, :N] = self.L_inv  # dI/dt = L⁻¹ V

        # --- C matrix (finite difference) ---
        C = np.zeros((p, n))
        eps_x = 1e-6
        for j in range(n):
            x_plus = x0.copy()
            x_plus[j] += eps_x
            y_plus = self.forward(x_plus, u0)
            C[:, j] = (y_plus - y0) / eps_x

        # --- D matrix (finite difference) ---
        D = np.zeros((p, m))
        eps_u = 1e-6
        for j in range(m):
            u_plus = u0.copy()
            u_plus[j] += eps_u
            y_plus = self.forward(x0, u_plus)
            D[:, j] = (y_plus - y0) / eps_u

        return A, B, C, D

    def jacobian_output_wrt_input(self, x: np.ndarray, u: np.ndarray
                                   ) -> np.ndarray:
        """Compute J = ∂y/∂u via central finite differences.

        Parameters
        ----------
        x : ndarray (n_state,)
        u : ndarray (m_input,)

        Returns
        -------
        J : ndarray (p_output, m_input)
        """
        p = 5
        m = len(u)
        J = np.zeros((p, m))
        eps = 1e-6

        for j in range(m):
            u_plus = u.copy()
            u_minus = u.copy()
            u_plus[j] += eps
            u_minus[j] -= eps
            y_plus = self.forward(x, u_plus)
            y_minus = self.forward(x, u_minus)
            J[:, j] = (y_plus - y_minus) / (2 * eps)

        return J

    def jacobian_output_wrt_amplitudes(self, amplitudes: np.ndarray,
                                        **kwargs: Any) -> np.ndarray:
        """Compute J = ∂y/∂α via central finite differences.

        This operates in parameter space rather than state space,
        which is more natural for the GA/SAC optimisers.

        Parameters
        ----------
        amplitudes : ndarray (N,)
        **kwargs : mode-specific parameters

        Returns
        -------
        J : ndarray (p_output, N)
        """
        p = 5
        N = len(amplitudes)
        J = np.zeros((p, N))
        eps = 1e-6

        for j in range(N):
            a_plus = amplitudes.copy()
            a_minus = amplitudes.copy()
            a_plus[j] += eps
            a_minus[j] -= eps
            # Clamp to non-negative
            a_plus[j] = max(a_plus[j], 0.0)
            a_minus[j] = max(a_minus[j], 0.0)
            y_plus = self.forward_from_params(a_plus, **kwargs)
            y_minus = self.forward_from_params(a_minus, **kwargs)
            J[:, j] = (y_plus - y_minus) / (a_plus[j] - a_minus[j] + 1e-30)

        return J

    # -----------------------------------------------------------------
    # State construction helpers
    # -----------------------------------------------------------------

    def make_state(self, currents: Optional[np.ndarray] = None,
                   V_m: float = 0.0,
                   temperatures: Optional[np.ndarray] = None,
                   M_target: float = 0.0) -> np.ndarray:
        """Construct a state vector from components."""
        N = self.n_coils
        x = np.zeros(2 * N + 2)
        if currents is not None:
            x[:N] = currents
        x[N] = V_m
        if temperatures is not None:
            x[N + 1:2 * N + 1] = temperatures
        else:
            x[N + 1:2 * N + 1] = self.T_ambient
        x[2 * N + 1] = M_target
        return x

    def make_input(self, voltages: Optional[np.ndarray] = None) -> np.ndarray:
        """Construct an input vector."""
        if voltages is not None:
            return np.asarray(voltages, dtype=float)
        return np.zeros(self.n_coils)

    # -----------------------------------------------------------------
    # Derived analysis
    # -----------------------------------------------------------------

    def condition_number(self, x: np.ndarray, u: np.ndarray) -> float:
        """Condition number κ = σ_max/σ_min of the output Jacobian.

        κ ≈ 1 means well-conditioned (all inputs equally useful).
        κ ≫ 1 means some inputs barely affect outputs.
        """
        J = self.jacobian_output_wrt_input(x, u)
        sv = np.linalg.svd(J, compute_uv=False)
        sv_nonzero = sv[sv > 1e-15]
        if len(sv_nonzero) < 2:
            return float("inf")
        return float(sv_nonzero[0] / sv_nonzero[-1])

    def condition_number_params(self, amplitudes: np.ndarray,
                                **kwargs: Any) -> float:
        """Condition number of the parameter-space Jacobian ∂y/∂α."""
        J = self.jacobian_output_wrt_amplitudes(amplitudes, **kwargs)
        sv = np.linalg.svd(J, compute_uv=False)
        sv_nonzero = sv[sv > 1e-15]
        if len(sv_nonzero) < 2:
            return float("inf")
        return float(sv_nonzero[0] / sv_nonzero[-1])


# =====================================================================
# §4 — Abstract Controller
# =====================================================================

class Controller(ABC):
    """Abstract base class for a stimulation controller.

    Controllers select actions (voltage commands) given the current
    plant state and a goal specification.  They are agnostic to the
    specific plant — any PlantModel can be controlled.
    """

    @abstractmethod
    def select_action(self, state: np.ndarray, goal: GoalSpec,
                      plant: PlantModel) -> np.ndarray:
        """Compute the next control input.

        Parameters
        ----------
        state : ndarray (n_state,)
            Current plant state.
        goal : GoalSpec
            What the system should achieve.
        plant : PlantModel
            The plant to control (for forward model queries).

        Returns
        -------
        u : ndarray (m_input,)
            Voltage command vector.
        """
        ...


class GreedyJacobianController(Controller):
    """Simple controller: step in the gradient direction of the goal.

    At each step, computes the Jacobian ∂y/∂u and takes a step that
    increases M_target while decreasing M_surface.  This is a single-step
    greedy policy — not optimal, but useful as a baseline.
    """

    def __init__(self, step_size: float = 0.1, max_voltage: float = 60.0):
        self.step_size = step_size
        self.max_voltage = max_voltage

    def select_action(self, state: np.ndarray, goal: GoalSpec,
                      plant: PlantModel) -> np.ndarray:
        """Gradient-based action selection."""
        N = plant.dims().n_coils
        u_current = state[:N] * plant.R[np.diag_indices(N)[0], np.diag_indices(N)[1]]

        J = plant.jacobian_output_wrt_input(state, u_current)

        # Weighted gradient: increase M_target, decrease M_surface, decrease SAR
        w = np.zeros(5)
        w[0] = goal.weights.get(SystemGoal.FOCAL_DEPTH, 1.0)
        w[1] = -goal.weights.get(SystemGoal.SPATIAL_SELECTIVITY, 0.5)
        w[3] = -0.1  # SAR penalty
        w[4] = -goal.weights.get(SystemGoal.THERMAL_HEADROOM, 0.05)

        # Direction: Jᵀ w gives ∂(wᵀy)/∂u
        direction = J.T @ w

        # Normalise and scale
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction = direction / norm

        u_new = u_current + self.step_size * direction

        # Clamp
        u_new = np.clip(u_new, -self.max_voltage, self.max_voltage)

        return u_new


# =====================================================================
# §5 — Cost Function Builder
# =====================================================================

def build_cost_function(plant: PlantModel, goal: GoalSpec,
                        mode: StimulationMode = StimulationMode.TI
                        ):
    """Build a scalar cost function from a plant and goal specification.

    Returns a callable cost_fn(amplitudes, **kwargs) → float
    suitable for scipy.optimize or the GA.

    The cost is:
        J = -w_depth · y[0] + w_select · y[1] + w_power · Σα²
            + penalty(SAR) + penalty(T)
    """
    w_depth = goal.weights.get(SystemGoal.FOCAL_DEPTH, 1.0)
    w_select = goal.weights.get(SystemGoal.SPATIAL_SELECTIVITY, 0.5)
    w_power = goal.weights.get(SystemGoal.POWER_EFFICIENCY, 0.01)
    sar_limit = goal.constraints.get("SAR_max", 3.2)
    t_limit = goal.constraints.get("T_max", 41.0)

    def cost_fn(amplitudes: np.ndarray, **kwargs: Any) -> float:
        y = plant.forward_from_params(amplitudes, **kwargs)
        y_map = map_output_vector(y)

        cost = -(w_depth * y_map["target_metric"]
                 - w_select * y_map["surface_metric"]
                 - w_power * float(np.sum(amplitudes ** 2)))

        # Safety penalties
        if y_map["sar_max"] > sar_limit:
            cost += 1000.0 * (y_map["sar_max"] / sar_limit)
        if y_map["temperature_max"] > t_limit:
            cost += 1000.0 * (y_map["temperature_max"] / t_limit)

        return cost

    return cost_fn


# =====================================================================
# §6 — Convenience Analysis Functions
# =====================================================================

def analyse_plant(plant: TMSPlant,
                  amplitudes: Optional[np.ndarray] = None,
                  **kwargs: Any) -> dict:
    """Run a comprehensive analysis of the plant at an operating point.

    Returns a dictionary with forward outputs, Jacobian, condition number,
    controllability rank, and linearized system matrices.
    """
    N = plant.n_coils

    if amplitudes is None:
        amplitudes = np.ones(N) * 0.5

    # State at operating point
    x0 = plant.make_state(currents=amplitudes)
    u0 = plant.make_input(amplitudes * plant.config.coil.coil_resistance_ohm)

    # Forward pass
    y = plant.forward(x0, u0)
    y_state_map = map_output_vector(y)

    # Parameter-space forward
    y_params = plant.forward_from_params(amplitudes, **kwargs)
    y_params_map = map_output_vector(y_params)

    # Jacobians
    J_state = plant.jacobian_output_wrt_input(x0, u0)
    J_params = plant.jacobian_output_wrt_amplitudes(amplitudes, **kwargs)

    # SVD of parameter Jacobian
    sv = np.linalg.svd(J_params, compute_uv=False)

    # Condition number
    kappa_state = plant.condition_number(x0, u0)
    kappa_params = plant.condition_number_params(amplitudes, **kwargs)

    # Linearize
    A, B, C, D = plant.linearize(x0, u0)

    # Controllability
    ctrl_rank = plant.controllability_rank(A, B)
    obs_rank = plant.observability_rank(A, C)
    out_ctrl_rank = plant.output_controllability_rank(A, B, C)

    return {
        # Outputs
        "y_state": y,
        "y_params": y_params,
        "y_state_map": y_state_map,
        "y_params_map": y_params_map,
        "output_labels": STANDARD_OUTPUT_LABELS,
        "output_labels_legacy": LEGACY_OUTPUT_LABELS,
        # Jacobians
        "J_state": J_state,
        "J_params": J_params,
        "singular_values": sv,
        # Condition numbers
        "kappa_state": kappa_state,
        "kappa_params": kappa_params,
        # State-space
        "A": A, "B": B, "C": C, "D": D,
        # Controllability
        "controllability_rank": ctrl_rank,
        "observability_rank": obs_rank,
        "output_controllability_rank": out_ctrl_rank,
        "n_state": A.shape[0],
        "n_input": B.shape[1],
        "n_output": C.shape[0],
    }
