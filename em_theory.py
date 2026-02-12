"""First-principles electromagnetic theory for the Omnidream TMS array.

Implements the mathematical objects derived in ``Blueprints/em_foundations.md``:
    - Quasi-static validation
    - Cole-Cole tissue conductivity model
    - Coil self- and mutual inductance formulas
    - Steinmetz core loss
    - Neural membrane low-pass filter
    - SAR and thermal steady-state estimates

This module is standalone: it depends only on numpy and the Omnidream ``config``
module.  It does NOT depend on ti_fields, nts_timing, or basis_fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# =====================================================================
# Constants
# =====================================================================

MU_0 = 4 * math.pi * 1e-7  # vacuum permeability [H/m]
C_LIGHT = 2.998e8           # speed of light [m/s]
C_MEMBRANE = 0.01           # universal membrane capacitance [F/m²]


# =====================================================================
# §1 — Maxwell / Quasi-Static Validation
# =====================================================================

class MaxwellBasis:
    """Quasi-static field theory utilities."""

    @staticmethod
    def wavelength(freq_hz: float) -> float:
        """Electromagnetic wavelength λ = c/f  [m]."""
        if freq_hz <= 0:
            return float("inf")
        return C_LIGHT / freq_hz

    @staticmethod
    def quasi_static_validity(freq_hz: float, head_diameter_m: float = 0.2,
                              safety_factor: float = 100.0) -> bool:
        """Return True if λ > safety_factor × head diameter.

        Default criterion: λ > 100 × 0.2 m = 20 m, satisfied for f < 15 MHz.
        """
        lam = MaxwellBasis.wavelength(freq_hz)
        return lam > safety_factor * head_diameter_m

    @staticmethod
    def quasi_static_ratio(freq_hz: float, head_diameter_m: float = 0.2) -> float:
        """Return λ / head_diameter (should be ≫ 1)."""
        return MaxwellBasis.wavelength(freq_hz) / head_diameter_m

    @staticmethod
    def linearity_residual(E_combined: np.ndarray,
                           E_individual_sum: np.ndarray) -> float:
        """Relative L2 error between combined FEM field and superposition.

        Parameters
        ----------
        E_combined : ndarray
            |E| on sample points from multi-coil FEM simulation.
        E_individual_sum : ndarray
            Σ αᵢ |Eᵢ| from individual basis fields.

        Returns
        -------
        float
            ||E_combined − E_sum||₂ / ||E_combined||₂
        """
        norm = np.linalg.norm(E_combined)
        if norm < 1e-30:
            return 0.0
        return float(np.linalg.norm(E_combined - E_individual_sum) / norm)


# =====================================================================
# §2 — Tissue Conductivity (Cole-Cole)
# =====================================================================

@dataclass
class TissueModel:
    """Frequency-dependent tissue conductivity via Cole-Cole relaxation.

    σ(ω) = σ_∞ + (σ₀ − σ_∞) / (1 + (jωτ)^α)

    References: Gabriel et al. 1996, IT'IS tissue database.
    """

    name: str
    sigma_0: float    # DC conductivity for this dispersion [S/m]
    sigma_inf: float  # high-frequency limit [S/m]
    tau_s: float      # relaxation time [s]
    alpha: float      # broadening exponent (0 < α ≤ 1)
    rho_kgm3: float   # mass density [kg/m³]

    def conductivity(self, freq_hz: float) -> complex:
        """Complex conductivity σ(ω) at a given frequency."""
        omega = 2 * math.pi * freq_hz
        denom = 1.0 + (1j * omega * self.tau_s) ** self.alpha
        return self.sigma_inf + (self.sigma_0 - self.sigma_inf) / denom

    def conductivity_magnitude(self, freq_hz: float) -> float:
        """Real part of σ (dominates losses)."""
        return float(np.real(self.conductivity(freq_hz)))

    def conductivity_sweep(self, freq_array: np.ndarray) -> np.ndarray:
        """Vectorized: |Re(σ)| over a frequency array."""
        return np.array([self.conductivity_magnitude(f) for f in freq_array])


# Standard tissue presets (values at β-dispersion range, ~100 Hz – 100 kHz)
TissueModel.GRAY_MATTER = TissueModel(
    name="gray_matter", sigma_0=0.020, sigma_inf=0.106,
    tau_s=7.96e-3, alpha=0.1, rho_kgm3=1040.0,
)
TissueModel.WHITE_MATTER = TissueModel(
    name="white_matter", sigma_0=0.015, sigma_inf=0.065,
    tau_s=7.96e-3, alpha=0.1, rho_kgm3=1040.0,
)
TissueModel.CSF = TissueModel(
    name="csf", sigma_0=1.654, sigma_inf=1.654,
    tau_s=1e-6, alpha=1.0, rho_kgm3=1007.0,
)
TissueModel.SKULL = TissueModel(
    name="skull", sigma_0=0.010, sigma_inf=0.012,
    tau_s=1e-2, alpha=0.15, rho_kgm3=1900.0,
)
TissueModel.SCALP = TissueModel(
    name="scalp", sigma_0=0.43, sigma_inf=0.45,
    tau_s=1e-2, alpha=0.1, rho_kgm3=1100.0,
)


# =====================================================================
# §3 — Coil Inductance
# =====================================================================

class CoilInductance:
    """Self- and mutual inductance formulas for TMS coils."""

    @staticmethod
    def wheeler_rectangular(n_turns: int, width_m: float, height_m: float,
                            depth_m: float, mu_r: float = 1.0) -> float:
        """Wheeler approximate self-inductance for a rectangular solenoid.

        L = μ₀ μ_r N² A / ℓ  (simplified, no Nagaoka correction)

        Parameters
        ----------
        n_turns : int
        width_m, height_m : float – core cross-section [m]
        depth_m : float – winding length [m]
        mu_r : float – relative permeability of core

        Returns
        -------
        float – self-inductance [H]
        """
        A = width_m * height_m
        return MU_0 * mu_r * n_turns ** 2 * A / depth_m

    @staticmethod
    def magnetic_circuit_c_shape(n_turns: int, core_area_m2: float,
                                 core_length_m: float, gap_length_m: float,
                                 mu_r: float) -> float:
        """Self-inductance of C-shaped coil via magnetic circuit theory.

        L = N² / (R_core + R_gap)
        where R = ℓ / (μ A) is the magnetic reluctance.

        The air gap dominates reluctance for high-μ_r cores.
        """
        R_core = core_length_m / (mu_r * MU_0 * core_area_m2)
        R_gap = gap_length_m / (MU_0 * core_area_m2)
        return n_turns ** 2 / (R_core + R_gap)

    @staticmethod
    def self_inductance_c_shape(n_turns: int = 30,
                                width_m: float = 7e-3,
                                height_m: float = 4e-3,
                                core_length_m: float = 15e-3,
                                gap_length_m: float = 5e-3,
                                mu_r: float = 75.0) -> float:
        """Self-inductance for the Omnidream C-shaped coil.

        Uses magnetic circuit theory with core + air gap reluctances.
        """
        A = width_m * height_m
        R_core = core_length_m / (mu_r * MU_0 * A)
        R_gap = gap_length_m / (MU_0 * A)
        return n_turns ** 2 / (R_core + R_gap)

    @staticmethod
    def neumann_dipole_mutual(pos_i_m: np.ndarray, pos_j_m: np.ndarray,
                              n_turns: int, area_m2: float) -> float:
        """Dipole-approximation mutual inductance.

        M_ij ≈ (μ₀/4π) · n² · A² / d³   [aligned dipoles, simplified]
        """
        d = np.linalg.norm(pos_i_m - pos_j_m)
        if d < 1e-12:
            return 0.0
        return MU_0 * n_turns ** 2 * area_m2 ** 2 / (4 * math.pi * d ** 3)

    @staticmethod
    def coupling_coefficient(M_ij: float, L_i: float, L_j: float) -> float:
        """k = M / √(L_i L_j)."""
        denom = math.sqrt(abs(L_i * L_j))
        if denom < 1e-30:
            return 0.0
        return abs(M_ij) / denom

    @staticmethod
    def steinmetz_core_loss(freq_hz: float, B_max_T: float,
                            k: float = 6.5, a: float = 1.3,
                            b: float = 2.5) -> float:
        """Steinmetz equation: P_v = k · f^a · B_max^b  [W/m³].

        Default parameters for carbonyl iron powder cores.
        """
        if freq_hz <= 0 or B_max_T <= 0:
            return 0.0
        return k * freq_hz ** a * B_max_T ** b


# =====================================================================
# §4 — Neural Response
# =====================================================================

class NeuralResponse:
    """Membrane potential dynamics and low-pass filtering.

    Implements the integrate-and-fire approximation:
        τ_m dV_m/dt = -V_m + r_m I_ext
    """

    def __init__(self, tau_m_s: float = 3e-3, axon_diam_um: float = 1.0):
        self.tau_m = tau_m_s
        self.axon_diam = axon_diam_um * 1e-6  # convert to metres

    def low_pass_cutoff_hz(self) -> float:
        """Neural membrane cutoff frequency f_c = 1/(2π τ_m)."""
        return 1.0 / (2 * math.pi * self.tau_m)

    def transfer_function(self, freq_hz: float) -> complex:
        """H(f) = 1 / (1 + j·2π f τ_m)."""
        omega = 2 * math.pi * freq_hz
        return 1.0 / (1.0 + 1j * omega * self.tau_m)

    def transfer_function_magnitude(self, freq_hz: float) -> float:
        """|H(f)| = 1 / √(1 + (2πfτ_m)²)."""
        x = 2 * math.pi * freq_hz * self.tau_m
        return 1.0 / math.sqrt(1.0 + x * x)

    def carrier_attenuation_db(self, carrier_hz: float) -> float:
        """Attenuation of TI carrier by membrane low-pass [dB]."""
        mag = self.transfer_function_magnitude(carrier_hz)
        if mag < 1e-30:
            return -300.0
        return 20.0 * math.log10(mag)

    def beat_transmission(self, beat_hz: float) -> float:
        """Fraction of beat envelope that passes through membrane filter."""
        return self.transfer_function_magnitude(beat_hz)

    def ti_effective_ratio(self, carrier_hz: float, beat_hz: float) -> float:
        """Ratio of beat signal to carrier leakage through membrane.

        Higher is better — means neurons respond to beat, not carrier.
        """
        carrier_mag = self.transfer_function_magnitude(carrier_hz)
        beat_mag = self.transfer_function_magnitude(beat_hz)
        if carrier_mag < 1e-30:
            return float("inf")
        return beat_mag / carrier_mag

    def electrotonic_length_constant_m(self, R_m_ohm_m2: float = 0.3,
                                        R_i_ohm_m: float = 1.5) -> float:
        """λ = √(a R_m / (4 R_i)) where a = axon diameter."""
        return math.sqrt(self.axon_diam * R_m_ohm_m2 / (4 * R_i_ohm_m))

    def membrane_time_constant_s(self, R_m_ohm_m2: float = 0.3) -> float:
        """τ_m = R_m · C_m."""
        return R_m_ohm_m2 * C_MEMBRANE


# =====================================================================
# §5 — SAR and Thermal
# =====================================================================

class SARThermal:
    """Specific Absorption Rate and thermal modeling."""

    @staticmethod
    def sar_pointwise(E_rms_vpm: np.ndarray, sigma_sm: float = 0.106,
                      rho_kgm3: float = 1040.0) -> np.ndarray:
        """SAR = σ |E_rms|² / ρ  [W/kg].

        Parameters
        ----------
        E_rms_vpm : ndarray – RMS E-field magnitude at each point [V/m]
        sigma_sm : float – tissue conductivity [S/m]
        rho_kgm3 : float – tissue density [kg/m³]
        """
        return sigma_sm * E_rms_vpm ** 2 / rho_kgm3

    @staticmethod
    def sar_ti(A1_vpm: np.ndarray, A2_vpm: np.ndarray,
               sigma_sm: float = 0.106, rho_kgm3: float = 1040.0) -> np.ndarray:
        """SAR for TI: two sinusoids at different frequencies.

        E_rms² = 0.5(|A₁|² + |A₂|²)  (orthogonal frequencies).
        """
        E_rms_sq = 0.5 * (A1_vpm ** 2 + A2_vpm ** 2)
        return sigma_sm * E_rms_sq / rho_kgm3

    @staticmethod
    def pennes_steady_state_delta_T(sar_wkg: float,
                                     rho_kgm3: float = 1040.0,
                                     rho_b_cb: float = 3.6e6,
                                     w_b: float = 0.01) -> float:
        """Steady-state temperature rise from Pennes bioheat.

        ΔT = SAR · ρ / (ρ_b c_b w_b)

        Parameters
        ----------
        sar_wkg : float – peak SAR [W/kg]
        rho_kgm3 : float – tissue density
        rho_b_cb : float – blood volumetric heat capacity [J/(m³·K)]
        w_b : float – blood perfusion rate [1/s]
        """
        return sar_wkg * rho_kgm3 / (rho_b_cb * w_b)

    @staticmethod
    def thermal_time_constant_s(rho_kgm3: float = 1040.0,
                                 c_jkgk: float = 3630.0,
                                 rho_b_cb: float = 3.6e6,
                                 w_b: float = 0.01) -> float:
        """τ_thermal = ρc / (ρ_b c_b w_b)  [s]."""
        return rho_kgm3 * c_jkgk / (rho_b_cb * w_b)

    @staticmethod
    def transient_fraction(t_s: float, tau_thermal_s: float) -> float:
        """Fraction of steady-state ΔT reached after time t.

        ΔT(t) / ΔT_ss = 1 - exp(-t/τ)
        """
        return 1.0 - math.exp(-t_s / tau_thermal_s)

    @staticmethod
    def coil_heating_W(current_A: float, resistance_ohm: float,
                       duty_cycle: float = 1.0) -> float:
        """Average power dissipated in coil: P = I² R × duty."""
        return current_A ** 2 * resistance_ohm * duty_cycle
