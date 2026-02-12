"""Unified configuration for the Omnidream TMS array system.

Provides dataclass-based configs for coil parameters, helmet geometry,
TI/NTS optimization, safety limits, and simulation settings.  Two preset
parameter sets are included:

* ``paper_baseline`` — Jiang et al. (2023) values (μ_r=75, 5 mm gap)
* ``omnidream_variant`` — local blueprint values (μ_r=5000, 0.5 mm gap)

All length units are **millimetres** and all electrical units are **SI**
unless noted otherwise in field docstrings.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Coil parameters
# ---------------------------------------------------------------------------

@dataclass
class CoilConfig:
    """Physical parameters for a single C-shaped miniature TMS coil."""

    total_turns: int = 30
    turns_per_base: int = 15
    winding_width_mm: float = 7.0
    winding_height_mm: float = 4.0
    base_gap_mm: float = 5.0
    base_axis_angle_deg: float = 75.0
    turn_pitch_mm: float = 0.20
    layers_per_base: int = 2
    layer_spacing_mm: float = 0.25
    segments_per_edge: int = 24
    z_offset_mm: float = -2.0
    # Metadata upper bound for coil-stimulator pairing in simulation artifacts.
    # Set above calibrated operating points (~6.87e8 A/s).
    stimulator_max_didt_as: float = 1.0e9
    casing_distance_mm: float = 1.0
    casing_thickness_mm: float = 0.5
    # Core material
    relative_permeability: float = 75.0
    wire_diameter_mm: float = 0.20
    core_conductivity_sm: float = 1.12e6
    # Electrical
    coil_resistance_ohm: float = 2.0  # approximate DC resistance
    self_inductance_h: float = 5e-6  # approximate
    effective_loop_area_m2: float = 28e-6  # 7 mm × 4 mm in m²


# ---------------------------------------------------------------------------
# Helmet geometry
# ---------------------------------------------------------------------------

@dataclass
class HelmetConfig:
    """Parameters for the concave helmet coil array."""

    r_inner_mm: float = 90.0
    theta_max_deg: float = 120.0
    n_coils: int = 32
    min_spacing_mm: float = 20.0
    clearance_mm: float = 5.0  # gap between coil surface and scalp


# ---------------------------------------------------------------------------
# Temporal Interference (TI)
# ---------------------------------------------------------------------------

@dataclass
class TIConfig:
    """Parameters for Temporal Interference optimisation."""

    freq_carrier_hz: float = 1000.0
    delta_freq_min_hz: float = 1.0
    delta_freq_max_hz: float = 100.0
    delta_freq_default_hz: float = 10.0
    lambda_surface_mean: float = 1.0
    lambda_surface_max: float = 5.0
    lambda_power: float = 0.01


# ---------------------------------------------------------------------------
# Neural Temporal Summation (NTS)
# ---------------------------------------------------------------------------

@dataclass
class NTSConfig:
    """Parameters for Neural Temporal Summation optimisation."""

    tau_m_s: float = 3e-3  # membrane time constant (seconds)
    tau_window_s: float = 5e-3  # integration window
    t_guard_s: float = 200e-6  # minimum inter-pulse spacing
    q_pulse: float = 1.0  # normalised pulse charge
    lambda_surface_max: float = 5.0
    lambda_per_pulse: float = 2.0
    lambda_power: float = 0.01


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

@dataclass
class SafetyConfig:
    """Safety thresholds for TMS operation."""

    # Thermal — pulsed mode
    temp_warning_pulsed_c: float = 40.0
    temp_critical_pulsed_c: float = 45.0
    # Thermal — continuous (TI) mode
    temp_warning_continuous_c: float = 38.0
    temp_critical_continuous_c: float = 41.0
    # Electrical
    max_current_a: float = 5.0
    max_voltage_v: float = 60.0
    # SAR
    sar_limit_wkg: float = 3.2  # IEEE C95.1 head, 10 g avg
    # Tissue properties for SAR estimation
    sigma_gm_sm: float = 0.106  # gray-matter conductivity (S/m)
    rho_tissue_kgm3: float = 1040.0  # tissue density (kg/m³)
    # Duty cycle
    max_duty_cycle_pulsed: float = 0.05
    max_duty_cycle_continuous: float = 1.0
    # Per-pulse surface threshold (V/m) from Jiang paper
    surface_e_threshold_vpm: float = 7.2


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Paths and simulation parameters."""

    head_mesh_path: str = ""
    coil_file_path: str = ""
    gm_tag: int = 2
    output_root: str = "simulations"
    basis_cache_dir: str = "basis_cache"
    results_dir: str = "results"
    didt_reference_as: float = 1e6


# ---------------------------------------------------------------------------
# Atlas (Stage 14)
# ---------------------------------------------------------------------------

@dataclass
class AtlasConfig:
    """Parameters for the perceptual outcome atlas (Stage 14)."""

    # --- Perceptual weights for metric / distance ---
    w_phi: float = 2.0
    w_sync_R: float = 1.0
    w_energy: float = 0.5
    w_J_phi: float = 1.0
    w_J_arch: float = 0.5
    w_J_sync: float = 1.0
    w_J_task: float = 0.5

    # --- Numerical ---
    finite_diff_eps: float = 1e-5  # step size for perceptual Jacobian
    boundary_safety_threshold: float = 0.2  # margin fraction for is_boundary

    # --- Valid-radius heuristic ---
    valid_radius_threshold: float = 0.1
    valid_radius_min: float = 0.001
    valid_radius_max: float = 10.0

    # --- Chart graph ---
    k_neighbors: int = 5
    n_barrier_samples: int = 10  # samples along transition paths
    compute_hessians: bool = True

    # --- Densification ---
    max_new_charts: int = 20
    densify_perturbation_scale: float = 0.01
    densify_k_connect: int = 3
    densify_barrier_samples: int = 3
    distance_threshold_quantile: float = 0.75

    # --- Geodesic ---
    geodesic_n_steps: int = 100


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class OmnidreamConfig:
    """Top-level configuration aggregating all sub-configs."""

    coil: CoilConfig = field(default_factory=CoilConfig)
    helmet: HelmetConfig = field(default_factory=HelmetConfig)
    ti: TIConfig = field(default_factory=TIConfig)
    nts: NTSConfig = field(default_factory=NTSConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    atlas: AtlasConfig = field(default_factory=AtlasConfig)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def paper_baseline() -> OmnidreamConfig:
    """Jiang et al. (2023) parameter set."""
    return OmnidreamConfig(
        coil=CoilConfig(
            relative_permeability=75.0,
            base_gap_mm=5.0,
            wire_diameter_mm=0.20,
            turn_pitch_mm=0.20,
        ),
    )


def omnidream_variant() -> OmnidreamConfig:
    """Local blueprint parameter set (higher permeability core)."""
    return OmnidreamConfig(
        coil=CoilConfig(
            relative_permeability=5000.0,
            base_gap_mm=0.5,
            wire_diameter_mm=0.10,
            turn_pitch_mm=0.05,
            winding_width_mm=7.0,
            winding_height_mm=4.0,
        ),
    )


def final_build_v1() -> OmnidreamConfig:
    """Finalized v1 build constraints used for implementation gating.

    This profile is paper-baseline geometry with explicit safety/drive limits
    and a stimulator dI/dt metadata ceiling aligned with calibrated operation.
    """
    cfg = paper_baseline()
    cfg.coil.stimulator_max_didt_as = 1.0e9
    cfg.helmet.n_coils = 32
    cfg.helmet.min_spacing_mm = 20.0
    cfg.safety.max_current_a = 5.0
    cfg.safety.max_voltage_v = 60.0
    cfg.safety.temp_warning_pulsed_c = 40.0
    cfg.safety.temp_critical_pulsed_c = 45.0
    cfg.safety.temp_warning_continuous_c = 38.0
    cfg.safety.temp_critical_continuous_c = 41.0
    cfg.safety.max_duty_cycle_pulsed = 0.05
    cfg.safety.sar_limit_wkg = 3.2
    cfg.ti.freq_carrier_hz = 1000.0
    cfg.ti.delta_freq_min_hz = 1.0
    cfg.ti.delta_freq_max_hz = 100.0
    cfg.nts.tau_m_s = 3e-3
    cfg.nts.tau_window_s = 5e-3
    cfg.nts.t_guard_s = 200e-6
    return cfg


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _to_dict(cfg: OmnidreamConfig) -> dict[str, Any]:
    return asdict(cfg)


def _from_dict(d: dict[str, Any]) -> OmnidreamConfig:
    return OmnidreamConfig(
        coil=CoilConfig(**d.get("coil", {})),
        helmet=HelmetConfig(**d.get("helmet", {})),
        ti=TIConfig(**d.get("ti", {})),
        nts=NTSConfig(**d.get("nts", {})),
        safety=SafetyConfig(**d.get("safety", {})),
        sim=SimConfig(**d.get("sim", {})),
        atlas=AtlasConfig(**d.get("atlas", {})),
    )


def save_config(cfg: OmnidreamConfig, path: str | Path) -> None:
    """Save configuration to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.dump(_to_dict(cfg), f, default_flow_style=False, sort_keys=False)


def load_config(path: str | Path) -> OmnidreamConfig:
    """Load configuration from a YAML file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return _from_dict(d)


def save_config_json(cfg: OmnidreamConfig, path: str | Path) -> None:
    """Save configuration to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_to_dict(cfg), f, indent=2, sort_keys=False)


def load_config_json(path: str | Path) -> OmnidreamConfig:
    """Load configuration from a JSON file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return _from_dict(d)
