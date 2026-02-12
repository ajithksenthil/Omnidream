"""Tests for trajectory planning module (trajectory.py).

Covers:
    - Interpolation algorithms (linear, geodesic, Jacobian-steered)
    - Constraint checking and enforcement
    - H-theorem enforcement
    - Waypoint resolution
    - End-to-end trajectory generation
    - Pipeline Stage 13 integration
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trajectory import (
    PropertyGoal,
    TrajectoryPoint,
    TrajectoryResult,
    TrajectorySpec,
    check_point_safety,
    compute_energy_flow,
    compute_h_flow,
    enforce_safety_along_path,
    interpolate_linear,
    interpolate_jacobian_steered,
    interpolate_segments,
    plan_trajectory,
    print_trajectory_summary,
    resolve_waypoints,
    save_trajectory,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_basis(n_points: int = 200, n_coils: int = 8) -> np.ndarray:
    """Create a synthetic basis matrix with realistic structure."""
    rng = np.random.default_rng(42)
    basis = rng.standard_normal((n_points, n_coils)) * 0.1
    # Boost target region (centroid of array)
    basis[n_points // 2, :] *= 5.0
    return basis


def _make_L_matrix(n_coils: int = 8) -> np.ndarray:
    """Create a synthetic inductance matrix."""
    L_self = 5e-6
    L = np.eye(n_coils) * L_self
    for i in range(n_coils):
        for j in range(i + 1, n_coils):
            M = L_self * 0.01 * np.exp(-abs(i - j) / 3.0)
            L[i, j] = M
            L[j, i] = M
    return L


def _make_plant_and_worlds(n_coils: int = 8, n_worlds: int = 10):
    """Create a TMSPlant and synthetic WorldState objects for testing."""
    from config import paper_baseline
    from control_framework import TMSPlant, StimulationMode
    from cp_bridge import WorldState, CPBridgeConfig, compute_total_energy
    from cp_bridge import compute_transfer_entropy_matrix, compute_phases_from_groups

    basis = _make_basis(n_coils=n_coils)
    target_idx = basis.shape[0] // 2
    surface_indices = np.arange(0, 20)
    config = paper_baseline()

    plant = TMSPlant(
        basis_matrix=basis,
        target_idx=target_idx,
        surface_indices=surface_indices,
        config=config,
        mode=StimulationMode.TI,
    )

    # Build worlds from random amplitude vectors
    rng = np.random.default_rng(123)
    N = n_coils
    group = np.zeros(N)
    group[N // 2:] = 1.0
    f1 = config.ti.freq_carrier_hz
    f2 = f1 + config.ti.delta_freq_default_hz
    phases = compute_phases_from_groups(group, f1, f2, t=0.0)
    eta_field = np.where(group > 0.5, 1.0, 0.1)
    te_matrix = compute_transfer_entropy_matrix(plant.L)
    cp_config = CPBridgeConfig()

    worlds = []
    for k in range(n_worlds):
        amps = rng.uniform(0.1, 0.9, N)
        y = plant.forward_from_params(amps)
        E_dict = compute_total_energy(
            plant, amps, te_matrix, phases, basis, target_idx,
            eta_field, group, cp_config=cp_config)
        worlds.append(WorldState(
            amplitudes=amps,
            outputs=y,
            energy=E_dict["E_total"],
            phi=E_dict["phi_collective"],
            probability=1.0 / n_worlds,
            world_id=k,
        ))

    return plant, worlds, basis, target_idx, surface_indices, config


# =====================================================================
# Test Interpolation
# =====================================================================

class TestLinearInterpolation(unittest.TestCase):
    """Tests for interpolate_linear()."""

    def test_endpoints_match(self):
        """Linear interpolation passes through start and end."""
        a = np.array([0.0, 0.5, 1.0])
        b = np.array([1.0, 0.0, 0.5])
        path = interpolate_linear(a, b, n_steps=11)
        np.testing.assert_allclose(path[0], a)
        np.testing.assert_allclose(path[-1], b)

    def test_correct_length(self):
        """Produces the requested number of steps."""
        a = np.zeros(5)
        b = np.ones(5)
        path = interpolate_linear(a, b, n_steps=20)
        self.assertEqual(len(path), 20)

    def test_midpoint(self):
        """Midpoint is the average of endpoints."""
        a = np.array([0.0, 0.0])
        b = np.array([2.0, 4.0])
        path = interpolate_linear(a, b, n_steps=3)
        np.testing.assert_allclose(path[1], [1.0, 2.0])

    def test_single_step(self):
        """n_steps=1 gives only the start point."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        path = interpolate_linear(a, b, n_steps=1)
        self.assertEqual(len(path), 1)
        np.testing.assert_allclose(path[0], a)


class TestJacobianSteered(unittest.TestCase):
    """Tests for interpolate_jacobian_steered()."""

    def test_endpoints(self):
        """Jacobian-steered path starts and ends at correct points."""
        plant, worlds, basis, target_idx, surface_indices, config = _make_plant_and_worlds()
        a = worlds[0].amplitudes
        b = worlds[1].amplitudes
        path = interpolate_jacobian_steered(a, b, n_steps=10, plant=plant)
        np.testing.assert_allclose(path[0], a)
        np.testing.assert_allclose(path[-1], b)

    def test_correct_length(self):
        """Produces the requested number of steps."""
        plant, worlds, basis, target_idx, surface_indices, config = _make_plant_and_worlds()
        a = worlds[0].amplitudes
        b = worlds[1].amplitudes
        path = interpolate_jacobian_steered(a, b, n_steps=15, plant=plant)
        self.assertEqual(len(path), 15)

    def test_non_negative(self):
        """All amplitudes remain non-negative."""
        plant, worlds, basis, target_idx, surface_indices, config = _make_plant_and_worlds()
        a = worlds[0].amplitudes
        b = worlds[1].amplitudes
        path = interpolate_jacobian_steered(a, b, n_steps=20, plant=plant)
        for p in path:
            self.assertTrue(np.all(p >= 0), f"Negative amplitude found: {p.min()}")


class TestMultiSegment(unittest.TestCase):
    """Tests for interpolate_segments()."""

    def test_correct_total_length(self):
        """Multi-segment produces correct total number of points."""
        wps = [np.zeros(4), np.ones(4) * 0.5, np.ones(4)]
        path, wp_idx = interpolate_segments(wps, n_steps_per_segment=10, method="linear")
        # 2 segments × 10 steps, but shared midpoint is deduplicated: 10 + 9 = 19
        self.assertEqual(len(path), 19)

    def test_waypoint_indices(self):
        """Waypoint indices correctly locate original waypoints."""
        wps = [np.zeros(4), np.ones(4) * 0.5, np.ones(4)]
        path, wp_idx = interpolate_segments(wps, n_steps_per_segment=10, method="linear")
        self.assertEqual(len(wp_idx), 3)
        np.testing.assert_allclose(path[wp_idx[0]], wps[0])
        np.testing.assert_allclose(path[wp_idx[1]], wps[1])
        np.testing.assert_allclose(path[wp_idx[2]], wps[2])


# =====================================================================
# Test Constraints
# =====================================================================

class TestConstraints(unittest.TestCase):
    """Tests for constraint checking and enforcement."""

    def test_safe_point_margins_computed(self):
        """Constraint check returns correct margin structure."""
        plant, worlds, basis, target_idx, surface_indices, config = _make_plant_and_worlds()
        amps = np.ones(8) * 0.1
        check = check_point_safety(plant, amps, config)
        # May or may not be satisfied depending on synthetic basis scale,
        # but margins must always be computed.
        self.assertIn("sar", check["margins"])
        self.assertIn("thermal", check["margins"])
        self.assertIn("current", check["margins"])
        self.assertIn("voltage", check["margins"])
        # Current margin should be positive for small amps
        self.assertGreater(check["margins"]["current"], 0)

    def test_overcurrent_detected(self):
        """Large amplitudes violate current constraint."""
        plant, worlds, basis, target_idx, surface_indices, config = _make_plant_and_worlds()
        amps = np.ones(8) * 10.0  # Way above I_max = 5.0
        check = check_point_safety(plant, amps, config)
        self.assertFalse(check["satisfied"])
        self.assertLess(check["margins"]["current"], 0)

    def test_enforcement_reduces_amplitudes(self):
        """enforce_safety_along_path scales down violating amplitudes."""
        plant, worlds, basis, target_idx, surface_indices, config = _make_plant_and_worlds()
        path = [np.ones(8) * 10.0, np.ones(8) * 10.0]  # All violating
        corrected, violations = enforce_safety_along_path(path, plant, config)
        # After enforcement, amplitudes should be smaller than original
        for a_orig, a_corr in zip(path, corrected):
            self.assertLess(np.max(a_corr), np.max(a_orig))

    def test_zero_amps_safe(self):
        """Zero amplitudes should not have current violations."""
        plant, worlds, basis, target_idx, surface_indices, config = _make_plant_and_worlds()
        path = [np.zeros(8), np.zeros(8)]
        corrected, violations = enforce_safety_along_path(path, plant, config)
        # Zero amps produce zero SAR and zero current, so no violations
        np.testing.assert_allclose(corrected[0], path[0])
        np.testing.assert_allclose(corrected[1], path[1])


# =====================================================================
# Test H-Theorem
# =====================================================================

class TestHTheorem(unittest.TestCase):
    """Tests for H-theorem enforcement."""

    def test_monotone_no_violations(self):
        """Monotonically decreasing energy has no violations."""
        energies = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        dE_dt, violations = compute_energy_flow(energies)
        self.assertEqual(len(violations), 0)

    def test_uphill_detected(self):
        """Energy increase is flagged as violation."""
        energies = np.array([5.0, 3.0, 7.0, 4.0])  # 3→7 is uphill
        dE_dt, violations = compute_energy_flow(energies)
        self.assertIn(1, violations)  # Index 1 (transition from 3→7)

    def test_constant_energy_no_violations(self):
        """Flat energy profile has no violations."""
        energies = np.ones(10) * 5.0
        dE_dt, violations = compute_energy_flow(energies)
        self.assertEqual(len(violations), 0)

    def test_h_flow_detects_uphill(self):
        """dH/dt > 0 is flagged for cognitive H-theorem checks."""
        h_vals = np.array([5.0, 4.0, 4.5, 4.2])
        dH_dt, violations = compute_h_flow(h_vals)
        self.assertIn(1, violations)  # transition 4.0 -> 4.5


# =====================================================================
# Test Waypoint Resolution
# =====================================================================

class TestWaypointResolution(unittest.TestCase):
    """Tests for resolve_waypoints()."""

    def test_world_sequence(self):
        """World sequence resolves to correct amplitudes."""
        plant, worlds, *_ = _make_plant_and_worlds(n_worlds=5)
        spec = TrajectorySpec(spec_type="world_sequence", world_ids=[0, 2, 4])
        waypoints = resolve_waypoints(spec, worlds)
        self.assertEqual(len(waypoints), 3)
        np.testing.assert_allclose(waypoints[0], worlds[0].amplitudes)
        np.testing.assert_allclose(waypoints[2], worlds[4].amplitudes)

    def test_property_sequence(self):
        """Property sequence finds closest worlds."""
        plant, worlds, *_ = _make_plant_and_worlds(n_worlds=10)
        spec = TrajectorySpec(
            spec_type="property_sequence",
            property_targets=[
                PropertyGoal(phi_target=worlds[0].phi),
                PropertyGoal(phi_target=worlds[5].phi),
            ],
        )
        waypoints = resolve_waypoints(spec, worlds)
        self.assertEqual(len(waypoints), 2)

    def test_energy_descent(self):
        """Energy descent resolves start and end."""
        plant, worlds, *_ = _make_plant_and_worlds(n_worlds=5)
        spec = TrajectorySpec(
            spec_type="energy_descent",
            start_world_id=0,
            end_world_id=4,
        )
        waypoints = resolve_waypoints(spec, worlds)
        self.assertGreaterEqual(len(waypoints), 2)
        np.testing.assert_allclose(waypoints[0], worlds[0].amplitudes)
        np.testing.assert_allclose(waypoints[-1], worlds[4].amplitudes)

    def test_invalid_spec_raises(self):
        """Invalid spec_type raises ValueError."""
        plant, worlds, *_ = _make_plant_and_worlds()
        spec = TrajectorySpec(spec_type="invalid_type")
        with self.assertRaises(ValueError):
            resolve_waypoints(spec, worlds)


# =====================================================================
# Test End-to-End Trajectory Generation
# =====================================================================

class TestTrajectoryGeneration(unittest.TestCase):
    """Tests for plan_trajectory()."""

    def test_world_sequence_e2e(self):
        """End-to-end world_sequence trajectory generation."""
        plant, worlds, basis, target_idx, surface_indices, config = _make_plant_and_worlds()

        spec = TrajectorySpec(
            spec_type="world_sequence",
            world_ids=[0, 3, 7],
            n_steps_per_segment=10,
            interpolation="linear",
            safety_mode="permissive",
            h_theorem_enforce=False,
            name="test_e2e",
        )

        result = plan_trajectory(
            spec=spec,
            worlds=worlds,
            plant=plant,
            basis_matrix=basis,
            target_idx=target_idx,
            surface_indices=surface_indices,
            L_matrix=plant.L,
            config=config,
        )

        self.assertIsInstance(result, TrajectoryResult)
        self.assertEqual(result.n_timesteps, 19)  # 10 + 9 (deduped midpoint)
        self.assertEqual(len(result.points), 19)
        self.assertEqual(result.amplitudes_array.shape[1], 8)  # N coils
        self.assertEqual(result.outputs_array.shape[1], 5)

    def test_result_arrays_consistent(self):
        """Result arrays have consistent shapes."""
        plant, worlds, basis, target_idx, surface_indices, config = _make_plant_and_worlds()

        spec = TrajectorySpec(
            spec_type="world_sequence",
            world_ids=[0, 5],
            n_steps_per_segment=15,
            interpolation="linear",
            safety_mode="permissive",
            h_theorem_enforce=False,
        )

        result = plan_trajectory(
            spec=spec, worlds=worlds, plant=plant,
            basis_matrix=basis, target_idx=target_idx,
            surface_indices=surface_indices, L_matrix=plant.L, config=config,
        )

        T = result.n_timesteps
        self.assertEqual(len(result.time_array), T)
        self.assertEqual(result.amplitudes_array.shape[0], T)
        self.assertEqual(result.outputs_array.shape[0], T)
        self.assertEqual(len(result.energy_array), T)
        self.assertEqual(len(result.phi_array), T)
        self.assertEqual(len(result.sync_array), T)

    def test_trajectory_summary_prints(self):
        """print_trajectory_summary produces non-empty string."""
        plant, worlds, basis, target_idx, surface_indices, config = _make_plant_and_worlds()

        spec = TrajectorySpec(
            spec_type="world_sequence",
            world_ids=[0, 5],
            n_steps_per_segment=5,
            interpolation="linear",
            safety_mode="permissive",
            h_theorem_enforce=False,
        )

        result = plan_trajectory(
            spec=spec, worlds=worlds, plant=plant,
            basis_matrix=basis, target_idx=target_idx,
            surface_indices=surface_indices, L_matrix=plant.L, config=config,
        )

        summary = print_trajectory_summary(result)
        self.assertIn("Trajectory", summary)
        self.assertIn("Energy", summary)
        self.assertIn("Φ", summary)

    def test_energy_descent_mode(self):
        """Energy descent trajectory produces valid result."""
        plant, worlds, basis, target_idx, surface_indices, config = _make_plant_and_worlds()

        spec = TrajectorySpec(
            spec_type="energy_descent",
            start_world_id=0,
            end_world_id=5,
            n_steps_per_segment=10,
            interpolation="linear",
            safety_mode="permissive",
            h_theorem_enforce=False,
        )

        result = plan_trajectory(
            spec=spec, worlds=worlds, plant=plant,
            basis_matrix=basis, target_idx=target_idx,
            surface_indices=surface_indices, L_matrix=plant.L, config=config,
        )

        self.assertIsInstance(result, TrajectoryResult)
        self.assertEqual(result.n_timesteps, 10)


# =====================================================================
# Test Pipeline Stage 13 Integration
# =====================================================================

class TestPipelineStage13(unittest.TestCase):
    """Integration tests for Stage 13 within the pipeline context."""

    def _run_stage13(self, mode: str):
        """Helper: simulate Stage 13 for a given mode."""
        from config import paper_baseline
        from control_framework import TMSPlant, StimulationMode
        from sensitivity import run_sensitivity_analysis
        from cp_bridge import run_cp_bridge_analysis

        n_coils = 8
        basis = _make_basis(n_coils=n_coils)
        target_idx = basis.shape[0] // 2
        surface_indices = np.arange(0, 20)
        config = paper_baseline()

        mode_map = {"TI": StimulationMode.TI, "NTS": StimulationMode.NTS}
        plant = TMSPlant(
            basis_matrix=basis,
            target_idx=target_idx,
            surface_indices=surface_indices,
            config=config,
            mode=mode_map.get(mode, StimulationMode.TI),
        )

        # Run sensitivity (Stage 11) for Pareto front
        sens_results = run_sensitivity_analysis(
            basis_matrix=basis,
            target_idx=target_idx,
            surface_indices=surface_indices,
            config=config,
            mode=mode,
            alpha_max=1.0,
            n_reachable_samples=50,
            n_pareto_weights=20,
        )

        # Get Pareto results
        if mode.upper() == "TI":
            pareto_results = sens_results.get("pareto_ti")
        else:
            pareto_results = sens_results.get("pareto_nts")

        # Run CP Bridge (Stage 12)
        cp_results = run_cp_bridge_analysis(
            basis_matrix=basis,
            target_idx=target_idx,
            surface_indices=surface_indices,
            config=config,
            mode=mode,
            pareto_results=pareto_results,
        )

        worlds = cp_results.get("worlds", [])
        self.assertGreaterEqual(len(worlds), 3,
                                f"Need ≥3 worlds, got {len(worlds)}")

        # Run Stage 13: Trajectory Planning
        sorted_ids = sorted(range(len(worlds)), key=lambda i: worlds[i].energy)
        n_w = len(sorted_ids)
        waypoint_ids = [sorted_ids[0], sorted_ids[n_w // 2], sorted_ids[-1]]

        spec = TrajectorySpec(
            spec_type="world_sequence",
            world_ids=waypoint_ids,
            n_steps_per_segment=10,
            interpolation="linear",
            safety_mode="permissive",
            h_theorem_enforce=False,
            name=f"{mode}_stage13_test",
        )

        result = plan_trajectory(
            spec=spec,
            worlds=worlds,
            plant=plant,
            basis_matrix=basis,
            target_idx=target_idx,
            surface_indices=surface_indices,
            L_matrix=plant.L,
            config=config,
        )

        # Validate
        self.assertIsInstance(result, TrajectoryResult)
        self.assertGreater(result.n_timesteps, 0)
        self.assertGreater(result.energy_barrier, 0)

        summary = print_trajectory_summary(result)
        self.assertIn(mode, summary)

        return result

    def test_stage13_ti(self):
        """Stage 13 runs successfully in TI mode."""
        result = self._run_stage13("TI")
        self.assertEqual(result.spec.name, "TI_stage13_test")

    def test_stage13_nts(self):
        """Stage 13 runs successfully in NTS mode."""
        result = self._run_stage13("NTS")
        self.assertEqual(result.spec.name, "NTS_stage13_test")


if __name__ == "__main__":
    unittest.main()
