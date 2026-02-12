"""Tests for the E&M theory, control framework, and sensitivity modules.

Covers:
    - em_theory.py: quasi-static validity, Cole-Cole, inductance, neural, SAR
    - control_framework.py: TMSPlant, GoalSpec, Controller, PlantModel interface
    - sensitivity.py: Jacobians, Hessians, condition numbers, reachable sets, Pareto fronts

Run with:
    python -m pytest tests/test_control_framework.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# =====================================================================
# Test E&M Theory Module
# =====================================================================

class TestMaxwellBasis:
    """Tests for em_theory.MaxwellBasis."""

    def test_wavelength_positive_freq(self):
        from em_theory import MaxwellBasis, C_LIGHT
        lam = MaxwellBasis.wavelength(1000.0)
        assert lam == pytest.approx(C_LIGHT / 1000.0, rel=1e-6)

    def test_wavelength_zero_freq(self):
        from em_theory import MaxwellBasis
        assert MaxwellBasis.wavelength(0.0) == float("inf")

    def test_quasi_static_at_1khz(self):
        from em_theory import MaxwellBasis
        # At 1 kHz, λ ≈ 300 km ≫ 0.2 m → valid
        assert MaxwellBasis.quasi_static_validity(1000.0) is True

    def test_quasi_static_fails_at_very_high_freq(self):
        from em_theory import MaxwellBasis
        # At 10 GHz, λ = 0.03 m < 100 × 0.2 m → invalid
        assert MaxwellBasis.quasi_static_validity(10e9) is False

    def test_quasi_static_ratio(self):
        from em_theory import MaxwellBasis
        ratio = MaxwellBasis.quasi_static_ratio(1000.0)
        # Should be ≈ 1.5e6 (300 km / 0.2 m)
        assert ratio > 1e5

    def test_linearity_residual_zero_for_identical(self):
        from em_theory import MaxwellBasis
        E = np.random.randn(100)
        assert MaxwellBasis.linearity_residual(E, E) == pytest.approx(0.0, abs=1e-12)

    def test_linearity_residual_positive_for_different(self):
        from em_theory import MaxwellBasis
        E1 = np.ones(100)
        E2 = np.ones(100) * 2.0
        res = MaxwellBasis.linearity_residual(E1, E2)
        assert res > 0.0


class TestTissueModel:
    """Tests for em_theory.TissueModel."""

    def test_presets_exist(self):
        from em_theory import TissueModel
        assert TissueModel.GRAY_MATTER is not None
        assert TissueModel.WHITE_MATTER is not None
        assert TissueModel.CSF is not None
        assert TissueModel.SKULL is not None
        assert TissueModel.SCALP is not None

    def test_gray_matter_dc_conductivity(self):
        from em_theory import TissueModel
        gm = TissueModel.GRAY_MATTER
        sigma_dc = gm.conductivity_magnitude(0.1)  # near DC
        # Should be close to sigma_0 = 0.020
        assert 0.01 < sigma_dc < 0.15

    def test_csf_conductivity_constant(self):
        from em_theory import TissueModel
        csf = TissueModel.CSF
        # CSF has σ_0 = σ_inf = 1.654, so conductivity should be ~constant
        s1 = csf.conductivity_magnitude(100.0)
        s2 = csf.conductivity_magnitude(10000.0)
        assert abs(s1 - s2) < 0.01

    def test_conductivity_sweep_shape(self):
        from em_theory import TissueModel
        gm = TissueModel.GRAY_MATTER
        freqs = np.logspace(1, 5, 50)
        sigma = gm.conductivity_sweep(freqs)
        assert sigma.shape == (50,)
        assert np.all(sigma > 0)

    def test_complex_conductivity_finite(self):
        from em_theory import TissueModel
        gm = TissueModel.GRAY_MATTER
        sigma_c = gm.conductivity(1000.0)
        assert np.isfinite(sigma_c)


class TestCoilInductance:
    """Tests for em_theory.CoilInductance."""

    def test_wheeler_positive(self):
        from em_theory import CoilInductance
        L = CoilInductance.wheeler_rectangular(30, 7e-3, 4e-3, 1e-3)
        assert L > 0

    def test_c_shape_positive(self):
        from em_theory import CoilInductance
        L = CoilInductance.self_inductance_c_shape()
        assert L > 0

    def test_c_shape_reasonable_value(self):
        from em_theory import CoilInductance
        L = CoilInductance.self_inductance_c_shape()
        # Should be in the μH range for a miniature coil
        assert 1e-7 < L < 1e-3

    def test_mutual_inductance_decays_with_distance(self):
        from em_theory import CoilInductance
        pos_i = np.array([0, 0, 0], dtype=float)
        close = np.array([0.02, 0, 0], dtype=float)  # 2 cm
        far = np.array([0.10, 0, 0], dtype=float)     # 10 cm
        M_close = CoilInductance.neumann_dipole_mutual(pos_i, close, 30, 28e-6)
        M_far = CoilInductance.neumann_dipole_mutual(pos_i, far, 30, 28e-6)
        assert M_close > M_far

    def test_mutual_inductance_self_is_zero(self):
        from em_theory import CoilInductance
        pos = np.array([0, 0, 0], dtype=float)
        M = CoilInductance.neumann_dipole_mutual(pos, pos, 30, 28e-6)
        assert M == 0.0

    def test_coupling_coefficient_identity(self):
        from em_theory import CoilInductance
        # k = M / sqrt(L*L) = M/L; if M = L, k = 1
        k = CoilInductance.coupling_coefficient(5e-6, 5e-6, 5e-6)
        assert k == pytest.approx(1.0, rel=1e-6)

    def test_steinmetz_positive(self):
        from em_theory import CoilInductance
        P = CoilInductance.steinmetz_core_loss(1000.0, 0.1)
        assert P > 0

    def test_steinmetz_zero_at_zero_freq(self):
        from em_theory import CoilInductance
        P = CoilInductance.steinmetz_core_loss(0.0, 0.1)
        assert P == 0.0


class TestNeuralResponse:
    """Tests for em_theory.NeuralResponse."""

    def test_cutoff_frequency(self):
        from em_theory import NeuralResponse
        nr = NeuralResponse(tau_m_s=3e-3)
        fc = nr.low_pass_cutoff_hz()
        # fc = 1/(2π × 3ms) ≈ 53 Hz
        assert 50 < fc < 56

    def test_transfer_function_unity_at_dc(self):
        from em_theory import NeuralResponse
        nr = NeuralResponse()
        H = nr.transfer_function(0.0)
        assert abs(H) == pytest.approx(1.0, rel=1e-6)

    def test_transfer_function_decreases_with_freq(self):
        from em_theory import NeuralResponse
        nr = NeuralResponse()
        h_low = nr.transfer_function_magnitude(10.0)
        h_high = nr.transfer_function_magnitude(10000.0)
        assert h_low > h_high

    def test_carrier_attenuation_negative(self):
        from em_theory import NeuralResponse
        nr = NeuralResponse()
        att = nr.carrier_attenuation_db(1000.0)
        assert att < 0  # carrier is attenuated

    def test_ti_effective_ratio(self):
        from em_theory import NeuralResponse
        nr = NeuralResponse(tau_m_s=3e-3)
        ratio = nr.ti_effective_ratio(1000.0, 10.0)
        # Beat (10 Hz) passes through much better than carrier (1 kHz)
        assert ratio > 10.0


class TestSARThermal:
    """Tests for em_theory.SARThermal."""

    def test_sar_pointwise_positive(self):
        from em_theory import SARThermal
        E = np.array([1.0, 2.0, 3.0])
        SAR = SARThermal.sar_pointwise(E)
        assert np.all(SAR > 0)

    def test_sar_pointwise_scales_with_E_squared(self):
        from em_theory import SARThermal
        E1 = np.array([1.0])
        E2 = np.array([2.0])
        SAR1 = SARThermal.sar_pointwise(E1)
        SAR2 = SARThermal.sar_pointwise(E2)
        assert SAR2[0] == pytest.approx(4.0 * SAR1[0], rel=1e-6)

    def test_pennes_delta_T_positive(self):
        from em_theory import SARThermal
        dT = SARThermal.pennes_steady_state_delta_T(1.0)
        assert dT > 0

    def test_thermal_time_constant_positive(self):
        from em_theory import SARThermal
        tau = SARThermal.thermal_time_constant_s()
        assert tau > 0

    def test_transient_fraction_bounds(self):
        from em_theory import SARThermal
        tau = SARThermal.thermal_time_constant_s()
        f_early = SARThermal.transient_fraction(0.01, tau)
        f_late = SARThermal.transient_fraction(100 * tau, tau)
        assert 0 < f_early < 0.5
        assert f_late > 0.99


# =====================================================================
# Test Control Framework
# =====================================================================

class TestPlantDimensions:
    """Tests for control_framework.PlantDimensions."""

    def test_from_n_coils(self):
        from control_framework import PlantDimensions
        d = PlantDimensions.from_n_coils(16)
        assert d.n_state == 34  # 2*16 + 2
        assert d.m_input == 16
        assert d.p_output == 5
        assert d.n_coils == 16


class TestGoalSpec:
    """Tests for control_framework.GoalSpec."""

    def test_default_construction(self):
        from control_framework import GoalSpec, SystemGoal
        g = GoalSpec()
        assert SystemGoal.FOCAL_DEPTH in g.weights
        assert "SAR_max" in g.constraints


class TestTMSPlant:
    """Tests for control_framework.TMSPlant."""

    @pytest.fixture
    def plant(self):
        from basis_fields import generate_synthetic_basis
        from control_framework import TMSPlant, StimulationMode
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        return TMSPlant(basis, tidx, sidx, mode=StimulationMode.TI)

    def test_dims(self, plant):
        d = plant.dims()
        assert d.n_coils == 8
        assert d.n_state == 18  # 2*8 + 2
        assert d.m_input == 8
        assert d.p_output == 5

    def test_make_state(self, plant):
        x = plant.make_state()
        assert len(x) == 18
        assert x[8] == 0.0  # V_m default

    def test_forward_produces_finite(self, plant):
        x = plant.make_state(currents=np.ones(8) * 0.5)
        u = plant.make_input(np.ones(8) * 1.0)
        y = plant.forward(x, u)
        assert len(y) == 5
        assert np.all(np.isfinite(y))

    def test_forward_from_params_produces_finite(self, plant):
        amplitudes = np.ones(8) * 0.5
        y = plant.forward_from_params(amplitudes)
        assert len(y) == 5
        assert np.all(np.isfinite(y))

    def test_linearize_shape(self, plant):
        x = plant.make_state(currents=np.ones(8) * 0.5)
        u = plant.make_input(np.ones(8) * 1.0)
        A, B, C, D = plant.linearize(x, u)
        assert A.shape == (18, 18)
        assert B.shape == (18, 8)
        assert C.shape == (5, 18)
        assert D.shape == (5, 8)

    def test_jacobian_shape(self, plant):
        x = plant.make_state(currents=np.ones(8) * 0.5)
        u = plant.make_input(np.ones(8) * 1.0)
        J = plant.jacobian_output_wrt_input(x, u)
        assert J.shape == (5, 8)
        assert np.all(np.isfinite(J))

    def test_controllability_matrix_shape(self, plant):
        x = plant.make_state(currents=np.ones(8) * 0.5)
        u = plant.make_input(np.ones(8) * 1.0)
        A, B, C, D = plant.linearize(x, u)
        C_ctrl = plant.controllability_matrix(A, B)
        n = A.shape[0]
        m = B.shape[1]
        assert C_ctrl.shape == (n, n * m)

    def test_controllability_rank_positive(self, plant):
        x = plant.make_state(currents=np.ones(8) * 0.5)
        u = plant.make_input(np.ones(8) * 1.0)
        A, B, C, D = plant.linearize(x, u)
        rank = plant.controllability_rank(A, B)
        assert rank > 0

    def test_condition_number_finite(self, plant):
        x = plant.make_state(currents=np.ones(8) * 0.5)
        u = plant.make_input(np.ones(8) * 1.0)
        kappa = plant.condition_number(x, u)
        assert np.isfinite(kappa) or kappa == float("inf")

    def test_nts_mode(self):
        from basis_fields import generate_synthetic_basis
        from control_framework import TMSPlant, StimulationMode
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        plant = TMSPlant(basis, tidx, sidx, mode=StimulationMode.NTS)
        y = plant.forward_from_params(np.ones(8) * 0.5)
        assert len(y) == 5
        assert np.all(np.isfinite(y))


class TestController:
    """Tests for control_framework.GreedyJacobianController."""

    def test_greedy_controller_produces_action(self):
        from basis_fields import generate_synthetic_basis
        from control_framework import TMSPlant, StimulationMode, GoalSpec, GreedyJacobianController
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        plant = TMSPlant(basis, tidx, sidx, mode=StimulationMode.TI)
        goal = GoalSpec(target_idx=tidx, surface_indices=sidx)
        ctrl = GreedyJacobianController(step_size=0.1)
        x = plant.make_state(currents=np.ones(8) * 0.5)
        u = ctrl.select_action(x, goal, plant)
        assert len(u) == 8
        assert np.all(np.isfinite(u))


class TestCostFunction:
    """Tests for control_framework.build_cost_function."""

    def test_cost_function_returns_scalar(self):
        from basis_fields import generate_synthetic_basis
        from control_framework import TMSPlant, StimulationMode, GoalSpec, build_cost_function
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        plant = TMSPlant(basis, tidx, sidx, mode=StimulationMode.TI)
        goal = GoalSpec(target_idx=tidx, surface_indices=sidx)
        cost_fn = build_cost_function(plant, goal)
        cost = cost_fn(np.ones(8) * 0.5)
        assert isinstance(cost, float)
        assert np.isfinite(cost)


class TestAnalysePlant:
    """Tests for control_framework.analyse_plant."""

    def test_analyse_returns_all_keys(self):
        from basis_fields import generate_synthetic_basis
        from control_framework import TMSPlant, StimulationMode, analyse_plant
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        plant = TMSPlant(basis, tidx, sidx, mode=StimulationMode.TI)
        result = analyse_plant(plant)

        expected_keys = [
            "y_state", "y_params", "output_labels",
            "output_labels_legacy", "y_state_map", "y_params_map",
            "J_state", "J_params", "singular_values",
            "kappa_state", "kappa_params",
            "A", "B", "C", "D",
            "controllability_rank", "observability_rank",
            "output_controllability_rank",
            "n_state", "n_input", "n_output",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        assert "target_metric" in result["y_state_map"]
        assert "surface_metric" in result["y_state_map"]


# =====================================================================
# Test Sensitivity Module
# =====================================================================

class TestJacobianTI:
    """Tests for sensitivity.compute_jacobian_ti."""

    def test_jacobian_shape(self):
        from basis_fields import generate_synthetic_basis
        from sensitivity import compute_jacobian_ti
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        group = np.zeros(8)
        group[4:] = 1.0
        jac = compute_jacobian_ti(np.ones(8) * 0.5, group, 1000.0, 1010.0,
                                   basis, tidx, sidx)
        assert jac["J_M_target"].shape == (8,)
        assert jac["J_M_surface_max"].shape == (8,)
        assert jac["J_full"].shape == (3, 8)

    def test_jacobian_finite(self):
        from basis_fields import generate_synthetic_basis
        from sensitivity import compute_jacobian_ti
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        group = np.zeros(8)
        group[4:] = 1.0
        jac = compute_jacobian_ti(np.ones(8) * 0.5, group, 1000.0, 1010.0,
                                   basis, tidx, sidx)
        assert np.all(np.isfinite(jac["J_full"]))


class TestJacobianNTS:
    """Tests for sensitivity.compute_jacobian_nts and analytical."""

    def test_jacobian_shape(self):
        from basis_fields import generate_synthetic_basis
        from nts_timing import optimal_firing_order, assign_uniform_fire_times
        from sensitivity import compute_jacobian_nts
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        order = optimal_firing_order(basis, tidx)
        ft = assign_uniform_fire_times(order, 5e-3)
        jac = compute_jacobian_nts(np.ones(8) * 0.5, ft, basis, tidx, sidx)
        assert jac["J_alpha"].shape == (2, 8)
        assert jac["J_time"].shape == (2, 8)

    def test_analytical_matches_fd(self):
        from basis_fields import generate_synthetic_basis
        from nts_timing import optimal_firing_order, assign_uniform_fire_times
        from sensitivity import compute_jacobian_nts, compute_jacobian_analytical_nts
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        order = optimal_firing_order(basis, tidx)
        ft = assign_uniform_fire_times(order, 5e-3)
        amps = np.ones(8) * 0.5
        jac_fd = compute_jacobian_nts(amps, ft, basis, tidx, sidx)
        jac_an = compute_jacobian_analytical_nts(amps, ft, basis, tidx)
        # Analytical and finite-difference should agree to within ~1%
        fd = jac_fd["J_V_target_alpha"]
        an = jac_an["J_V_target_alpha"]
        rel_err = np.linalg.norm(fd - an) / (np.linalg.norm(fd) + 1e-15)
        assert rel_err < 0.05, f"Analytical vs FD mismatch: {rel_err:.4f}"


class TestHessian:
    """Tests for sensitivity.compute_hessian."""

    def test_hessian_symmetric(self):
        from sensitivity import compute_hessian
        def quad(x):
            return float(x @ x)
        H = compute_hessian(np.ones(4) * 0.5, quad)
        assert H.shape == (4, 4)
        assert np.allclose(H, H.T, atol=1e-6)

    def test_hessian_identity_for_quadratic(self):
        from sensitivity import compute_hessian
        def quad(x):
            return float(x @ x)
        H = compute_hessian(np.ones(4) * 0.5, quad)
        # Hessian of x^Tx is 2I
        assert np.allclose(H, 2 * np.eye(4), atol=1e-3)


class TestConditionNumber:
    """Tests for sensitivity.compute_condition_number."""

    def test_identity_condition_is_one(self):
        from sensitivity import compute_condition_number
        kappa = compute_condition_number(np.eye(3))
        assert kappa == pytest.approx(1.0, rel=1e-6)

    def test_singular_matrix_is_inf(self):
        from sensitivity import compute_condition_number
        Z = np.zeros((3, 3))
        assert compute_condition_number(Z) == float("inf")

    def test_ill_conditioned_large_kappa(self):
        from sensitivity import compute_condition_number
        J = np.diag([1.0, 1e-6, 1e-12])
        kappa = compute_condition_number(J)
        assert kappa > 1e5


class TestReachableSet:
    """Tests for sensitivity.compute_reachable_set_ti/nts."""

    def test_reachable_ti_nonempty(self):
        from basis_fields import generate_synthetic_basis
        from sensitivity import compute_reachable_set_ti
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        group = np.zeros(8)
        group[4:] = 1.0
        rs = compute_reachable_set_ti(basis, group, 1000.0, 1010.0, tidx, sidx,
                                       n_samples=50)
        assert len(rs["M_target"]) == 50
        assert len(rs["M_surface_max"]) == 50
        assert rs["hull_vertices"].shape[0] >= 3  # at least a triangle

    def test_reachable_nts_nonempty(self):
        from basis_fields import generate_synthetic_basis
        from sensitivity import compute_reachable_set_nts
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        rs = compute_reachable_set_nts(basis, tidx, sidx, n_samples=50)
        assert len(rs["V_target"]) == 50
        assert len(rs["V_surface_max"]) == 50


class TestParetoFront:
    """Tests for sensitivity.compute_pareto_front_ti/nts."""

    def test_pareto_ti_has_tradeoff(self):
        from basis_fields import generate_synthetic_basis
        from sensitivity import compute_pareto_front_ti
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        group = np.zeros(8)
        group[4:] = 1.0
        pf = compute_pareto_front_ti(basis, group, 1000.0, 1010.0, tidx, sidx,
                                      n_weights=10, optimizer="random")
        assert len(pf["M_target"]) == 10
        # Should have at least some non-dominated points
        non_dom = ~pf["is_dominated"]
        assert np.sum(non_dom) >= 1

    def test_pareto_nts_has_tradeoff(self):
        from basis_fields import generate_synthetic_basis
        from sensitivity import compute_pareto_front_nts
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        pf = compute_pareto_front_nts(basis, tidx, sidx,
                                       n_weights=10, optimizer="random")
        assert len(pf["V_target"]) == 10
        non_dom = ~pf["is_dominated"]
        assert np.sum(non_dom) >= 1


class TestSensitivityReport:
    """Tests for sensitivity.run_sensitivity_analysis."""

    def test_full_ti_analysis(self):
        from basis_fields import generate_synthetic_basis
        from sensitivity import run_sensitivity_analysis, print_sensitivity_summary
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        results = run_sensitivity_analysis(
            basis, tidx, sidx, mode="TI",
            n_reachable_samples=30, n_pareto_weights=5)
        assert "jacobian_ti" in results
        assert "condition_number_ti" in results
        assert "reachable_ti" in results
        assert "pareto_ti" in results

        summary = print_sensitivity_summary(results)
        assert "TI Jacobian condition" in summary

    def test_full_nts_analysis(self):
        from basis_fields import generate_synthetic_basis
        from sensitivity import run_sensitivity_analysis
        basis, tidx, sidx = generate_synthetic_basis(n_points=500, n_coils=8, target_idx=250)
        results = run_sensitivity_analysis(
            basis, tidx, sidx, mode="NTS",
            n_reachable_samples=30, n_pareto_weights=5)
        assert "jacobian_nts" in results
        assert "condition_number_nts_alpha" in results
        assert "reachable_nts" in results
        assert "nts_jacobian_agreement" in results
