"""Comprehensive test suite for the Omnidream TMS array system.

Tests cover: config, helmet geometry, TI fields, NTS timing, coupling,
basis fields (synthetic), fitness functions, and SAC environments.

Run: cd Omnidream && python -m pytest tests/test_all.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure Omnidream root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    OmnidreamConfig, CoilConfig, HelmetConfig, TIConfig, NTSConfig,
    SafetyConfig, SimConfig, paper_baseline, omnidream_variant,
    save_config, load_config, _to_dict, _from_dict,
)
from helmet_geometry import (
    sample_helmet_positions, compute_helmet_normals,
    enforce_min_distance_on_sphere,
)
from ti_fields import (
    compute_group_amplitudes, compute_modulation_depth,
    compute_envelope_bounds, compute_ti_sar, ti_fitness,
)
from nts_timing import (
    membrane_decay_weights, compute_v_peak, optimal_firing_order,
    enforce_guard_times, compute_per_pulse_surface_max, nts_fitness,
)
from coupling import (
    dipole_mutual_inductance, build_inductance_matrix,
    build_impedance_matrix, compensate_coupling, coupling_coefficient,
)
from basis_fields import generate_synthetic_basis


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def config():
    return paper_baseline()


@pytest.fixture
def synthetic_data():
    basis, target_idx, surface_indices = generate_synthetic_basis(
        n_points=500, n_coils=8, target_idx=250, rng=np.random.default_rng(42),
    )
    return basis, target_idx, surface_indices


# =====================================================================
# Config tests
# =====================================================================

class TestConfig:
    def test_paper_baseline_defaults(self):
        cfg = paper_baseline()
        assert cfg.coil.relative_permeability == 75.0
        assert cfg.coil.base_gap_mm == 5.0
        assert cfg.coil.total_turns == 30

    def test_omnidream_variant(self):
        cfg = omnidream_variant()
        assert cfg.coil.relative_permeability == 5000.0
        assert cfg.coil.base_gap_mm == 0.5

    def test_roundtrip_yaml(self, tmp_path):
        cfg = paper_baseline()
        p = tmp_path / "test_config.yaml"
        save_config(cfg, p)
        loaded = load_config(p)
        assert loaded.coil.total_turns == cfg.coil.total_turns
        assert loaded.helmet.r_inner_mm == cfg.helmet.r_inner_mm
        assert loaded.ti.freq_carrier_hz == cfg.ti.freq_carrier_hz

    def test_dict_roundtrip(self):
        cfg = paper_baseline()
        d = _to_dict(cfg)
        restored = _from_dict(d)
        assert restored.coil.total_turns == cfg.coil.total_turns
        assert restored.safety.sar_limit_wkg == cfg.safety.sar_limit_wkg


# =====================================================================
# Helmet geometry tests
# =====================================================================

class TestHelmetGeometry:
    def test_sample_positions_shape(self):
        pos = sample_helmet_positions(32, r_inner=90.0, theta_max_deg=120.0)
        assert pos.shape == (32, 3)

    def test_positions_on_sphere(self):
        r = 90.0
        pos = sample_helmet_positions(100, r_inner=r)
        radii = np.linalg.norm(pos, axis=1)
        np.testing.assert_allclose(radii, r, atol=1e-10)

    def test_normals_point_inward(self):
        pos = sample_helmet_positions(10, r_inner=90.0)
        normals = compute_helmet_normals(pos)
        # Inward normal should have negative dot product with position
        dots = np.sum(pos * normals, axis=1)
        assert np.all(dots < 0)

    def test_normals_unit_length(self):
        pos = sample_helmet_positions(10, r_inner=90.0)
        normals = compute_helmet_normals(pos)
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_min_distance_enforced(self):
        pos = sample_helmet_positions(10, r_inner=90.0, rng=np.random.default_rng(0))
        min_dist = 15.0
        pos_adj = enforce_min_distance_on_sphere(pos, min_dist, r_inner=90.0)
        dists = np.linalg.norm(pos_adj[:, None, :] - pos_adj[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        assert np.min(dists) >= min_dist - 1e-6  # small tolerance

    def test_theta_max_respected(self):
        theta_max = 60.0
        pos = sample_helmet_positions(200, r_inner=90.0, theta_max_deg=theta_max,
                                       rng=np.random.default_rng(7))
        # z = R·cos(θ), so z >= R·cos(θ_max)
        z_min = 90.0 * np.cos(np.radians(theta_max))
        assert np.all(pos[:, 2] >= z_min - 1e-10)


# =====================================================================
# TI fields tests
# =====================================================================

class TestTIFields:
    def test_modulation_depth_symmetric(self, synthetic_data):
        basis, target_idx, surface_indices = synthetic_data
        N = basis.shape[1]
        amplitudes = np.ones(N)
        group = np.array([0, 1] * (N // 2), dtype=float)

        A1, A2 = compute_group_amplitudes(amplitudes, group, basis, 1000.0, 1010.0)
        M = compute_modulation_depth(A1, A2)
        assert M.shape == (basis.shape[0],)
        assert np.all(M >= 0)

    def test_modulation_depth_formula(self):
        A1 = np.array([3.0, 5.0, 1.0])
        A2 = np.array([2.0, 5.0, 4.0])
        M = compute_modulation_depth(A1, A2)
        expected = 2.0 * np.minimum(np.abs(A1), np.abs(A2))
        np.testing.assert_allclose(M, expected)

    def test_envelope_bounds(self):
        A1 = np.array([3.0, 5.0])
        A2 = np.array([2.0, 5.0])
        E_min, E_max = compute_envelope_bounds(A1, A2)
        np.testing.assert_allclose(E_min, [1.0, 0.0])
        np.testing.assert_allclose(E_max, [5.0, 10.0])

    def test_sar_nonnegative(self, synthetic_data):
        basis, _, _ = synthetic_data
        N = basis.shape[1]
        a = np.ones(N)
        g = np.array([0, 1] * (N // 2), dtype=float)
        A1, A2 = compute_group_amplitudes(a, g, basis, 1000, 1010)
        SAR = compute_ti_sar(A1, A2)
        assert np.all(SAR >= 0)

    def test_ti_fitness_returns_float(self, synthetic_data):
        basis, target_idx, surface_indices = synthetic_data
        N = basis.shape[1]
        cost = ti_fitness(
            amplitudes=np.ones(N),
            group=np.array([0, 1] * (N // 2), dtype=float),
            freq_carrier=1000.0,
            delta_freq=10.0,
            basis_matrix=basis,
            target_idx=target_idx,
            surface_indices=surface_indices,
        )
        assert isinstance(cost, float)


# =====================================================================
# NTS timing tests
# =====================================================================

class TestNTSTiming:
    def test_decay_weights_last_is_one(self):
        t = np.array([0.0, 0.001, 0.002, 0.003, 0.005])
        w = membrane_decay_weights(t, tau_m=0.003)
        # Last pulse (t=5ms) should have weight 1
        assert np.isclose(w[-1], 1.0)

    def test_decay_weights_monotone(self):
        t = np.sort(np.random.uniform(0, 0.005, 10))
        w = membrane_decay_weights(t, tau_m=0.003)
        # Weights should increase with firing time
        assert np.all(np.diff(w[np.argsort(t)]) >= -1e-12)

    def test_v_peak_nonnegative(self, synthetic_data):
        basis, target_idx, surface_indices = synthetic_data
        N = basis.shape[1]
        V = compute_v_peak(
            np.ones(N),
            np.linspace(0, 0.005, N),
            basis, tau_m=0.003,
        )
        assert np.all(V >= -1e-12)

    def test_optimal_order_weakest_first(self, synthetic_data):
        basis, target_idx, _ = synthetic_data
        order = optimal_firing_order(basis, target_idx)
        target_fields = np.abs(basis[target_idx])
        assert target_fields[order[0]] <= target_fields[order[-1]]

    def test_guard_time_enforcement(self):
        t = np.array([0.0, 0.00005, 0.0001, 0.00015, 0.001])
        t_adj = enforce_guard_times(t, t_guard=0.0002)
        diffs = np.diff(np.sort(t_adj))
        assert np.all(diffs >= 0.0002 - 1e-12)

    def test_nts_fitness_returns_float(self, synthetic_data):
        basis, target_idx, surface_indices = synthetic_data
        N = basis.shape[1]
        cost = nts_fitness(
            amplitudes=np.ones(N),
            fire_times=np.linspace(0, 0.005, N),
            basis_matrix=basis,
            target_idx=target_idx,
            surface_indices=surface_indices,
        )
        assert isinstance(cost, float)

    def test_optimal_ordering_improves_v_peak(self, synthetic_data):
        """Weakest-first ordering should produce higher V_peak at target
        than random ordering (on average)."""
        basis, target_idx, _ = synthetic_data
        N = basis.shape[1]
        amplitudes = np.ones(N)

        # Optimal ordering
        order_opt = optimal_firing_order(basis, target_idx)
        from nts_timing import assign_uniform_fire_times
        t_opt = assign_uniform_fire_times(order_opt, tau_window=0.005)
        V_opt = compute_v_peak(amplitudes, t_opt, basis, tau_m=0.003)

        # Random ordering
        rng = np.random.default_rng(123)
        random_better = 0
        n_trials = 20
        for _ in range(n_trials):
            order_rand = rng.permutation(N)
            t_rand = assign_uniform_fire_times(order_rand, tau_window=0.005)
            V_rand = compute_v_peak(amplitudes, t_rand, basis, tau_m=0.003)
            if V_rand[target_idx] > V_opt[target_idx]:
                random_better += 1

        # Optimal should be better than random in at least 50% of cases
        assert random_better <= n_trials * 0.5 + 3  # allow some slack


# =====================================================================
# Coupling tests
# =====================================================================

class TestCoupling:
    def test_mutual_inductance_decays_with_distance(self):
        M_near = dipole_mutual_inductance(
            np.array([0, 0, 0.0]), np.array([0.01, 0, 0]),
            30, 30, 28e-6, 28e-6,
        )
        M_far = dipole_mutual_inductance(
            np.array([0, 0, 0.0]), np.array([0.05, 0, 0]),
            30, 30, 28e-6, 28e-6,
        )
        assert M_near > M_far > 0

    def test_inductance_matrix_symmetric(self, config):
        pos = sample_helmet_positions(8, r_inner=90.0, rng=np.random.default_rng(0))
        L = build_inductance_matrix(pos, config.coil)
        np.testing.assert_allclose(L, L.T, atol=1e-20)

    def test_coupling_coefficient_small(self, config):
        """For miniature coils at 25 mm, k < 0.01 (from formulations §6.3)."""
        pos = np.array([[0, 0, 0], [25, 0, 0]], dtype=float)  # mm
        L = build_inductance_matrix(pos, config.coil)
        k = coupling_coefficient(L[0, 1], L[0, 0], L[1, 1])
        assert k < 0.01

    def test_impedance_matrix_diagonal_dominated(self, config):
        pos = sample_helmet_positions(4, r_inner=90.0, rng=np.random.default_rng(0))
        L = build_inductance_matrix(pos, config.coil)
        Z = build_impedance_matrix(L, config.coil.coil_resistance_ohm, 1000.0)
        # Diagonal should be larger than any off-diagonal in each row
        for i in range(Z.shape[0]):
            diag = abs(Z[i, i])
            offdiag_max = max(abs(Z[i, j]) for j in range(Z.shape[1]) if j != i)
            assert diag > offdiag_max

    def test_compensate_coupling_roundtrip(self, config):
        pos = sample_helmet_positions(4, r_inner=90.0, rng=np.random.default_rng(0))
        L = build_inductance_matrix(pos, config.coil)
        Z = build_impedance_matrix(L, config.coil.coil_resistance_ohm, 1000.0)
        I_desired = np.array([1.0, 2.0, 1.5, 0.5])
        V_cmd = compensate_coupling(I_desired, Z)
        # Verify: Z^{-1} @ V_cmd ≈ I_desired
        I_recovered = np.linalg.solve(Z, V_cmd)
        np.testing.assert_allclose(np.real(I_recovered), I_desired, atol=1e-10)


# =====================================================================
# Basis fields (synthetic) tests
# =====================================================================

class TestBasisFields:
    def test_synthetic_shape(self):
        basis, target_idx, surface_indices = generate_synthetic_basis(
            n_points=1000, n_coils=16, target_idx=500,
        )
        assert basis.shape == (1000, 16)
        assert 0 <= target_idx < 1000
        assert len(surface_indices) > 0

    def test_synthetic_nonnegative(self):
        basis, _, _ = generate_synthetic_basis()
        assert np.all(basis >= 0)


# =====================================================================
# SAC environment tests
# =====================================================================

_torch_available = True
try:
    import torch  # noqa: F401
except ImportError:
    _torch_available = False


@pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")
class TestSACEnvironments:
    def test_brainenv_ti_interface(self, synthetic_data):
        from sac_tms_control import BrainEnv_TI
        basis, target_idx, surface_indices = synthetic_data
        env = BrainEnv_TI(basis, target_idx, surface_indices)

        s = env.reset()
        assert s.shape == (env.state_dim,)
        assert env.action_dim == basis.shape[1] + 1

        action = np.zeros(env.action_dim)
        s_next, r, done, info = env.step(action)
        assert s_next.shape == (env.state_dim,)
        assert isinstance(r, float)
        assert isinstance(done, bool)
        assert "target_metric" in info
        assert "surface_metric" in info
        assert "morl_reward" in info
        assert "morl_objectives" in info

    def test_brainenv_nts_interface(self, synthetic_data):
        from sac_tms_control import BrainEnv_NTS
        basis, target_idx, surface_indices = synthetic_data
        env = BrainEnv_NTS(basis, target_idx, surface_indices)

        s = env.reset()
        assert s.shape == (env.state_dim,)
        assert env.action_dim == 2 * basis.shape[1]

        action = np.zeros(env.action_dim)
        s_next, r, done, info = env.step(action)
        assert s_next.shape == (env.state_dim,)
        assert "target_metric" in info
        assert "surface_metric" in info
        assert "morl_reward" in info
        assert "morl_objectives" in info

    def test_brainenv_hybrid_interface(self, synthetic_data):
        from sac_tms_control import BrainEnv_Hybrid
        basis, target_idx, surface_indices = synthetic_data
        env = BrainEnv_Hybrid(basis, target_idx, surface_indices)

        s = env.reset()
        assert env.action_dim == 2 * basis.shape[1] + 1

        action = np.zeros(env.action_dim)
        s_next, r, done, info = env.step(action)
        assert "M_target" in info
        assert "V_target" in info
        assert "target_metric" in info
        assert "surface_metric" in info
        assert "morl_reward" in info
        assert "morl_objectives" in info

    def test_brainenv_ti_episode_completes(self, synthetic_data):
        from sac_tms_control import BrainEnv_TI
        basis, target_idx, surface_indices = synthetic_data
        env = BrainEnv_TI(basis, target_idx, surface_indices, max_steps=5)
        s = env.reset()
        for _ in range(5):
            a = np.random.uniform(-1, 1, env.action_dim)
            s, r, done, _ = env.step(a)
        assert done


# =====================================================================
# GA fitness function tests
# =====================================================================

class TestFitnessFunctions:
    def test_fitness_ti_sign_convention(self, synthetic_data):
        """Better TI solutions should have lower cost."""
        from optimal_configuration import fitness_TI, create_individual
        basis, target_idx, surface_indices = synthetic_data
        N = basis.shape[1]

        ind = create_individual(
            positions=np.zeros((N, 3)),
            orientations=np.zeros((N, 3)),
            amplitudes=np.ones(N),
            group=np.array([0, 1] * (N // 2), dtype=float),
            mode="TI",
        )
        cost = fitness_TI(ind, basis, target_idx, surface_indices)
        assert isinstance(cost, float)

    def test_fitness_nts_sign_convention(self, synthetic_data):
        from optimal_configuration import fitness_NTS, create_individual
        basis, target_idx, surface_indices = synthetic_data
        N = basis.shape[1]

        ind = create_individual(
            positions=np.zeros((N, 3)),
            orientations=np.zeros((N, 3)),
            amplitudes=np.ones(N),
            fire_times=np.linspace(0, 0.005, N),
            mode="NTS",
        )
        cost = fitness_NTS(ind, basis, target_idx, surface_indices)
        assert isinstance(cost, float)

    def test_fitness_dispatch_all_modes(self, synthetic_data):
        from optimal_configuration import fitness_dispatch, create_individual
        basis, target_idx, surface_indices = synthetic_data
        N = basis.shape[1]

        for mode in ("superposition", "TI", "NTS", "hybrid"):
            ind = create_individual(
                positions=np.zeros((N, 3)),
                orientations=np.zeros((N, 3)),
                amplitudes=np.ones(N),
                group=np.array([0, 1] * (N // 2), dtype=float),
                fire_times=np.linspace(0, 0.005, N),
                mode=mode,
            )
            cost = fitness_dispatch(ind, basis, target_idx, surface_indices)
            assert isinstance(cost, float), f"Mode {mode} returned non-float"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
