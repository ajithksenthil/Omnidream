"""Tests for the perceptual outcome atlas module (atlas.py).

Covers:
    - Perceptual state evaluation
    - Perceptual Jacobian computation
    - Chart construction and properties
    - Transition maps between charts
    - Atlas construction from WorldState objects
    - Atlas densification
    - Geodesic path finding
    - Topology analysis
    - Serialisation round-trip

Performance: all fixtures are cached at module level so the expensive
atlas build happens only once.  Uses 3 coils × 4 worlds for speed.
"""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas import (
    Atlas,
    Chart,
    PerceptualState,
    TopologyInfo,
    TransitionMap,
    analyse_topology,
    build_atlas,
    build_chart,
    compute_perceptual_distance,
    compute_perceptual_jacobian,
    compute_transition,
    densify_atlas,
    dijkstra_through_charts,
    evaluate_perceptual_state,
    find_geodesic,
    find_nearest_chart,
    identify_coverage_gaps,
    print_atlas_summary,
    save_atlas,
    load_atlas_summary,
    _state_to_vector,
    _weights_from_config,
    _PERCEPTUAL_DIMS,
    DEFAULT_PERCEPTUAL_WEIGHTS,
)
from config import AtlasConfig, OmnidreamConfig, save_config, load_config


# =====================================================================
# Helpers — shared fixtures (module-level for speed)
# =====================================================================

_FIXTURES = None
_ATLAS = None
_CHART_A = None
_CHART_B = None
_TRANSITION = None
_DENSE_ATLAS = None
_GEODESIC = None

# Tiny config for speed:  3 coils, 4 worlds
_N_COILS = 3
_N_WORLDS = 4


def _make_basis(n_points: int = 30, n_coils: int = _N_COILS) -> np.ndarray:
    """Create a small synthetic basis matrix."""
    rng = np.random.default_rng(42)
    basis = rng.standard_normal((n_points, n_coils)) * 0.1
    basis[n_points // 2, :] *= 5.0  # boost target
    return basis


def _make_L_matrix(n_coils: int = _N_COILS) -> np.ndarray:
    """Create a synthetic inductance matrix."""
    L_self = 5e-6
    L = np.eye(n_coils) * L_self
    for i in range(n_coils):
        for j in range(i + 1, n_coils):
            M = L_self * 0.01 * np.exp(-abs(i - j) / 2.0)
            L[i, j] = M
            L[j, i] = M
    return L


def _get_fixtures():
    """Create or return cached test fixtures."""
    global _FIXTURES
    if _FIXTURES is not None:
        return _FIXTURES

    from config import paper_baseline
    from control_framework import TMSPlant, StimulationMode, GoalSpec
    from cp_bridge import (
        WorldState, CPBridgeConfig, compute_total_energy,
        compute_transfer_entropy_matrix, compute_phases_from_groups,
        compute_sync_order_parameter,
    )

    n_coils = _N_COILS
    n_worlds = _N_WORLDS
    basis = _make_basis(n_coils=n_coils)
    target_idx = basis.shape[0] // 2
    surface_indices = np.arange(0, min(10, basis.shape[0]))
    config = paper_baseline()

    plant = TMSPlant(
        basis_matrix=basis,
        target_idx=target_idx,
        surface_indices=surface_indices,
        config=config,
        mode=StimulationMode.NTS,
    )

    L = _make_L_matrix(n_coils)
    cp_config = CPBridgeConfig()
    te_matrix = compute_transfer_entropy_matrix(L, cp_config.noise_variance)

    group = np.zeros(n_coils)
    group[n_coils // 2:] = 1.0
    f1 = config.ti.freq_carrier_hz
    f2 = f1 + config.ti.delta_freq_default_hz
    phases = compute_phases_from_groups(group, f1, f2, t=0.0)
    eta_field = np.where(group > 0.5, 1.0, 0.1)
    goal = GoalSpec(target_idx=target_idx, surface_indices=surface_indices)

    # Create synthetic worlds with varying amplitudes
    rng = np.random.default_rng(123)
    worlds = []
    for i in range(n_worlds):
        amps = rng.uniform(0.1, 0.8, n_coils)
        outputs = plant.forward_from_params(amps)
        e_dict = compute_total_energy(
            plant, amps, te_matrix, phases, basis,
            target_idx, eta_field, group, goal, cp_config,
        )
        R = compute_sync_order_parameter(phases)
        worlds.append(WorldState(
            amplitudes=amps,
            outputs=outputs,
            energy=e_dict["E_total"],
            phi=e_dict["phi_collective"],
            probability=1.0 / n_worlds,
            world_id=i,
            sync_R=R,
        ))

    _FIXTURES = dict(
        plant=plant,
        basis_matrix=basis,
        target_idx=target_idx,
        surface_indices=surface_indices,
        config=config,
        te_matrix=te_matrix,
        phases=phases,
        eta_field=eta_field,
        group=group,
        goal=goal,
        cp_config=cp_config,
        L_matrix=L,
        worlds=worlds,
    )
    return _FIXTURES


def _common_kwargs(fx):
    """Extract the common kwargs dict used by most atlas functions."""
    return dict(
        plant=fx["plant"],
        te_matrix=fx["te_matrix"],
        phases=fx["phases"],
        basis_matrix=fx["basis_matrix"],
        target_idx=fx["target_idx"],
        eta_field=fx["eta_field"],
        group=fx["group"],
        config=fx["config"],
        goal=fx["goal"],
        cp_config=fx["cp_config"],
    )


def _get_atlas():
    """Create or return cached atlas (built once for all tests)."""
    global _ATLAS
    if _ATLAS is not None:
        return _ATLAS

    fx = _get_fixtures()
    _ATLAS = build_atlas(
        worlds=fx["worlds"],
        k_neighbors=2,
        compute_hessians=False,
        **_common_kwargs(fx),
    )
    return _ATLAS


def _get_charts():
    """Return two pre-built charts (cached)."""
    global _CHART_A, _CHART_B
    if _CHART_A is not None:
        return _CHART_A, _CHART_B

    fx = _get_fixtures()
    ck = _common_kwargs(fx)
    _CHART_A = build_chart(
        amplitudes=fx["worlds"][0].amplitudes,
        chart_id=0,
        compute_hessian_flag=True,
        **ck,
    )
    _CHART_B = build_chart(
        amplitudes=fx["worlds"][1].amplitudes,
        chart_id=1,
        compute_hessian_flag=False,
        **ck,
    )
    return _CHART_A, _CHART_B


def _get_transition():
    """Return a pre-computed transition map (cached)."""
    global _TRANSITION
    if _TRANSITION is not None:
        return _TRANSITION

    fx = _get_fixtures()
    chart_a, chart_b = _get_charts()
    _TRANSITION = compute_transition(
        chart_a, chart_b,
        n_barrier_samples=3,
        **_common_kwargs(fx),
    )
    return _TRANSITION


def _get_dense_atlas():
    """Return a cached densified atlas (one new chart)."""
    global _DENSE_ATLAS
    if _DENSE_ATLAS is not None:
        return _DENSE_ATLAS

    fx = _get_fixtures()
    # Deep copy so we don't mutate the cached atlas
    atlas = copy.deepcopy(_get_atlas())
    _DENSE_ATLAS = densify_atlas(
        atlas,
        max_new=1,
        **_common_kwargs(fx),
    )
    return _DENSE_ATLAS


def _get_geodesic():
    """Return a cached geodesic result."""
    global _GEODESIC
    if _GEODESIC is not None:
        return _GEODESIC

    fx = _get_fixtures()
    atlas = _get_atlas()
    ids = sorted(atlas.charts.keys())
    state_a = atlas.charts[ids[0]].center_state
    state_b = atlas.charts[ids[-1]].center_state

    _GEODESIC = find_geodesic(
        atlas, state_a, state_b,
        n_steps=5,
        **_common_kwargs(fx),
    )
    return _GEODESIC


# =====================================================================
# Test Classes
# =====================================================================

class TestPerceptualState(unittest.TestCase):
    """Test evaluate_perceptual_state."""

    def test_returns_all_fields(self):
        """Perceptual state has all expected attributes."""
        fx = _get_fixtures()
        amps = fx["worlds"][0].amplitudes
        state = evaluate_perceptual_state(amps, **_common_kwargs(fx))
        self.assertIsInstance(state.phi, float)
        self.assertIsInstance(state.sync_R, float)
        self.assertIsInstance(state.energy, float)
        self.assertIsInstance(state.energy_components, dict)
        self.assertIsInstance(state.morl_objectives, dict)
        self.assertEqual(state.outputs.shape, (5,))
        self.assertIsInstance(state.safety_margins, dict)

    def test_phi_is_positive(self):
        """Phi should be positive for non-zero amplitudes."""
        fx = _get_fixtures()
        amps = fx["worlds"][0].amplitudes
        state = evaluate_perceptual_state(amps, **_common_kwargs(fx))
        self.assertGreater(state.phi, 0.0)

    def test_state_to_vector(self):
        """_state_to_vector returns correct shape."""
        fx = _get_fixtures()
        amps = fx["worlds"][0].amplitudes
        state = evaluate_perceptual_state(amps, **_common_kwargs(fx))
        vec = _state_to_vector(state)
        self.assertEqual(vec.shape, (len(_PERCEPTUAL_DIMS),))
        self.assertAlmostEqual(vec[0], state.phi)
        self.assertAlmostEqual(vec[1], state.sync_R)
        self.assertAlmostEqual(vec[2], state.energy)


class TestPerceptualJacobian(unittest.TestCase):
    """Test compute_perceptual_jacobian."""

    def test_shape(self):
        """Jacobian has shape (P, N)."""
        fx = _get_fixtures()
        amps = fx["worlds"][0].amplitudes
        J = compute_perceptual_jacobian(amps, **_common_kwargs(fx))
        P = len(_PERCEPTUAL_DIMS)
        N = len(amps)
        self.assertEqual(J.shape, (P, N))

    def test_nonzero(self):
        """Jacobian should have non-zero entries for active coils."""
        fx = _get_fixtures()
        amps = fx["worlds"][0].amplitudes
        J = compute_perceptual_jacobian(amps, **_common_kwargs(fx))
        # At least the energy row should have non-zero entries
        self.assertGreater(np.max(np.abs(J[2, :])), 0.0,
                           "Energy row of Jacobian should be non-zero")


class TestChart(unittest.TestCase):
    """Test build_chart and Chart properties."""

    def test_all_fields_populated(self):
        chart_a, _ = _get_charts()
        c = chart_a
        self.assertEqual(c.chart_id, 0)
        self.assertIsInstance(c.center_state, PerceptualState)
        self.assertEqual(c.J_output.shape[0], 5)
        self.assertEqual(c.J_perceptual.shape[0], len(_PERCEPTUAL_DIMS))
        self.assertIn("S", c.svd)
        self.assertIn("rank", c.svd)

    def test_svd_rank(self):
        chart_a, _ = _get_charts()
        P, N = chart_a.J_perceptual.shape
        self.assertLessEqual(chart_a.svd["rank"], min(P, N))

    def test_metric_tensor_shape(self):
        chart_a, _ = _get_charts()
        N = len(chart_a.center_amplitudes)
        self.assertEqual(chart_a.metric_tensor.shape, (N, N))

    def test_metric_tensor_psd(self):
        chart_a, _ = _get_charts()
        eigvals = np.linalg.eigvalsh(chart_a.metric_tensor)
        self.assertTrue(np.all(eigvals >= -1e-10),
                        f"Negative eigenvalue: {eigvals.min()}")

    def test_hessian_shape(self):
        chart_a, _ = _get_charts()
        N = len(chart_a.center_amplitudes)
        self.assertEqual(chart_a.hessian_energy.shape, (N, N))

    def test_hessian_symmetric(self):
        chart_a, _ = _get_charts()
        H = chart_a.hessian_energy
        np.testing.assert_allclose(H, H.T, atol=1e-6)

    def test_valid_radius_positive(self):
        chart_a, _ = _get_charts()
        self.assertGreater(chart_a.valid_radius, 0.0)

    def test_effective_dim(self):
        chart_a, _ = _get_charts()
        self.assertGreaterEqual(chart_a.effective_dim, 1)


class TestTransitionMap(unittest.TestCase):
    """Test compute_transition."""

    def test_perceptual_distance_positive(self):
        trans = _get_transition()
        self.assertGreater(trans.perceptual_distance, 0.0)

    def test_energy_barrier_nonnegative(self):
        trans = _get_transition()
        self.assertGreaterEqual(trans.energy_barrier, 0.0)

    def test_jacobian_ab_shape(self):
        chart_a, _ = _get_charts()
        trans = _get_transition()
        N = len(chart_a.center_amplitudes)
        self.assertEqual(trans.jacobian_ab.shape, (N, N))

    def test_overlap_center(self):
        chart_a, chart_b = _get_charts()
        trans = _get_transition()
        expected = 0.5 * (chart_a.center_amplitudes + chart_b.center_amplitudes)
        np.testing.assert_allclose(trans.overlap_center, expected, atol=1e-10)

    def test_is_valid_flag(self):
        trans = _get_transition()
        self.assertIsInstance(trans.is_valid, bool)


class TestAtlasBuild(unittest.TestCase):
    """Test build_atlas from WorldState objects."""

    def test_chart_count(self):
        atlas = _get_atlas()
        self.assertEqual(len(atlas.charts), _N_WORLDS)

    def test_transitions_exist(self):
        atlas = _get_atlas()
        self.assertGreater(len(atlas.transitions), 0)

    def test_topology_present(self):
        atlas = _get_atlas()
        self.assertIsInstance(atlas.topology, TopologyInfo)
        self.assertGreater(atlas.topology.n_components, 0)

    def test_perceptual_bounds(self):
        atlas = _get_atlas()
        for dim in _PERCEPTUAL_DIMS:
            self.assertIn(dim, atlas.perceptual_bounds)
            lo, hi = atlas.perceptual_bounds[dim]
            self.assertLessEqual(lo, hi)

    def test_metadata(self):
        atlas = _get_atlas()
        self.assertIn("n_charts", atlas.metadata)
        self.assertIn("build_time_s", atlas.metadata)
        self.assertEqual(atlas.metadata["n_charts"], _N_WORLDS)

    def test_all_charts_connected(self):
        atlas = _get_atlas()
        self.assertEqual(atlas.topology.n_components, 1)


class TestDensification(unittest.TestCase):
    """Test densify_atlas."""

    def test_charts_added(self):
        atlas = _get_dense_atlas()
        # Should have more charts than the original
        self.assertGreater(len(atlas.charts), _N_WORLDS)
        self.assertIn("n_densified", atlas.metadata)
        self.assertGreater(atlas.metadata["n_densified"], 0)

    def test_topology_reanalysed(self):
        atlas = _get_dense_atlas()
        # After densification, should still be connected
        self.assertEqual(atlas.topology.n_components, 1)


class TestGeodesic(unittest.TestCase):
    """Test find_geodesic."""

    def test_geodesic_returns_path(self):
        result = _get_geodesic()
        self.assertIn("path_amplitudes", result)
        self.assertEqual(len(result["path_amplitudes"]), 5)
        self.assertEqual(len(result["path_energies"]), 5)
        self.assertGreater(len(result["chart_sequence"]), 0)

    def test_geodesic_cost_positive(self):
        result = _get_geodesic()
        self.assertGreater(result["total_cost"], 0.0)


class TestTopology(unittest.TestCase):
    """Test analyse_topology (unit tests with synthetic charts)."""

    def _make_chart(self, cid, energy=1.0, boundary=False, dim=4):
        return Chart(
            chart_id=cid,
            center_amplitudes=np.ones(2) * cid,
            center_state=PerceptualState(
                phi=1.0, sync_R=1.0, energy=energy,
                energy_components={}, morl_objectives={},
                outputs=np.zeros(5),
                safety_margins={"sar": 0.1 if boundary else 1.0,
                                "thermal": 1.0, "current": 1.0, "voltage": 1.0},
            ),
            J_output=np.zeros((5, 2)),
            J_perceptual=np.zeros((4, 2)),
            svd={"S": np.ones(2), "rank": dim, "condition_number": 1.0,
                 "explained_variance": np.ones(2)},
            metric_tensor=np.eye(2),
            hessian_energy=np.zeros((2, 2)),
            valid_radius=0.1,
            neighbor_ids=[],
            is_boundary=boundary,
            effective_dim=dim,
        )

    def test_single_component(self):
        charts = {i: self._make_chart(i) for i in range(5)}
        for i in range(5):
            charts[i].neighbor_ids = [j for j in range(5) if j != i]
        transitions = {
            (i, i + 1): TransitionMap(i, i + 1, np.ones(2), np.eye(2), 1.0, 0.1, True)
            for i in range(4)
        }
        topo = analyse_topology(charts, transitions)
        self.assertEqual(topo.n_components, 1)
        self.assertEqual(topo.mean_dimension, 4.0)

    def test_two_components(self):
        charts = {i: self._make_chart(i) for i in range(4)}
        charts[0].neighbor_ids = [1]
        charts[1].neighbor_ids = [0]
        charts[2].neighbor_ids = [3]
        charts[3].neighbor_ids = [2]
        transitions = {
            (0, 1): TransitionMap(0, 1, np.ones(2), np.eye(2), 1.0, 0.0, True),
            (2, 3): TransitionMap(2, 3, np.ones(2), np.eye(2), 1.0, 0.0, True),
        }
        topo = analyse_topology(charts, transitions)
        self.assertEqual(topo.n_components, 2)

    def test_boundary_detection(self):
        charts = {0: self._make_chart(0, boundary=True)}
        topo = analyse_topology(charts, {})
        self.assertIn(0, topo.boundary_chart_ids)


class TestSerialization(unittest.TestCase):
    """Test save_atlas and load_atlas_summary."""

    def test_save_load_roundtrip(self):
        atlas = _get_atlas()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "atlas_test.npz"
            save_atlas(atlas, path)
            data = load_atlas_summary(path)
            self.assertIn("chart_ids", data)
            self.assertIn("energies", data)
            self.assertIn("phis", data)
            self.assertIn("n_components", data)
            self.assertEqual(len(data["chart_ids"]), len(atlas.charts))

    def test_print_summary(self):
        atlas = _get_atlas()
        summary = print_atlas_summary(atlas)
        self.assertIn("Perceptual Outcome Atlas", summary)
        self.assertIn("Charts:", summary)
        self.assertIn("Topology:", summary)


class TestPerceptualDistance(unittest.TestCase):
    """Test compute_perceptual_distance."""

    def test_zero_distance(self):
        state = PerceptualState(
            phi=1.0, sync_R=0.5, energy=10.0,
            energy_components={}, morl_objectives={"J_phi": 1.0, "J_arch": 2.0, "J_sync": 0.5, "J_task": 0.1},
            outputs=np.zeros(5), safety_margins={},
        )
        d = compute_perceptual_distance(state, state)
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_positive_distance(self):
        state_a = PerceptualState(
            phi=1.0, sync_R=0.5, energy=10.0,
            energy_components={}, morl_objectives={"J_phi": 1.0, "J_arch": 2.0, "J_sync": 0.5, "J_task": 0.1},
            outputs=np.zeros(5), safety_margins={},
        )
        state_b = PerceptualState(
            phi=2.0, sync_R=0.8, energy=15.0,
            energy_components={}, morl_objectives={"J_phi": 1.5, "J_arch": 3.0, "J_sync": 0.9, "J_task": 0.2},
            outputs=np.ones(5), safety_margins={},
        )
        d = compute_perceptual_distance(state_a, state_b)
        self.assertGreater(d, 0.0)

    def test_triangle_inequality(self):
        states = []
        for phi, energy in [(1.0, 10.0), (2.0, 15.0), (1.5, 12.0)]:
            states.append(PerceptualState(
                phi=phi, sync_R=0.5, energy=energy,
                energy_components={}, morl_objectives={"J_phi": phi, "J_arch": 0.0, "J_sync": 0.5, "J_task": 0.0},
                outputs=np.zeros(5), safety_margins={},
            ))
        d01 = compute_perceptual_distance(states[0], states[1])
        d02 = compute_perceptual_distance(states[0], states[2])
        d12 = compute_perceptual_distance(states[1], states[2])
        self.assertLessEqual(d01, d02 + d12 + 1e-10)


class TestDijkstra(unittest.TestCase):
    """Test dijkstra_through_charts."""

    def test_same_start_end(self):
        atlas = _get_atlas()
        path = dijkstra_through_charts(atlas, 0, 0)
        self.assertEqual(path, [0])

    def test_path_connects(self):
        atlas = _get_atlas()
        ids = sorted(atlas.charts.keys())
        path = dijkstra_through_charts(atlas, ids[0], ids[-1])
        self.assertEqual(path[0], ids[0])
        self.assertEqual(path[-1], ids[-1])
        self.assertGreaterEqual(len(path), 2)


class TestAtlasConfig(unittest.TestCase):
    """Test AtlasConfig integration."""

    def test_config_has_atlas(self):
        """OmnidreamConfig includes AtlasConfig."""
        from config import paper_baseline
        cfg = paper_baseline()
        self.assertIsInstance(cfg.atlas, AtlasConfig)

    def test_weights_from_config(self):
        """_weights_from_config returns matching dict."""
        ac = AtlasConfig(w_phi=3.0, w_sync_R=2.0)
        w = _weights_from_config(ac)
        self.assertEqual(w["phi"], 3.0)
        self.assertEqual(w["sync_R"], 2.0)
        self.assertEqual(w["energy"], 0.5)  # default

    def test_atlas_config_forwarded(self):
        """build_atlas accepts and uses atlas_config parameter."""
        fx = _get_fixtures()
        ac = AtlasConfig(k_neighbors=2, compute_hessians=False)
        atlas = build_atlas(
            worlds=fx["worlds"],
            atlas_config=ac,
            **_common_kwargs(fx),
        )
        self.assertEqual(len(atlas.charts), _N_WORLDS)
        self.assertEqual(atlas.metadata["k_neighbors"], 2)

    def test_yaml_roundtrip(self):
        """AtlasConfig survives YAML save/load."""
        from config import paper_baseline
        cfg = paper_baseline()
        cfg.atlas.k_neighbors = 7
        cfg.atlas.w_phi = 9.0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_cfg.yaml"
            save_config(cfg, path)
            loaded = load_config(path)
            self.assertEqual(loaded.atlas.k_neighbors, 7)
            self.assertAlmostEqual(loaded.atlas.w_phi, 9.0)


if __name__ == "__main__":
    unittest.main()
