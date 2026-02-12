"""Tests for the CP bridge module (cp_bridge.py).

Covers:
    - Transfer entropy from mutual inductance
    - Collective Φ computation
    - Agent state construction and free energy
    - Group free energy and MORL objectives
    - 4D NCA energy functional
    - Many-worlds computation
    - Pipeline Stage 12 integration
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure Omnidream root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basis_fields import generate_synthetic_basis
from config import paper_baseline


# =====================================================================
# Fixtures
# =====================================================================

def _make_basis(n_points=500, n_coils=8):
    """Small synthetic basis for fast tests."""
    return generate_synthetic_basis(
        n_points=n_points, n_coils=n_coils,
        target_idx=n_points // 2,
        surface_fraction=0.1,
        rng=np.random.default_rng(42),
    )


def _make_L_matrix(n_coils=8, self_L=5e-6, mutual_L=1e-7):
    """Build a simple inductance matrix."""
    L = np.eye(n_coils) * self_L
    for i in range(n_coils):
        for j in range(i + 1, n_coils):
            # Coupling decays with distance in index space
            L[i, j] = mutual_L / (abs(i - j) + 1)
            L[j, i] = L[i, j]
    return L


# =====================================================================
# Test: Transfer Entropy
# =====================================================================

class TestTransferEntropy(unittest.TestCase):
    """Tests for compute_transfer_entropy_matrix and related functions."""

    def setUp(self):
        self.N = 8
        self.L = _make_L_matrix(self.N)

    def test_shape(self):
        from cp_bridge import compute_transfer_entropy_matrix
        T = compute_transfer_entropy_matrix(self.L)
        self.assertEqual(T.shape, (self.N, self.N))

    def test_non_negative(self):
        from cp_bridge import compute_transfer_entropy_matrix
        T = compute_transfer_entropy_matrix(self.L)
        self.assertTrue(np.all(T >= 0), "Transfer entropy must be non-negative")

    def test_diagonal_zeros(self):
        from cp_bridge import compute_transfer_entropy_matrix
        T = compute_transfer_entropy_matrix(self.L)
        np.testing.assert_array_equal(np.diag(T), 0.0)

    def test_symmetric_L_gives_symmetric_T(self):
        from cp_bridge import compute_transfer_entropy_matrix
        # Symmetric L with uniform self-inductance → symmetric T
        L_sym = _make_L_matrix(self.N, self_L=5e-6, mutual_L=1e-7)
        T = compute_transfer_entropy_matrix(L_sym)
        np.testing.assert_allclose(T, T.T, atol=1e-15)

    def test_higher_mutual_gives_higher_T(self):
        from cp_bridge import compute_transfer_entropy_matrix
        L1 = _make_L_matrix(self.N, mutual_L=1e-7)
        L2 = _make_L_matrix(self.N, mutual_L=5e-7)
        T1 = compute_transfer_entropy_matrix(L1)
        T2 = compute_transfer_entropy_matrix(L2)
        # Higher mutual inductance → higher transfer entropy
        self.assertGreater(np.sum(T2), np.sum(T1))


class TestAttachment(unittest.TestCase):

    def test_attachment_binary(self):
        from cp_bridge import compute_transfer_entropy_matrix, compute_attachment_matrix
        L = _make_L_matrix(8)
        T = compute_transfer_entropy_matrix(L)
        A = compute_attachment_matrix(T, threshold=0.01)
        self.assertEqual(A.dtype, bool)

    def test_mutual_attachment_symmetric(self):
        from cp_bridge import compute_transfer_entropy_matrix, compute_mutual_attachment
        L = _make_L_matrix(8)
        T = compute_transfer_entropy_matrix(L)
        MA = compute_mutual_attachment(T)
        np.testing.assert_allclose(MA, MA.T, atol=1e-15)


# =====================================================================
# Test: Collective Φ
# =====================================================================

class TestCollectivePhi(unittest.TestCase):

    def setUp(self):
        self.basis, self.target_idx, self.surface = _make_basis()
        self.N = self.basis.shape[1]
        self.amplitudes = np.ones(self.N) * 0.5
        self.L = _make_L_matrix(self.N)

    def test_individual_phi_sums_to_one(self):
        from cp_bridge import compute_individual_phi
        phi_i = compute_individual_phi(self.basis, self.amplitudes, self.target_idx)
        self.assertAlmostEqual(float(np.sum(phi_i)), 1.0, places=10)

    def test_collective_phi_non_negative(self):
        from cp_bridge import compute_collective_phi, compute_transfer_entropy_matrix
        T = compute_transfer_entropy_matrix(self.L)
        phases = np.zeros(self.N)  # All in phase → max coupling
        phi = compute_collective_phi(self.basis, self.amplitudes, T, phases, self.target_idx)
        self.assertGreaterEqual(phi, 0.0)

    def test_phase_coherence_boosts_phi(self):
        from cp_bridge import compute_collective_phi, compute_transfer_entropy_matrix
        T = compute_transfer_entropy_matrix(self.L)
        phases_coherent = np.zeros(self.N)
        phases_random = np.random.default_rng(42).uniform(0, 2 * np.pi, self.N)
        phi_coh = compute_collective_phi(self.basis, self.amplitudes, T, phases_coherent, self.target_idx)
        phi_rand = compute_collective_phi(self.basis, self.amplitudes, T, phases_random, self.target_idx)
        self.assertGreater(phi_coh, phi_rand)

    def test_zero_amplitudes_zero_individual_phi(self):
        from cp_bridge import compute_individual_phi
        phi_i = compute_individual_phi(self.basis, np.zeros(self.N), self.target_idx)
        np.testing.assert_allclose(phi_i, 0.0, atol=1e-15,
                                    err_msg="Zero amplitudes should give zero individual Φ")

    def test_phases_from_groups(self):
        from cp_bridge import compute_phases_from_groups
        group = np.zeros(8)
        group[4:] = 1.0
        phases = compute_phases_from_groups(group, 1000.0, 1040.0, t=0.001)
        # Group 0: 2π·1000·0.001 = 2π
        # Group 1: 2π·1040·0.001 = 2.08π
        self.assertEqual(len(phases), 8)
        # All group 0 phases should be equal
        np.testing.assert_allclose(phases[:4], phases[0])
        # All group 1 phases should be equal
        np.testing.assert_allclose(phases[4:], phases[4])
        # Group 1 should differ from group 0
        self.assertNotAlmostEqual(phases[0], phases[4], places=3)


# =====================================================================
# Test: Agent State
# =====================================================================

class TestAgentState(unittest.TestCase):

    def test_free_energy_positive(self):
        from cp_bridge import compute_agent_free_energy
        N = 8
        L = _make_L_matrix(N)
        R = np.ones(N) * 2.0
        amps = np.ones(N) * 0.5
        omega = 2 * np.pi * 1000
        for i in range(N):
            fe = compute_agent_free_energy(i, amps, L, R, omega)
            self.assertGreater(fe, 0.0, f"Agent {i} free energy should be positive")

    def test_back_emf_from_coupling(self):
        from cp_bridge import compute_back_emf
        N = 8
        L = _make_L_matrix(N, mutual_L=1e-6)  # stronger coupling
        amps = np.ones(N) * 0.5
        omega = 2 * np.pi * 1000
        v = compute_back_emf(0, amps, L, omega)
        self.assertGreater(v, 0.0, "Back-EMF should be positive with coupling")

    def test_mf_label_from_group(self):
        from cp_bridge import build_agent_states
        N = 8
        L = _make_L_matrix(N)
        R = np.ones(N) * 2.0
        group = np.zeros(N)
        group[4:] = 1.0
        amps = np.ones(N) * 0.5
        omega = 2 * np.pi * 1000
        agents = build_agent_states(amps, L, R, group, omega)
        for a in agents[:4]:
            self.assertEqual(a.mf_label, "M")
        for a in agents[4:]:
            self.assertEqual(a.mf_label, "F")


# =====================================================================
# Test: Group State and MORL
# =====================================================================

class TestGroupState(unittest.TestCase):

    def setUp(self):
        self.basis, self.target_idx, self.surface = _make_basis()
        self.N = self.basis.shape[1]
        self.config = paper_baseline()

    def test_morl_objectives_finite(self):
        from cp_bridge import compute_morl_objectives
        from control_framework import TMSPlant, StimulationMode

        plant = TMSPlant(self.basis, self.target_idx, self.surface,
                         self.config, StimulationMode.TI)
        L = _make_L_matrix(self.N)
        from cp_bridge import compute_transfer_entropy_matrix
        T = compute_transfer_entropy_matrix(L)
        phases = np.zeros(self.N)
        amps = np.ones(self.N) * 0.5
        obj = compute_morl_objectives(plant, amps, T, phases, self.basis, self.target_idx)
        for k, v in obj.items():
            self.assertTrue(np.isfinite(v), f"MORL {k} should be finite, got {v}")

    def test_sync_R_in_range(self):
        from cp_bridge import compute_sync_order_parameter
        # All in phase → R = 1
        R1 = compute_sync_order_parameter(np.zeros(8))
        self.assertAlmostEqual(R1, 1.0, places=10)
        # Evenly spaced → R ≈ 0
        R2 = compute_sync_order_parameter(np.linspace(0, 2 * np.pi, 9)[:-1])
        self.assertAlmostEqual(R2, 0.0, places=5)

    def test_scalarise_matches_manual(self):
        from cp_bridge import scalarize_morl
        obj = {"J_phi": 1.0, "J_arch": 0.5, "J_sync": 0.8, "J_task": 2.0}
        weights = {"J_phi": 1.0, "J_arch": -1.0, "J_sync": 1.0, "J_task": 1.0}
        expected = 1.0 * 1.0 + (-1.0) * 0.5 + 1.0 * 0.8 + 1.0 * 2.0
        result = scalarize_morl(obj, weights)
        self.assertAlmostEqual(result, expected, places=10)

    def test_group_free_energy_finite(self):
        from cp_bridge import (compute_group_free_energy, compute_transfer_entropy_matrix,
                                build_agent_states)
        from control_framework import TMSPlant, StimulationMode

        plant = TMSPlant(self.basis, self.target_idx, self.surface,
                         self.config, StimulationMode.TI)
        L = _make_L_matrix(self.N)
        R = np.ones(self.N) * 2.0
        group = np.zeros(self.N)
        group[self.N // 2:] = 1.0
        T = compute_transfer_entropy_matrix(L)
        phases = np.zeros(self.N)
        amps = np.ones(self.N) * 0.5
        omega = 2 * np.pi * 1000
        agents = build_agent_states(amps, L, R, group, omega, self.basis, self.target_idx)
        F2 = compute_group_free_energy(plant, amps, T, phases, self.basis, self.target_idx, agents)
        self.assertTrue(np.isfinite(F2), f"Group free energy should be finite, got {F2}")


# =====================================================================
# Test: 4D NCA Energy
# =====================================================================

class TestNCAEnergy(unittest.TestCase):

    def setUp(self):
        self.basis, self.target_idx, self.surface = _make_basis()
        self.N = self.basis.shape[1]
        self.config = paper_baseline()

    def test_nca_energy_non_negative(self):
        from cp_bridge import compute_nca_energy
        E = compute_nca_energy(self.basis, np.ones(self.N) * 0.5, self.target_idx)
        self.assertGreaterEqual(E, 0.0)

    def test_nca_energy_zero_for_zero_amps(self):
        from cp_bridge import compute_nca_energy
        E = compute_nca_energy(self.basis, np.zeros(self.N), self.target_idx)
        self.assertAlmostEqual(E, 0.0, places=10)

    def test_mf_energy_non_negative(self):
        from cp_bridge import compute_mf_energy
        group = np.zeros(self.N)
        group[self.N // 2:] = 1.0
        eta = np.where(group > 0.5, 1.0, 0.1)
        E = compute_mf_energy(eta, group)
        self.assertGreaterEqual(E, 0.0)

    def test_phi_energy_non_positive(self):
        from cp_bridge import compute_phi_energy
        # Positive Φ → negative E_phi
        E = compute_phi_energy(1.0, lambda_phi=2.0)
        self.assertLessEqual(E, 0.0)

    def test_coupling_energy_non_positive_in_phase(self):
        from cp_bridge import compute_coupling_energy, compute_transfer_entropy_matrix
        L = _make_L_matrix(self.N)
        T = compute_transfer_entropy_matrix(L)
        phases = np.zeros(self.N)  # All in phase
        E = compute_coupling_energy(T, phases)
        self.assertLessEqual(E, 0.0)

    def test_total_energy_is_sum(self):
        from cp_bridge import compute_total_energy, CPBridgeConfig
        from control_framework import TMSPlant, StimulationMode

        plant = TMSPlant(self.basis, self.target_idx, self.surface,
                         self.config, StimulationMode.TI)
        L = _make_L_matrix(self.N)
        from cp_bridge import compute_transfer_entropy_matrix
        T = compute_transfer_entropy_matrix(L)
        group = np.zeros(self.N)
        group[self.N // 2:] = 1.0
        phases = np.zeros(self.N)
        amps = np.ones(self.N) * 0.5
        eta = np.where(group > 0.5, 1.0, 0.1)
        cfg = CPBridgeConfig()

        E = compute_total_energy(plant, amps, T, phases, self.basis,
                                  self.target_idx, eta, group, cp_config=cfg)

        sum_parts = E["E_nca"] + E["E_mf"] + E["E_arch"] + E["E_phi"] + E["E_couple"] + E["E_morl"]
        self.assertAlmostEqual(E["E_total"], sum_parts, places=8)


# =====================================================================
# Test: Many Worlds
# =====================================================================

class TestManyWorlds(unittest.TestCase):

    def test_probabilities_sum_to_one(self):
        from cp_bridge import compute_world_probabilities
        energies = np.array([1.0, 2.0, 3.0, 0.5])
        probs = compute_world_probabilities(energies, temperature=1.0)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=10)

    def test_boltzmann_high_T_uniform(self):
        from cp_bridge import compute_world_probabilities
        energies = np.array([1.0, 2.0, 3.0, 4.0])
        probs = compute_world_probabilities(energies, temperature=1e6)
        # At very high T, all should be nearly equal
        np.testing.assert_allclose(probs, 0.25, atol=1e-3)

    def test_boltzmann_low_T_concentrated(self):
        from cp_bridge import compute_world_probabilities
        energies = np.array([1.0, 2.0, 3.0, 4.0])
        probs = compute_world_probabilities(energies, temperature=0.001)
        # At very low T, nearly all probability on minimum energy
        self.assertGreater(probs[0], 0.99)

    def test_boltzmann_zero_T(self):
        from cp_bridge import compute_world_probabilities
        energies = np.array([1.0, 2.0, 0.5, 4.0])
        probs = compute_world_probabilities(energies, temperature=0.0)
        self.assertAlmostEqual(probs[2], 1.0, places=10)  # min energy at idx 2

    def test_world_coherence(self):
        from cp_bridge import WorldState, compute_world_coherence
        worlds = [
            WorldState(np.ones(4), np.zeros(5), 1.0, 0.5, 0.25, 0),
            WorldState(np.ones(4), np.zeros(5), 2.0, 0.8, 0.25, 1),
            WorldState(np.ones(4), np.zeros(5), 0.5, 1.2, 0.25, 2),
        ]
        coh = compute_world_coherence(worlds)
        np.testing.assert_array_equal(coh, [0.5, 0.8, 1.2])


# =====================================================================
# Test: Implacement
# =====================================================================

class TestImplacement(unittest.TestCase):

    def test_implacement_reduces_eta(self):
        from cp_bridge import apply_implacement
        eta = np.array([0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0])
        pairs = [(0, 4), (1, 5)]
        eta_new, dphi = apply_implacement(eta, pairs, decay_rate=0.5)
        # F coils (4, 5) should have reduced η
        self.assertLess(eta_new[4], eta[4])
        self.assertLess(eta_new[5], eta[5])
        # M coils unchanged
        np.testing.assert_array_equal(eta_new[:4], eta[:4])
        # ΔΦ should be positive
        self.assertGreater(dphi, 0.0)


# =====================================================================
# Test: Pipeline Stage 12
# =====================================================================

class TestPipelineStage12(unittest.TestCase):
    """Integration tests for the full CP bridge analysis."""

    def setUp(self):
        self.basis, self.target_idx, self.surface = _make_basis()
        self.N = self.basis.shape[1]
        self.config = paper_baseline()

    def test_run_ti_mode(self):
        from cp_bridge import run_cp_bridge_analysis, print_cp_bridge_summary
        results = run_cp_bridge_analysis(
            self.basis, self.target_idx, self.surface,
            config=self.config, mode="TI",
        )
        self.assertEqual(results["mode"], "TI")
        self.assertEqual(results["n_coils"], self.N)
        self.assertTrue(np.isfinite(results["collective_phi"]))
        self.assertGreaterEqual(results["sync_order_parameter"], 0.0)
        self.assertLessEqual(results["sync_order_parameter"], 1.0)
        self.assertTrue(np.isfinite(results["group_free_energy"]))

        # Energy should be a dict with all 6 terms
        E = results["energy"]
        for k in ["E_nca", "E_mf", "E_arch", "E_phi", "E_couple", "E_morl", "E_total"]:
            self.assertIn(k, E)
            self.assertTrue(np.isfinite(E[k]), f"{k} should be finite")

        # Summary should be printable
        summary = print_cp_bridge_summary(results)
        self.assertIn("Collective Φ", summary)

    def test_run_nts_mode(self):
        from cp_bridge import run_cp_bridge_analysis
        results = run_cp_bridge_analysis(
            self.basis, self.target_idx, self.surface,
            config=self.config, mode="NTS",
        )
        self.assertEqual(results["mode"], "NTS")
        self.assertTrue(np.isfinite(results["collective_phi"]))

    def test_with_pareto_results(self):
        """Test many-worlds generation with Pareto front data."""
        from cp_bridge import run_cp_bridge_analysis
        from sensitivity import compute_pareto_front_ti

        N = self.N
        group = np.zeros(N)
        group[N // 2:] = 1.0
        f1 = self.config.ti.freq_carrier_hz
        f2 = f1 + self.config.ti.delta_freq_default_hz

        pareto = compute_pareto_front_ti(
            self.basis, group, f1, f2, self.target_idx, self.surface,
            alpha_max=1.0, n_weights=10, optimizer="random")

        results = run_cp_bridge_analysis(
            self.basis, self.target_idx, self.surface,
            config=self.config, mode="TI",
            pareto_results=pareto,
        )

        self.assertGreater(results["n_worlds"], 0, "Should have at least 1 world")
        # Probabilities should sum to 1
        probs = results["world_probabilities"]
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=8)
        world_outputs = np.array([w.outputs for w in results["worlds"]])
        self.assertEqual(world_outputs.shape[1], 5)
        # Temperature channel should reflect physical output, not placeholder zeros.
        self.assertTrue(np.all(world_outputs[:, 4] > 0.0))


if __name__ == "__main__":
    unittest.main()
