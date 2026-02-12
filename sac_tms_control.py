"""Soft Actor-Critic (SAC) agent for real-time TMS array control.

Provides three physics-based environments:
  - ``BrainEnv_TI``: Temporal Interference amplitude/frequency control
  - ``BrainEnv_NTS``: Neural Temporal Summation amplitude/timing control
  - ``BrainEnv_Hybrid``: combined TI + NTS

The RL infrastructure (ReplayBuffer, MLP, GaussianPolicy, SACAgent) is
generic and shared across all environments.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cp_bridge import scalarize_morl
from config import OmnidreamConfig, paper_baseline
from control_framework import map_output_vector
from em_theory import SARThermal
from ti_fields import (
    compute_group_amplitudes,
    compute_modulation_depth,
    compute_ti_sar,
)
from nts_timing import (
    compute_v_peak,
    compute_per_pulse_surface_max,
    enforce_guard_times,
)


DEFAULT_MORL_WEIGHTS = {"J_phi": 1.0, "J_arch": -1.0, "J_sync": 1.0, "J_task": 1.0}


def _selectivity_score(target_metric: float, surface_metric: float) -> float:
    denom = abs(target_metric) + abs(surface_metric) + 1e-9
    return float((target_metric - surface_metric) / denom)


def _group_balance_sync(alphas: np.ndarray, group: np.ndarray) -> float:
    g0 = group < 0.5
    g1 = ~g0
    if not np.any(g0) or not np.any(g1):
        return 1.0
    p0 = float(np.mean(alphas[g0]))
    p1 = float(np.mean(alphas[g1]))
    denom = abs(p0) + abs(p1) + 1e-9
    return float(1.0 - abs(p0 - p1) / denom)


def _timing_sync(t_fire: np.ndarray, tau_window: float) -> float:
    if t_fire.size <= 1:
        return 1.0
    spread = float(np.std(t_fire))
    return float(np.clip(1.0 - spread / (tau_window + 1e-9), 0.0, 1.0))


def _build_morl_objectives(
    target_metric: float,
    surface_metric: float,
    sync_score: float,
    target_ref: float = 1.0,
    surface_ref: float = 0.0,
) -> dict[str, float]:
    return {
        "J_phi": _selectivity_score(target_metric, surface_metric),
        "J_arch": float((target_metric - target_ref) ** 2 + (surface_metric - surface_ref) ** 2),
        "J_sync": float(np.clip(sync_score, 0.0, 1.0)),
        "J_task": float(target_metric),
    }


# =====================================================================
# Environments
# =====================================================================

class BrainEnv_TI:
    """Physics-based environment for TI amplitude and frequency control.

    The GA (outer loop) fixes coil positions and group assignments.
    The SAC agent adjusts per-coil amplitudes and beat frequency in real time.

    Action space (N + 1):
      - action[:N]  → per-coil amplitude weights  ∈ [0, α_max]
      - action[N]   → Δf adjustment              ∈ [1, 100] Hz
    """

    def __init__(
        self,
        basis_matrix: np.ndarray,
        target_idx: int | np.ndarray,
        surface_indices: np.ndarray,
        group_assignment: np.ndarray | None = None,
        config: OmnidreamConfig | None = None,
        max_steps: int = 50,
        morl_weights: dict[str, float] | None = None,
    ):
        if config is None:
            config = paper_baseline()
        self.config = config
        self.basis_matrix = basis_matrix
        self.target_idx = target_idx
        self.surface_indices = surface_indices

        self.n_coils = basis_matrix.shape[1]
        self.action_dim = self.n_coils + 1
        self.action_range = (-1.0, 1.0)
        self.max_steps = max_steps

        if group_assignment is None:
            self.group = np.random.randint(0, 2, size=self.n_coils).astype(float)
        else:
            self.group = np.asarray(group_assignment, float)

        self.freq_carrier = config.ti.freq_carrier_hz
        self.tau_m = config.nts.tau_m_s
        self.alpha_max = config.safety.max_current_a
        self.morl_weights = morl_weights or DEFAULT_MORL_WEIGHTS.copy()

        # State: [prev_amplitudes(N), prev_M_target(1), prev_M_surface_max(1), target_coords_placeholder(3)]
        self.state_dim = self.n_coils + 5
        self.current_step = 0
        self.prev_amplitudes = np.zeros(self.n_coils, dtype=np.float32)
        self.prev_M_target = 0.0
        self.prev_M_surface_max = 0.0

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.prev_amplitudes = np.zeros(self.n_coils, dtype=np.float32)
        self.prev_M_target = 0.0
        self.prev_M_surface_max = 0.0
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=float)

        # Decode actions
        alphas = (action[:self.n_coils] + 1.0) / 2.0 * self.alpha_max
        delta_f = (action[self.n_coils] + 1.0) / 2.0 * 99.0 + 1.0  # [1, 100] Hz

        f1 = self.freq_carrier
        f2 = f1 + delta_f

        # Forward model
        A1, A2 = compute_group_amplitudes(alphas, self.group, self.basis_matrix, f1, f2)
        M = compute_modulation_depth(A1, A2)

        M_target = float(np.mean(M[self.target_idx]))
        M_surface = M[self.surface_indices]
        M_surface_max = float(np.max(M_surface)) if len(M_surface) > 0 else 0.0

        # SAR estimation
        SAR = compute_ti_sar(A1, A2, safety_cfg=self.config.safety)
        SAR_max = float(np.max(SAR[self.surface_indices])) if len(self.surface_indices) > 0 else 0.0
        temperature_max = 37.0 + SARThermal.pennes_steady_state_delta_T(SAR_max)

        beat_gain = 1.0 / np.sqrt(1.0 + (2.0 * np.pi * delta_f * self.tau_m) ** 2)
        membrane_metric = float(M_target * beat_gain)

        sync_score = _group_balance_sync(alphas, self.group)
        morl_objectives = _build_morl_objectives(M_target, M_surface_max, sync_score)
        morl_reward = float(scalarize_morl(morl_objectives, self.morl_weights))

        # MORL-shaped reward
        reward = morl_reward

        # Safety penalty
        if SAR_max > self.config.safety.sar_limit_wkg:
            reward -= 100.0
        if M_surface_max > M_target * 0.5:
            reward -= 50.0

        # Update state tracking
        self.prev_amplitudes = alphas.astype(np.float32)
        self.prev_M_target = M_target
        self.prev_M_surface_max = M_surface_max
        self.current_step += 1
        done = self.current_step >= self.max_steps

        y_map = map_output_vector(
            np.array([M_target, M_surface_max, membrane_metric, SAR_max, temperature_max])
        )
        info = {
            **y_map,
            "M_target": M_target,
            "M_surface_max": M_surface_max,
            "SAR_max": SAR_max,
            "morl_objectives": morl_objectives,
            "morl_reward": morl_reward,
            "sync_score": sync_score,
        }
        return self._observe(), reward, done, info

    def _observe(self) -> np.ndarray:
        return np.concatenate([
            self.prev_amplitudes,
            np.array([self.prev_M_target, self.prev_M_surface_max], dtype=np.float32),
            np.zeros(3, dtype=np.float32),  # target coords placeholder
        ])


class BrainEnv_NTS:
    """Physics-based environment for NTS amplitude and timing control.

    Action space (2N):
      - action[:N]    → per-coil amplitude weights  ∈ [0, α_max]
      - action[N:2N]  → per-coil timing offsets      ∈ [0, τ_window]
    """

    def __init__(
        self,
        basis_matrix: np.ndarray,
        target_idx: int | np.ndarray,
        surface_indices: np.ndarray,
        config: OmnidreamConfig | None = None,
        max_steps: int = 50,
        morl_weights: dict[str, float] | None = None,
    ):
        if config is None:
            config = paper_baseline()
        self.config = config
        self.basis_matrix = basis_matrix
        self.target_idx = target_idx
        self.surface_indices = surface_indices

        self.n_coils = basis_matrix.shape[1]
        self.action_dim = 2 * self.n_coils
        self.action_range = (-1.0, 1.0)
        self.max_steps = max_steps

        self.tau_m = config.nts.tau_m_s
        self.tau_window = config.nts.tau_window_s
        self.T_guard = config.nts.t_guard_s
        self.alpha_max = config.safety.max_current_a
        self.morl_weights = morl_weights or DEFAULT_MORL_WEIGHTS.copy()

        self.state_dim = self.n_coils + 5
        self.current_step = 0
        self.prev_amplitudes = np.zeros(self.n_coils, dtype=np.float32)
        self.prev_V_target = 0.0
        self.prev_V_surface_max = 0.0

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.prev_amplitudes = np.zeros(self.n_coils, dtype=np.float32)
        self.prev_V_target = 0.0
        self.prev_V_surface_max = 0.0
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=float)
        N = self.n_coils

        alphas = (action[:N] + 1.0) / 2.0 * self.alpha_max
        t_raw = (action[N:2 * N] + 1.0) / 2.0 * self.tau_window
        t_fire = enforce_guard_times(t_raw, self.T_guard)

        # Forward model
        V_peak = compute_v_peak(
            alphas, t_fire, self.basis_matrix,
            tau_m=self.tau_m, q_pulse=self.config.nts.q_pulse,
        )

        V_target = float(np.mean(V_peak[self.target_idx]))
        V_surface = V_peak[self.surface_indices]
        V_surface_max = float(np.max(V_surface)) if len(V_surface) > 0 else 0.0

        per_pulse_max = compute_per_pulse_surface_max(alphas, self.basis_matrix, self.surface_indices)
        E_rms = alphas * np.mean(np.abs(self.basis_matrix), axis=0)
        SAR = SARThermal.sar_pointwise(
            E_rms,
            sigma_sm=self.config.safety.sigma_gm_sm,
            rho_kgm3=self.config.safety.rho_tissue_kgm3,
        )
        SAR_max = float(np.max(SAR)) * self.config.safety.max_duty_cycle_pulsed
        temperature_max = 37.0 + SARThermal.pennes_steady_state_delta_T(SAR_max)

        sync_score = _timing_sync(t_fire, self.tau_window)
        morl_objectives = _build_morl_objectives(V_target, V_surface_max, sync_score)
        morl_reward = float(scalarize_morl(morl_objectives, self.morl_weights))

        # MORL-shaped reward
        reward = morl_reward

        # Safety penalty
        if np.max(per_pulse_max) > self.config.safety.surface_e_threshold_vpm:
            reward -= 100.0
        if SAR_max > self.config.safety.sar_limit_wkg:
            reward -= 100.0

        self.prev_amplitudes = alphas.astype(np.float32)
        self.prev_V_target = V_target
        self.prev_V_surface_max = V_surface_max
        self.current_step += 1
        done = self.current_step >= self.max_steps

        y_map = map_output_vector(
            np.array([V_target, V_surface_max, V_target, SAR_max, temperature_max])
        )
        info = {
            **y_map,
            "V_target": V_target,
            "V_surface_max": V_surface_max,
            "SAR_max": SAR_max,
            "morl_objectives": morl_objectives,
            "morl_reward": morl_reward,
            "sync_score": sync_score,
        }
        return self._observe(), reward, done, info

    def _observe(self) -> np.ndarray:
        return np.concatenate([
            self.prev_amplitudes,
            np.array([self.prev_V_target, self.prev_V_surface_max], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ])


class BrainEnv_Hybrid:
    """Combined TI + NTS environment.

    Action space (2N + 1):
      - action[:N]      → per-coil amplitudes     ∈ [0, α_max]
      - action[N:2N]    → intra-cycle timing offsets
      - action[2N]      → Δf adjustment
    """

    def __init__(
        self,
        basis_matrix: np.ndarray,
        target_idx: int | np.ndarray,
        surface_indices: np.ndarray,
        group_assignment: np.ndarray | None = None,
        config: OmnidreamConfig | None = None,
        max_steps: int = 50,
        w_ti: float = 0.6,
        w_nts: float = 0.4,
        morl_weights: dict[str, float] | None = None,
    ):
        if config is None:
            config = paper_baseline()
        self.config = config
        self.basis_matrix = basis_matrix
        self.target_idx = target_idx
        self.surface_indices = surface_indices
        self.w_ti = w_ti
        self.w_nts = w_nts

        self.n_coils = basis_matrix.shape[1]
        self.action_dim = 2 * self.n_coils + 1
        self.action_range = (-1.0, 1.0)
        self.max_steps = max_steps

        if group_assignment is None:
            self.group = np.random.randint(0, 2, size=self.n_coils).astype(float)
        else:
            self.group = np.asarray(group_assignment, float)

        self.freq_carrier = config.ti.freq_carrier_hz
        self.alpha_max = config.safety.max_current_a
        self.tau_m = config.nts.tau_m_s
        self.tau_window = config.nts.tau_window_s
        self.T_guard = config.nts.t_guard_s
        self.morl_weights = morl_weights or DEFAULT_MORL_WEIGHTS.copy()

        self.state_dim = self.n_coils + 5
        self.current_step = 0
        self.prev_amplitudes = np.zeros(self.n_coils, dtype=np.float32)
        self.prev_combined_target = 0.0
        self.prev_combined_surface_max = 0.0

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.prev_amplitudes = np.zeros(self.n_coils, dtype=np.float32)
        self.prev_combined_target = 0.0
        self.prev_combined_surface_max = 0.0
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=float)
        N = self.n_coils

        alphas = (action[:N] + 1.0) / 2.0 * self.alpha_max
        t_raw = (action[N:2 * N] + 1.0) / 2.0 * self.tau_window
        t_fire = enforce_guard_times(t_raw, self.T_guard)
        delta_f = (action[2 * N] + 1.0) / 2.0 * 99.0 + 1.0

        f1 = self.freq_carrier
        f2 = f1 + delta_f

        # TI component
        A1, A2 = compute_group_amplitudes(alphas, self.group, self.basis_matrix, f1, f2)
        M = compute_modulation_depth(A1, A2)
        M_target = float(np.mean(M[self.target_idx]))
        M_surface_max = float(np.max(M[self.surface_indices])) if len(self.surface_indices) > 0 else 0.0

        # NTS component
        V_peak = compute_v_peak(alphas, t_fire, self.basis_matrix, self.tau_m, self.config.nts.q_pulse)
        V_target = float(np.mean(V_peak[self.target_idx]))
        V_surface_max = float(np.max(V_peak[self.surface_indices])) if len(self.surface_indices) > 0 else 0.0

        # SAR penalty
        SAR = compute_ti_sar(A1, A2, safety_cfg=self.config.safety)
        SAR_max = float(np.max(SAR[self.surface_indices])) if len(self.surface_indices) > 0 else 0.0
        temperature_max = 37.0 + SARThermal.pennes_steady_state_delta_T(SAR_max)

        combined_target = float(self.w_ti * M_target + self.w_nts * V_target)
        combined_surface_max = float(max(M_surface_max, V_surface_max))
        sync_score = 0.5 * _group_balance_sync(alphas, self.group) + 0.5 * _timing_sync(t_fire, self.tau_window)
        morl_objectives = _build_morl_objectives(combined_target, combined_surface_max, sync_score)
        morl_reward = float(scalarize_morl(morl_objectives, self.morl_weights))

        # MORL-shaped reward
        reward = morl_reward

        if SAR_max > self.config.safety.sar_limit_wkg:
            reward -= 100.0

        self.prev_amplitudes = alphas.astype(np.float32)
        self.prev_combined_target = combined_target
        self.prev_combined_surface_max = combined_surface_max
        self.current_step += 1
        done = self.current_step >= self.max_steps

        y_map = map_output_vector(
            np.array([combined_target, combined_surface_max, combined_target, SAR_max, temperature_max])
        )
        info = {
            **y_map,
            "target_metric": combined_target,
            "surface_metric": combined_surface_max,
            "M_target": M_target, "V_target": V_target,
            "M_surface_max": M_surface_max, "V_surface_max": V_surface_max,
            "SAR_max": SAR_max,
            "morl_objectives": morl_objectives,
            "morl_reward": morl_reward,
            "sync_score": sync_score,
        }
        return self._observe(), reward, done, info

    def _observe(self) -> np.ndarray:
        return np.concatenate([
            self.prev_amplitudes,
            np.array([self.prev_combined_target, self.prev_combined_surface_max], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ])


# =====================================================================
# RL Infrastructure (unchanged from original)
# =====================================================================

class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, s, a, r, s_next, d):
        self.buffer.append((s, a, r, s_next, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(s_next), np.array(d)

    def __len__(self):
        return len(self.buffer)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.apply(init_weights)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 action_range=(-1, 1)):
        super().__init__()
        self.action_dim = action_dim
        self.action_range = action_range
        self.fc_mean = MLP(state_dim, action_dim, hidden_dim)
        self.fc_logstd = MLP(state_dim, action_dim, hidden_dim)

    def forward(self, state):
        mean = self.fc_mean(state)
        log_std = self.fc_logstd(state)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, torch.exp(log_std)

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        a_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - a_t.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return a_t, log_prob, mean, std

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            a, _, _, _ = self.sample(state)
        return a.cpu().numpy()[0]


class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, action_range=(-1, 1),
                 gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4,
                 batch_size=64, buffer_size=100000):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = GaussianPolicy(state_dim, action_dim, action_range=action_range).to(self.device)
        self.q1 = MLP(state_dim + action_dim, 1).to(self.device)
        self.q2 = MLP(state_dim + action_dim, 1).to(self.device)
        self.q1_target = MLP(state_dim + action_dim, 1).to(self.device)
        self.q2_target = MLP(state_dim + action_dim, 1).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        return self.policy.select_action(state)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        s, a, r, s_next, d = self.replay_buffer.sample(self.batch_size)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(-1).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            a_next, log_prob_next, _, _ = self.policy.sample(s_next)
            q1_next = self.q1_target(torch.cat([s_next, a_next], dim=1))
            q2_next = self.q2_target(torch.cat([s_next, a_next], dim=1))
            q_next = torch.min(q1_next, q2_next) - self.alpha * log_prob_next
            target_q = r + (1 - d) * self.gamma * q_next

        q1_pred = self.q1(torch.cat([s, a], dim=1))
        q2_pred = self.q2(torch.cat([s, a], dim=1))
        q1_loss = torch.mean((q1_pred - target_q) ** 2)
        q2_loss = torch.mean((q2_pred - target_q) ** 2)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        a_sample, log_prob, _, _ = self.policy.sample(s)
        q1_val = self.q1(torch.cat([s, a_sample], dim=1))
        q2_val = self.q2(torch.cat([s, a_sample], dim=1))
        q_val = torch.min(q1_val, q2_val)
        policy_loss = (self.alpha * log_prob - q_val).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# =====================================================================
# Training loop
# =====================================================================

def train_sac(env, episodes: int = 100, steps_per_episode: int = 50, verbose: bool = True):
    """Train an SAC agent on the given environment."""
    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        action_range=env.action_range,
        gamma=0.99, tau=0.005, alpha=0.2,
        lr=3e-4, batch_size=64, buffer_size=50000,
    )

    history = []
    for ep in range(episodes):
        s = env.reset()
        ep_reward = 0.0
        for t in range(steps_per_episode):
            a = agent.select_action(s)
            s_next, r, done, info = env.step(a)
            agent.replay_buffer.add(s, a, r, s_next, done)
            agent.update()
            s = s_next
            ep_reward += r
            if done:
                break
        history.append({"episode": ep, "reward": ep_reward, **info})
        if verbose and ep % 10 == 0:
            print(f"Episode {ep:4d}: reward={ep_reward:.3f}")

    return agent, history


if __name__ == "__main__":
    from basis_fields import generate_synthetic_basis

    basis, target_idx, surface_indices = generate_synthetic_basis(n_coils=8)
    print(f"Basis shape: {basis.shape}")

    print("\n--- Training TI agent ---")
    env_ti = BrainEnv_TI(basis, target_idx, surface_indices)
    agent_ti, hist_ti = train_sac(env_ti, episodes=30)

    print("\n--- Training NTS agent ---")
    env_nts = BrainEnv_NTS(basis, target_idx, surface_indices)
    agent_nts, hist_nts = train_sac(env_nts, episodes=30)

    print("\n--- Training Hybrid agent ---")
    env_hyb = BrainEnv_Hybrid(basis, target_idx, surface_indices)
    agent_hyb, hist_hyb = train_sac(env_hyb, episodes=30)

    print("\nDone.")
