import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

############################################################
# Environment Placeholder
############################################################

class BrainEnv:
    """
    A mock environment simulating brain states and rewards.
    Replace this with actual logic:
    - Load or compute current brain state s(t).
    - Given action a(t) (coil intensities), compute next state s(t+1).
    - Compute reward based on how close s(t+1) is to target pattern T(t+1).
    """
    def __init__(self, coil_count=5, state_dim=10, target_pattern=None):
        self.coil_count = coil_count
        self.action_dim = coil_count  # actions = coil intensities
        self.state_dim = state_dim
        self.action_range = (-1.0, 1.0)  # Just as a placeholder
        self.current_step = 0
        self.max_steps = 50

        # If target_pattern is not given, create a random target
        if target_pattern is None:
            self.target_pattern = np.random.uniform(-1, 1, size=(self.max_steps, self.state_dim))
        else:
            self.target_pattern = target_pattern

        self.state = np.zeros(self.state_dim, dtype=np.float32)

    def reset(self):
        self.current_step = 0
        # Start from some initial state (random or fixed)
        self.state = np.random.uniform(-0.5, 0.5, size=self.state_dim).astype(np.float32)
        return self.state

    def step(self, action):
        # Action: coil intensities
        # In a real scenario:
        # 1. Run forward model or call SimNIBS with these coil intensities.
        # 2. Obtain new brain state (e.g., from simulated or measured data).
        # Here we simulate next state randomly influenced by action.
        
        # Mock transition: next state = current_state + noise + scaled action
        noise = np.random.normal(0, 0.05, size=self.state_dim)
        self.state = self.state + 0.1 * action + noise

        # Compute reward
        # target for this timestep:
        target = self.target_pattern[self.current_step]
        # reward = negative MSE
        mse = np.mean((self.state - target)**2)
        reward = -mse

        self.current_step += 1
        done = (self.current_step >= self.max_steps)

        return self.state, reward, done, {}

############################################################
# Soft Actor-Critic Agent Implementation
############################################################

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s_next, d):
        self.buffer.append((s, a, r, s_next, d))

    def sample(self, batch_size):
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
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.apply(init_weights)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_range=(-1,1)):
        super(GaussianPolicy, self).__init__()
        self.action_dim = action_dim
        self.action_range = action_range
        self.fc_mean = MLP(state_dim, action_dim, hidden_dim)
        self.fc_logstd = MLP(state_dim, action_dim, hidden_dim)

    def forward(self, state):
        mean = self.fc_mean(state)
        log_std = self.fc_logstd(state)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        # Apply Tanh
        a_t = torch.tanh(x_t)
        # Log probability correction for Tanh
        log_prob = normal.log_prob(x_t) - torch.log(1 - a_t.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        # Scale to action range if needed
        # If action_range is other than [-1,1], scale here.
        return a_t, log_prob, mean, std

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            a, _, _, _ = self.sample(state)
        return a.cpu().numpy()[0]

class SACAgent:
    def __init__(self, state_dim, action_dim, action_range=(-1,1),
                 gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4, batch_size=64, buffer_size=100000):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.action_range = action_range

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor
        self.policy = GaussianPolicy(state_dim, action_dim, action_range=action_range).to(self.device)
        # Critics
        self.q1 = MLP(state_dim+action_dim, 1).to(self.device)
        self.q2 = MLP(state_dim+action_dim, 1).to(self.device)
        self.q1_target = MLP(state_dim+action_dim, 1).to(self.device)
        self.q2_target = MLP(state_dim+action_dim, 1).to(self.device)
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

        # Update Critic
        with torch.no_grad():
            a_next, log_prob_next, _, _ = self.policy.sample(s_next)
            q1_next = self.q1_target(torch.cat([s_next, a_next], dim=1))
            q2_next = self.q2_target(torch.cat([s_next, a_next], dim=1))
            q_next = torch.min(q1_next, q2_next) - self.alpha * log_prob_next
            target_q = r + (1 - d) * self.gamma * q_next

        q1_pred = self.q1(torch.cat([s, a], dim=1))
        q2_pred = self.q2(torch.cat([s, a], dim=1))
        q1_loss = torch.mean((q1_pred - target_q)**2)
        q2_loss = torch.mean((q2_pred - target_q)**2)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # Update Actor
        a_sample, log_prob, _, _ = self.policy.sample(s)
        q1_val = self.q1(torch.cat([s, a_sample], dim=1))
        q2_val = self.q2(torch.cat([s, a_sample], dim=1))
        q_val = torch.min(q1_val, q2_val)

        policy_loss = (self.alpha * log_prob - q_val).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Soft update targets
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

############################################################
# Main Training Loop
############################################################

if __name__ == "__main__":
    # Example usage
    env = BrainEnv(coil_count=5, state_dim=10)
    agent = SACAgent(
        state_dim=env.state_dim, 
        action_dim=env.action_dim, 
        action_range=env.action_range,
        gamma=0.99, tau=0.005, alpha=0.2,
        lr=3e-4, batch_size=64, buffer_size=10000
    )

    episodes = 10
    steps_per_episode = 50

    for ep in range(episodes):
        s = env.reset()
        episode_reward = 0.0
        for t in range(steps_per_episode):
            a = agent.select_action(s)
            s_next, r, done, info = env.step(a)
            agent.replay_buffer.add(s, a, r, s_next, done)

            agent.update()

            s = s_next
            episode_reward += r
            if done:
                break
        print(f"Episode {ep}, Reward: {episode_reward:.2f}")