"""DQN Agent that integrates the policy/target networks, replay buffer,
epsilon-greedy exploration, and the single-step training update."""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.dqn_model import DQN
from rl.replay_buffer import ReplayBuffer


class DQNAgent:
    """Deep Q-Network agent for the Snake environment."""

    def __init__(
        self,
        action_dim=4,
        lr=3e-4,
        gamma=0.99,
        buffer_capacity=50_000,
        batch_size=128,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=30_000,
        target_update_freq=1000,
        stack_size=1,
        tau=0.01,
    ):
        """Initialise networks, optimizer, replay buffer, and hyperparameters."""
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.stack_size = stack_size
        input_channels = 2 * self.stack_size
        self.policy_net = DQN(action_dim, input_channels=input_channels).to(self.device)
        self.target_net = DQN(action_dim, input_channels=input_channels).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5,
        )
        self.criterion = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(buffer_capacity)

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.steps_done = 0
        self.action_dim = action_dim

    def select_action(self, state):
        """Choose an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(np.array(state), dtype=torch.float32)
        state_t = state_t.unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Save a single experience tuple to the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """Run one gradient update on a batch sampled from the replay buffer."""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        if not hasattr(self, '_shape_printed'):
            print(f"[DEBUG train_step] states.shape={states.shape}, "
                  f"next_states.shape={next_states.shape}")
            self._shape_printed = True

        q_values = self.policy_net(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values_target = self.target_net(next_states)
            next_q = next_q_values_target.gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft update of target network (Polyak averaging)
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

        self.steps_done += 1

        # Linear epsilon decay based on total training steps
        fraction = min(1.0, self.steps_done / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start - fraction * (
            self.epsilon_start - self.epsilon_min
        )

        mean_q = current_q.mean().item()
        mean_target_q = target_q.mean().item()

        return {
            "loss": loss.item(),
            "mean_q": mean_q,
            "mean_target_q": mean_target_q,
        }
