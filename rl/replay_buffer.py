"""Experience Replay Buffer for DQN training.

Stores (state, action, reward, next_state, done) transitions in a fixed-size
circular buffer and provides uniform random sampling as PyTorch tensors.
"""

from collections import deque
import random

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer that stores experience tuples and samples random batches."""

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of transitions the buffer can hold.
                      Once full, the oldest transitions are discarded.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Append a single transition to the buffer.

        Args:
            state: numpy array of shape (2, grid_size, grid_size).
            action: int in [0, 3].
            reward: float reward signal.
            next_state: numpy array of shape (2, grid_size, grid_size).
            done: bool indicating episode termination.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Randomly sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors.
              - states:      float32, shape (batch_size, 2, grid_size, grid_size)
              - actions:     long,    shape (batch_size,)
              - rewards:     float32, shape (batch_size,)
              - next_states: float32, shape (batch_size, 2, grid_size, grid_size)
              - dones:       float32, shape (batch_size,)
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current number of stored transitions."""
        return len(self.memory)


if __name__ == "__main__":
    buf = ReplayBuffer(capacity=100)

    for _ in range(10):
        s = np.random.rand(2, 15, 15).astype(np.float32)
        a = random.randint(0, 3)
        r = random.uniform(-1.0, 1.0)
        s_next = np.random.rand(2, 15, 15).astype(np.float32)
        d = random.choice([True, False])
        buf.push(s, a, r, s_next, d)

    print(f"Buffer size: {len(buf)}")

    states, actions, rewards, next_states, dones = buf.sample(batch_size=4)

    print(f"states      : {states.shape}  dtype={states.dtype}")
    print(f"actions     : {actions.shape}  dtype={actions.dtype}")
    print(f"rewards     : {rewards.shape}  dtype={rewards.dtype}")
    print(f"next_states : {next_states.shape}  dtype={next_states.dtype}")
    print(f"dones       : {dones.shape}  dtype={dones.dtype}")
