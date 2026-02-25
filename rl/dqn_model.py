"""Deep Q-Network (DQN) architecture for the Snake RL agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Fully-connected DQN that maps a 15x15 grid observation to Q-values
    for each of the 4 movement actions (up, down, left, right).

    Accepts input shaped as (batch, 15, 15) or (batch, 225); the network
    flattens internally so callers don't need to reshape.
    """

    def __init__(self, input_dim: int = 225, output_dim: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values for every action given an observation batch.

        Args:
            x: Observation tensor of shape (batch, 15, 15) or (batch, 225).

        Returns:
            Tensor of shape (batch, 4) containing raw Q-values.
        """
        if x.dim() > 2:
            x = x.flatten(start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == "__main__":
    model = DQN()
    dummy_input = torch.randn(1, 15, 15)
    q_values = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Q-values:     {q_values.detach().numpy()}")
