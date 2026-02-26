"""CNN-based Deep Q-Network for the Snake RL agent."""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """CNN that maps a (batch, C, 15, 15) spatial observation to Q-values
    for each of the 4 movement actions (up, down, left, right).
    """

    def __init__(self, action_dim: int = 4, input_channels: int = 2) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 15, 15)
            conv_out = self.conv(dummy)
            self.flattened_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = DQN()
    dummy_input = torch.randn(1, 2, 15, 15)
    q_values = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Q-values:     {q_values.detach().numpy()}")
    print(f"Flattened size: {model.flattened_size}")
