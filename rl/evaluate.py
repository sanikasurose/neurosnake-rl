"""Deterministic evaluation of a trained DQN Snake agent.

Loads a saved policy-network checkpoint and runs greedy (argmax Q) episodes
with no exploration, no replay-buffer updates, and no training.

Usage:
    python -m rl.evaluate --model models/exp_3_best.pth --episodes 50
"""

import argparse
import os
import random

import numpy as np
import torch

from snake_env import SnakeEnv
from rl.agent import DQNAgent


def evaluate(
    model_path: str,
    num_episodes: int,
    stack_size: int,
    seed: int,
    verbose: bool = True,
) -> list[int]:
    """Run deterministic greedy evaluation and return per-episode scores."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = SnakeEnv(stack_size=stack_size)
    agent = DQNAgent(stack_size=stack_size)

    state_dict = torch.load(model_path, map_location=agent.device)
    agent.policy_net.load_state_dict(state_dict)
    agent.policy_net.eval()
    agent.epsilon = 0.0

    scores: list[int] = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            state_t = state_t.unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.policy_net(state_t).argmax(dim=1).item()
            state, _, done = env.step(action)

        scores.append(env.score)
        if verbose:
            print(f"Episode {ep + 1:>{len(str(num_episodes))}}/{num_episodes}  "
                  f"Score: {env.score}")

    if verbose:
        avg_score = sum(scores) / len(scores)
        print("\n--- Evaluation Results ---")
        print(f"Episodes : {num_episodes}")
        print(f"Avg Score: {avg_score:.2f}")
        print(f"Max Score: {max(scores)}")
        print(f"Min Score: {min(scores)}")

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN Snake agent",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the saved model checkpoint (.pth)",
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of evaluation episodes (default: 50)",
    )
    parser.add_argument(
        "--stack-size", type=int, default=4,
        help="Frame-stack size matching the trained model (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    evaluate(
        model_path=args.model,
        num_episodes=args.episodes,
        stack_size=args.stack_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
