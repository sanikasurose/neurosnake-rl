"""Deterministic evaluation of a trained DQN Snake agent."""

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

    expected_channels = 2 * stack_size

    scores: list[int] = []

    for ep in range(num_episodes):

        state = env.reset()
        done = False

        max_steps = 500
        steps = 0

        while not done and steps < max_steps:

            steps += 1

            state_t = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=agent.device,
            ).unsqueeze(0)

            assert state_t.shape == (
                1,
                expected_channels,
                env.grid_size,
                env.grid_size,
            )

            with torch.no_grad():
                action = agent.policy_net(state_t).argmax(dim=1).item()

            state, _, done = env.step(action)

        scores.append(env.score)

        if verbose:
            print(
                f"Episode {ep+1}/{num_episodes} "
                f"Score: {env.score} "
                f"Steps: {steps}"
            )

    if verbose:
        avg_score = sum(scores) / len(scores)

        print("\n--- Evaluation Results ---")
        print(f"Episodes : {num_episodes}")
        print(f"Avg Score: {avg_score:.2f}")
        print(f"Max Score: {max(scores)}")
        print(f"Min Score: {min(scores)}")

    return scores


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--stack-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(args.model)

    evaluate(
        model_path=args.model,
        num_episodes=args.episodes,
        stack_size=args.stack_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()