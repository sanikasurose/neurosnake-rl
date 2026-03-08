import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from snake_env import SnakeEnv
from rl.agent import DQNAgent
from analytics.db import init_db, create_experiment, log_episode, end_experiment
import random
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(_PROJECT_ROOT, "plots")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")


def train(
    num_episodes=5000,
    max_steps_per_episode=1000,
    stack_size=4,
    seed=42,
    lr=None,
):
    init_db()
    np.random.seed(seed)
    torch.manual_seed(seed)

    random.seed(seed)

    env = SnakeEnv(stack_size=stack_size)
    agent = DQNAgent(stack_size=stack_size)

    if lr is not None: 
        for param_group in agent.optimizer.param_groups: 
            param_group["lr"] = lr

    experiment_id = create_experiment(
        model_version="v0.5.0",
        learning_rate=agent.optimizer.param_groups[0]["lr"],
        gamma=agent.gamma,
        epsilon_start=agent.epsilon_start,
        epsilon_decay=agent.epsilon_decay_steps,
        batch_size=agent.batch_size,
        notes=f"seed={seed}"
    )
    experiment_model_path = os.path.join(MODELS_DIR, f"exp_{experiment_id}_best.pth")
    experiment_plot_dir = os.path.join(PLOT_DIR, f"exp_{experiment_id}")

    os.makedirs(experiment_plot_dir, exist_ok=True)

    episode_rewards = []
    scores = []
    rolling_avg_scores = []
    all_epsilons = []
    all_losses = []
    all_mean_qs = []
    all_mean_target_qs = []
    best_score = 0
    breakthrough_saved = False

    os.makedirs(MODELS_DIR, exist_ok=True)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_loss = 0.0
        episode_mean_q = 0.0
        episode_mean_target_q = 0.0
        train_updates = 0
        episode_steps = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            metrics = agent.train_step()

            if metrics is not None:
                episode_loss += metrics["loss"]
                episode_mean_q += metrics["mean_q"]
                episode_mean_target_q += metrics["mean_target_q"]
                train_updates += 1

            state = next_state
            total_reward += reward
            episode_steps += 1

            if done:
                break

        if train_updates > 0:
            avg_loss = episode_loss / train_updates
            avg_q = episode_mean_q / train_updates
            avg_target_q = episode_mean_target_q / train_updates
        else:
            avg_loss = 0.0
            avg_q = 0.0
            avg_target_q = 0.0

        episode_rewards.append(total_reward)
        score = env.score
        scores.append(score)

        all_epsilons.append(agent.epsilon)
        all_losses.append(avg_loss)
        all_mean_qs.append(avg_q)
        all_mean_target_qs.append(avg_target_q)

        if len(scores) >= 100:
            rolling_avg = sum(scores[-100:]) / 100
        else:
            rolling_avg = sum(scores) / len(scores)
        rolling_avg_scores.append(rolling_avg)

        if score > best_score:
            best_score = score
            torch.save(agent.policy_net.state_dict(), experiment_model_path)
            print(f"New Best Score: {best_score} — model saved to {experiment_model_path}")

        if not breakthrough_saved and rolling_avg >= 10:
            path = os.path.join(MODELS_DIR, "frame_stack_breakthrough.pth")
            torch.save(agent.policy_net.state_dict(), path)
            print(f"Model saved — achieved rolling avg >= 10 ({path})")
            breakthrough_saved = True

        if episode % 50 == 0:
            print(
                f"Episode {episode} | "
                f"Score: {score} | "
                f"Steps: {episode_steps} | "
                f"Rolling Avg (100): {rolling_avg:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"MeanQ: {avg_q:.2f} | "
                f"MeanTargetQ: {avg_target_q:.2f} | "
                f"Buffer: {len(agent.memory)}"
            )
            if abs(avg_q - avg_target_q) > 10:
                print("⚠ WARNING: Q/Target divergence detected")

        log_episode(
            experiment_id=experiment_id,
            episode_number=episode,
            score=score,
            total_reward=total_reward,
            avg_loss=avg_loss,
            epsilon=agent.epsilon,
            mean_q=avg_q,
            mean_target_q=avg_target_q,
        )

    end_experiment(experiment_id)

    # ---- plots ----
    os.makedirs(PLOT_DIR, exist_ok=True)

    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward vs Episode")
    plt.savefig(os.path.join(experiment_plot_dir, "reward_plot.png"))
    plt.close()

    plt.figure()
    plt.plot(all_epsilons)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.savefig(os.path.join(experiment_plot_dir, "epsilon_plot.png"))
    plt.close()

    moving_avg = np.convolve(episode_rewards, np.ones(50) / 50, mode='valid')
    plt.figure()
    plt.plot(moving_avg)
    plt.xlabel("Episode")
    plt.ylabel("Moving Avg Reward (50)")
    plt.title("Smoothed Reward Curve")
    plt.savefig(os.path.join(experiment_plot_dir, "reward_smoothed.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Snake Agent")

    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--stack-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=None)

    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        stack_size=args.stack_size,
        seed=args.seed,
        lr=args.lr,
    )