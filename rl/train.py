import csv
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from snake_env import SnakeEnv
from rl.agent import DQNAgent

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(_PROJECT_ROOT, "plots")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")


def train(
    num_episodes=5000,
    max_steps_per_episode=1000,
    stack_size=4,
    save_model=True,
    model_path="dqn_snake.pth",
):
    env = SnakeEnv(stack_size=stack_size)
    agent = DQNAgent(stack_size=stack_size)

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
            print(f"New Best Score: {best_score}")

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

    # ---- plots ----
    os.makedirs(PLOT_DIR, exist_ok=True)

    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward vs Episode")
    plt.savefig(os.path.join(PLOT_DIR, "reward_plot.png"))
    plt.close()

    plt.figure()
    plt.plot(all_epsilons)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.savefig(os.path.join(PLOT_DIR, "epsilon_plot.png"))
    plt.close()

    moving_avg = np.convolve(episode_rewards, np.ones(50) / 50, mode='valid')
    plt.figure()
    plt.plot(moving_avg)
    plt.xlabel("Episode")
    plt.ylabel("Moving Avg Reward (50)")
    plt.title("Smoothed Reward Curve")
    plt.savefig(os.path.join(PLOT_DIR, "reward_smoothed.png"))
    plt.close()

    # ---- CSV log ----
    csv_path = os.path.join(_PROJECT_ROOT, "training_log_phase2.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Score", "RollingAvg", "Loss", "MeanQ", "MeanTargetQ"])
        for i in range(len(scores)):
            writer.writerow([
                i,
                scores[i],
                rolling_avg_scores[i],
                all_losses[i],
                all_mean_qs[i],
                all_mean_target_qs[i],
            ])
    print(f"Training log saved to {csv_path}")

    if save_model:
        torch.save(agent.policy_net.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train(num_episodes=4000)
