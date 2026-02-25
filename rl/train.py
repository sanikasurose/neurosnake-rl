import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from snake_env import SnakeEnv
from rl.agent import DQNAgent

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")


def train(
    num_episodes=5000,
    max_steps_per_episode=1000,
    save_model=True,
    model_path="dqn_snake.pth",
):
    env = SnakeEnv()
    agent = DQNAgent()

    episode_rewards = []
    episode_scores = []
    all_epsilons = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)
        episode_scores.append(env.score)

        # Decay epsilon once per episode
        agent.epsilon = max(
            agent.epsilon_min,
            agent.epsilon * agent.epsilon_decay
        )
        all_epsilons.append(agent.epsilon)

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(
                f"Episode {episode} | "
                f"Score: {env.score} | "
                f"Avg Reward (last 50): {avg_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

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

    if save_model:
        torch.save(agent.policy_net.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train(num_episodes=250)
