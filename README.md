# 🐍 Snake AI — From Turtle Game to Deep Q-Learning

This project began as a simple implementation of the classic Snake game using Python’s Turtle graphics. It has since evolved into a reinforcement learning experiment where a Deep Q-Network (DQN) agent learns to play the game autonomously.

This repository documents that progression — from deterministic game logic to applied deep reinforcement learning.

---

# 🚀 Current Release: Basic DQN Agent

This release introduces a fully working Deep Q-Learning agent trained to play Snake using PyTorch.

The agent:

- Observes the game state as a grid representation
- Learns via reward-driven trial and error
- Uses experience replay
- Implements epsilon-greedy exploration
- Optimizes Q-values using gradient descent

At this stage, the model uses a fully connected neural network (MLP) and has plateaued around score 2–3 after 5000 episodes — which is expected given architectural limits.

This commit represents the first stable reinforcement learning version.

---

# 🧠 How the Agent Works

At its core, this project is applied mathematics.

There is no "intelligence" in the traditional sense — only:

- Function approximation
- Optimization
- Matrix multiplication
- Probability
- Gradient descent

The agent learns by approximating a function: 

Q(s, a) ≈ expected future reward

Where:

- `s` = state (the grid)
- `a` = action (up/down/left/right)
- `Q` = long-term expected value

The neural network acts as a function approximator: 

state → neural network → Q-values for each action

Training minimizes the Bellman error: 

Loss = (Q_predicted − Q_target)²

Where:

Q_target = reward + γ * max(Q_next_state)

This is pure applied math:
- Linear algebra (matrix multiplications)
- Calculus (gradients via backpropagation)
- Optimization (Adam)
- Probability (epsilon-greedy exploration)

---

# 🔁 Reinforcement Learning Loop

Each episode:

1. Reset game
2. Observe state
3. Choose action (epsilon-greedy)
4. Execute action
5. Receive reward
6. Store transition in replay buffer
7. Sample mini-batch
8. Update neural network

This cycle runs thousands of times.

The agent improves by gradually reducing prediction error on Q-values.


---

# 📦 Libraries Used

## 🐍 Python
Core language used for the entire system.

---

## 🔢 NumPy

Used for:
- Efficient grid representation
- Numerical operations
- Array manipulation
- Random sampling

NumPy provides fast vectorized operations and underpins much of the mathematical structure.

---

## 🔥 PyTorch

Used for:
- Neural network definition
- Automatic differentiation
- Backpropagation
- Optimizer (Adam)
- Tensor operations
- GPU acceleration (if enabled)

PyTorch handles the heavy linear algebra and gradient tracking required for DQN training.

---

## 🐢 Turtle (Initial Version)

The original implementation used Python’s built-in Turtle module for rendering and manual gameplay.

This version remains an important milestone — it created the environment the agent now learns from.

---

# 📊 Results So Far

After 5000 training episodes:

- Agent consistently scores 1
- Occasionally scores 2
- Rarely reaches 3
- Performance plateau observed

This plateau suggests:

- Reward shaping is functioning
- Exploration schedule is working
- Architecture capacity is currently the limiting factor

The current MLP model struggles with spatial reasoning.

Next upgrade: Convolutional Neural Network (CNN).

---

# 📚 What I Learned

## 1️⃣ Reinforcement Learning Is Inherently Unstable

Small changes in:
- Reward scaling
- Epsilon decay
- Learning rate

Can drastically affect outcomes.

Stability requires experimentation and tuning.

---

## 2️⃣ Most of "AI" Is Optimization

What feels like intelligence is simply:

- Minimizing loss
- Updating weights
- Iteratively approximating a function

It’s gradient descent layered over probability theory.

---

## 3️⃣ Reward Design Defines Behavior

The agent only learns what the reward function tells it to value.

Without shaped rewards, it learns almost nothing.

With distance-based shaping, learning improves dramatically.

Reward design determines the agent’s incentives.

---

## 4️⃣ Architecture Matters

Flattening a grid into a vector removes spatial structure.

An MLP cannot easily detect:
- Corridors
- Body adjacency
- Trap formations

Spatial problems require spatial models (CNNs).

---

## 5️⃣ Debugging Reinforcement Learning Is Different

You don’t debug correctness — you debug behavior trends.

Metrics like:
- Average reward
- Score distribution
- Plateau detection

Matter more than individual episode outcomes.

---

# 🧪 Training Performance

- ~5000 episodes in 10–15 minutes (CPU)
- Efficient enough for iterative experimentation

---

# 🔜 Next Steps

- Replace MLP with CNN for improved spatial reasoning
- Introduce target network
- Tune reward scaling
- Expand training visualizations
- Experiment with larger grids

---

# 🏷 Version History

- **v0.1** — Turtle-based manual Snake
- **v0.2** — Basic DQN with MLP
- **v0.3** — Modular RL architecture and cleaned repository structure

---

# 🏁 Why This Project Matters

This project demonstrates:

- Converting a deterministic game into a learning environment
- Implementing DQN from scratch
- Understanding reinforcement learning mechanics
- Applying linear algebra and calculus to real systems
- Iterative architectural refinement

It’s not about building a perfect Snake AI.

It’s about understanding how learning systems emerge from mathematics.
