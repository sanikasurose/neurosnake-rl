# 🐍 neurosnake-rl — From Turtle Game to Deep Reinforcement Learning

This project began as a simple implementation of the classic Snake game using Python’s Turtle graphics. It has since evolved into a fully autonomous deep reinforcement learning system where a Deep Q-Network (DQN) agent learns to play the game from raw grid observations.

This repository documents that progression — from deterministic game logic to convolutional neural networks with temporal awareness and stabilized value learning.

---

# 🚀 Current Release: v0.4.0 — CNN + Frame Stacking + Double DQN

Version 0.4.0 introduces architectural stabilization and full training convergence.

## 🔧 Major Upgrades

- ✅ Convolutional Neural Network (CNN) for spatial reasoning  
- ✅ Frame stacking for temporal awareness  
- ✅ **Double DQN** (reduced Q-value overestimation)  
- ✅ Target network stabilization  
- ✅ Replay buffer (50,000 capacity)  
- ✅ GPU-accelerated training (PyTorch + Apple MPS)  
- ✅ Rolling-average performance tracking  
- ✅ Structured training logs saved to CSV  
- ✅ Stable 4000-episode convergence run  

This version focuses on **numerical stability, value correction, and controlled convergence** rather than raw score spikes.

---

# 📈 Performance Evolution

## v0.2.0 — MLP DQN
- Plateaued at score **2–3**
- No spatial modeling
- Limited policy improvement

## v0.3.0 — CNN + Frame Stacking
- Major spatial upgrade
- Strong performance spikes
- Improved survival behavior
- Demonstrated architectural ceiling break

## 🆕 v0.4.0 — CNN + Frame Stacking + Double DQN
- 4000 training episodes
- Stable rolling average: **~2.4–2.7**
- Best score: **11**
- No Q-value divergence
- Stable loss (~0.25–0.32 late-stage)
- Fully converged training curve

This version emphasizes **correct learning dynamics over inflated Q-values**.

Unlike earlier versions, Q-values remain bounded and consistent with target estimates throughout training.

---

# 🧠 How the Agent Works

At its core, this project is applied mathematics.

There is no handcrafted strategy — only:

- Function approximation  
- Optimization  
- Linear algebra  
- Probability  
- Gradient descent  

The agent approximates:

Q(s, a) ≈ expected future reward

Where:
- `s` = stacked grid state (8-channel input)
- `a` = action (up, down, left, right)
- `Q` = long-term expected value

---

Neural network pipeline: 

state → CNN → fully connected layers → Q-values

Training minimizes the Bellman error: 

Loss = (Q_predicted − Q_target)²

With **Double DQN target computation**:

Q_target = r + γ * Q_target(s', argmax_a Q_online(s', a))

This reduces value overestimation and stabilizes learning.

---

# 🧬 Why CNN + Frame Stacking + Double DQN Matters

## 🧱 CNN — Spatial Awareness

Flattening a grid destroys structure.

A CNN preserves:
- Local adjacency  
- Body formations  
- Corridor geometry  
- Trap detection  

Spatial problems require spatial models.

---

## ⏳ Frame Stacking — Temporal Awareness

Multiple consecutive frames allow:

- Direction inference  
- Momentum awareness  
- Collision prediction  
- Reduced self-trapping  

Without temporal context, the agent cannot infer movement direction.

---

# 🔁 Reinforcement Learning Loop

Each episode:

1. Reset environment  
2. Observe stacked state  
3. Choose action (epsilon-greedy)  
4. Execute action  
5. Receive reward  
6. Store transition in replay buffer  
7. Sample mini-batch  
8. Update online network  
9. Periodically update target network  

Learning emerges from thousands of gradient steps across diverse replayed experiences.

---

# 📊 Training Results (v0.4.0)

- 4000 training episodes
- GPU-accelerated (Apple MPS)
- Replay buffer capacity: 50,000
- Batch size: 128
- Learning rate: 3e-4
- Stable convergence

### Observed Training Phases

1. Early exploration instability (episodes 0–800)
2. Q-value stabilization (~1000 episodes)
3. Policy improvement (1200–2500)
4. Controlled convergence plateau (~2.5 rolling avg)

The agent demonstrates:
- Consistent survival behavior
- Multi-food chaining
- Reduced early suicide
- Clear learned policy patterns

This version represents a **fully stabilized deep RL system**, not just a performance spike.

---

# 📦 Libraries Used

## 🐍 Python
Core language for the entire system.

## 🔢 NumPy
Grid representation and numerical operations.

## 🔥 PyTorch
- CNN architecture
- Automatic differentiation
- Adam optimizer
- MPS GPU backend

## 🐢 Turtle (v0.1)
Original rendering engine used for manual Snake implementation.

---

# 🏷 Version History

- **v0.1** — Manual Snake (Turtle)
- **v0.2** — Basic MLP DQN
- **v0.3** — CNN + Frame Stacking
- **v0.4** — CNN + Frame Stacking + Double DQN (Stable Convergence)

---

# 🏁 Why This Project Matters

This project demonstrates:

- Turning a deterministic game into a learning environment
- Implementing DQN and Double DQN from scratch
- Understanding how architecture impacts behavior
- Diagnosing instability in value learning
- Scaling experiments through structured iteration

This is not about building a perfect Snake AI.

It’s about understanding how learning systems emerge from mathematics — and how architectural decisions shape behavior.