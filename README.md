# 🐍 neurosnake-rl — From Turtle Game to Deep Q-Learning

This project began as a simple implementation of the classic Snake game using Python’s Turtle graphics. It has since evolved into a full deep reinforcement learning system where a Deep Q-Network (DQN) agent learns to play the game autonomously.

This repository documents that progression — from deterministic game logic to convolutional neural networks with temporal awareness.

---

# 🚀 Current Release: v0.3.0 — CNN + Frame Stacking

Version 0.3.0 introduces a major architectural upgrade:

- ✅ Convolutional Neural Network (CNN) for spatial reasoning  
- ✅ Frame stacking for temporal awareness  
- ✅ GPU-accelerated training (PyTorch + MPS)  
- ✅ Rolling-average performance tracking  
- ✅ Structured experiment logging  

---

## 📈 Performance Improvements

### v0.2.0 — MLP-based DQN
- Plateaued at score **2–3**
- Limited spatial understanding
- No temporal context

### v0.3.0 — CNN + Frame Stacking
- Max score: **20**
- Rolling average: **8.5+**
- Consistent double-digit performance
- Dramatically improved survival strategy

This version breaks the architectural ceiling of the original MLP and demonstrates how representation learning fundamentally changes behavior in reinforcement learning systems.

---

# 🧠 How the Agent Works

At its core, this project is applied mathematics.

There is no “intelligence” in the traditional sense — only:

- Function approximation  
- Optimization  
- Linear algebra  
- Probability  
- Gradient descent  

The agent approximates the function:

Q(s, a) ≈ expected future reward

Where:
- `s` = state (the grid)
- `a` = action (up/down/left/right)
- `Q` = long-term expected value

Neural network pipeline: 

state → CNN → fully connected layers → Q-values

Training minimizes the Bellman error: 

Loss = (Q_predicted − Q_target)²

Where:

Q_target = reward + γ * max(Q_next_state)

This system relies on:
- Matrix multiplication (convolutions + linear layers)
- Automatic differentiation (backpropagation)
- Stochastic optimization (Adam)
- Exploration via epsilon-greedy sampling

---
# 🧬 Why CNN + Frame Stacking Matters

## 🧱 CNN — Spatial Awareness

Flattening a grid destroys structure.

A CNN preserves:
- Local adjacency
- Corridors
- Body formations
- Trap geometry

Spatial problems require spatial models.

---

## ⏳ Frame Stacking — Temporal Awareness

Instead of observing a single grid snapshot, the agent now sees multiple consecutive frames.

This enables:
- Direction inference (momentum awareness)
- Collision prediction
- Loop planning
- Reduced self-trapping

The jump from score ~2 to 20 directly resulted from adding temporal context.

---

# 🔁 Reinforcement Learning Loop

Each episode:

1. Reset game  
2. Observe stacked state  
3. Choose action (epsilon-greedy)  
4. Execute action  
5. Receive reward  
6. Store transition in replay buffer  
7. Sample mini-batch  
8. Update neural network  

This loop runs thousands of times.

Learning emerges from minimizing prediction error over many sampled experiences.

---

# 📊 Training Results (v0.3.0)

- 2000 training episodes
- GPU accelerated (Apple MPS)
- Rolling 100-episode average tracked
- Model checkpoint saved at performance milestone

Observed behavior:
- Early instability (expected exploration phase)
- Inflection point around episode ~900
- Steady upward performance trend
- Stable late-stage average of 7–9
- Peak score of 20

The agent now performs at or above average human play.

---

# 📦 Libraries Used

## 🐍 Python
Core language used for the entire system.

## 🔢 NumPy
- Grid representation
- Efficient numerical operations
- Sampling and array manipulation

## 🔥 PyTorch
- Neural network architecture
- Automatic differentiation
- Adam optimizer
- GPU acceleration (MPS backend)

## 🐢 Turtle (Initial Version)
Original rendering engine used in the first manual implementation.

---

# 🧪 Training Performance

- ~2000 episodes in a few minutes (GPU)
- Efficient enough for rapid experimentation
- Performance improves with longer training

---

# 🏷 Version History

- **v0.1** — Manual Snake using Turtle
- **v0.2** — Basic DQN with MLP (plateau at ~2)
- **v0.3** — CNN + Frame Stacking (max score 20, rolling avg 8.5+)

---

# 🔜 Next Steps

- Double DQN for stabilization
- Larger grid scaling (21x21 → 29x29)
- Improved reward shaping experiments
- Full training visualizer with neural network activation display

---

# 🏁 Why This Project Matters

This project demonstrates:

- Turning a deterministic game into a learning environment
- Implementing DQN from scratch
- Understanding how architecture impacts behavior
- Applying linear algebra and calculus to real systems
- Iterative performance scaling through structured experimentation

It’s not about building a perfect Snake AI.

It’s about understanding how learning systems emerge from mathematics.