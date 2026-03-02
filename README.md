# 🐍 neurosnake-rl - From Snake Game to Deep RL + Analytics Platform

This project began as a simple implementation of the classic Snake game using Python’s Turtle graphics. It has since evolved into a fully autonomous Deep Reinforcement Learning system, and now into a structured experiment tracking and analytics platform.

What started as a simple game is now:

- A Deep Q-Network (DQN) implemented from scratch in PyTorch
- A convolutional architecture for spatial reasoning over grid states
- Frame stacking for temporal awareness and directional inference
- A Double DQN implementation to reduce Q-value overestimation
- Target network stabilization and replay buffer training
- A reproducible experiment tracking and analytics framework

This repository documents that progression; from deterministic game logic to stabilized deep value learning with analytical validation.

---

# 🚀 Current Release: v0.5.0 — RL + Experiment Tracking + Analytics

Version 0.5.0 introduces structured experiment logging and analytical validation on top of the RL system.

This transforms the project from “just training a model” into a reproducible, analyzable ML experiment framework.

---

# 🧠 Reinforcement Learning System (v0.4 Core)

## Major Architectural Features

- ✅ Convolutional Neural Network (CNN) for spatial reasoning  
- ✅ Frame stacking for temporal awareness  
- ✅ Double DQN (reduced Q-value overestimation)  
- ✅ Target network stabilization  
- ✅ Replay buffer (50,000 capacity)  
- ✅ GPU-accelerated training (PyTorch + Apple MPS)  
- ✅ Rolling-average performance tracking  
- ✅ Structured training logs  

---

## 📈 Performance Evolution

### v0.2.0 — MLP DQN
- Plateaued at score **2–3**
- No spatial modeling
- Limited policy improvement

### v0.3.0 — CNN + Frame Stacking
- Broke architectural ceiling
- Strong performance spikes
- Improved survival behavior

### v0.4.0 — CNN + Frame Stacking + Double DQN
- 4000 training episodes
- Stable rolling average: **~2.4–2.7**
- Best score: **11**
- No Q-value divergence
- Stable late-stage loss (~0.25–0.32)

This version emphasizes **correct learning dynamics over inflated Q-values**.

Q-values remain bounded and consistent with target estimates throughout training.

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

Training minimizes the Bellman error:

Loss = (Q_predicted − Q_target)²

With Double DQN target computation:

Q_target = r + γ * Q_target(s', argmax_a Q_online(s', a))

---

# 🗄 v0.5 — Structured Experiment Tracking

Training runs are now stored in a relational SQLite database:

`training.db`

## Database Design

### experiments
- experiment_id
- model_version
- created_at
- notes

### hyperparameters
- experiment_id (FK)
- learning_rate
- gamma
- batch_size
- epsilon_decay
- etc.

### episodes
- episode_id
- experiment_id (FK)
- episode_number
- score
- avg_loss

This enables:

- Reproducibility
- Cross-experiment comparison
- Hyperparameter impact analysis
- Convergence detection
- Data validation

---

# 📊 SQL Analytics Layer

The project now includes structured analytical queries:

`analytics/queries.sql`

Implemented concepts:

- SELECT
- JOIN
- GROUP BY
- HAVING
- Aggregations
- Window functions (rolling averages)

Example analyses:

- Average score by learning rate
- Best performing experiment
- Underperforming run detection
- Rolling 10-episode moving averages
- Convergence episode detection

---

# 🧪 analyze.py — Data Profiling & Validation Layer

`analytics/analyze.py` executes structured analytics directly against the database.

It performs:

## Dataset Profiling
- Total experiments
- Total episodes
- Score distributions
- Average episodes per experiment

## Performance Summary
- KPI reporting per experiment
- Best-performing run detection
- Hyperparameter comparisons

## Convergence Detection
- Window-function-based rolling average tracking
- Threshold-based convergence identification

## Data Integrity Checks
- NULL detection
- Orphaned records
- Duplicate episode numbers
- Experiments with zero episodes

This simulates a lightweight analytics engineering workflow:

- SQL validation
- KPI reporting
- Data quality checks
- Structured reporting

The project now resembles a mini data platform layered on top of an ML system.

---

# 📦 Libraries Used

## Python
Core language.

## NumPy
Grid representation and numerical operations.

## PyTorch
- CNN architecture
- Automatic differentiation
- Adam optimizer
- MPS GPU backend

## SQLite
- Structured experiment tracking
- Relational schema design
- Analytical querying
- Data validation

## Turtle (v0.1)
Original rendering engine.

---

# 🏷 Version History

- **v0.1** — Manual Snake (Turtle)
- **v0.2** — Basic MLP DQN
- **v0.3** — CNN + Frame Stacking
- **v0.4** — CNN + Frame Stacking + Double DQN
- **v0.5** — Experiment Database + SQL Analytics + Data Validation

---

# 🏁 Why This Project Matters

This project demonstrates:

- Implementing DQN and Double DQN from scratch
- Understanding architectural stabilization
- Designing a relational schema for ML experiments
- Writing analytical SQL with window functions
- Performing data profiling and validation
- Structuring experiments for reproducibility

It evolved from a simple game to:

- Applied reinforcement learning
- Experiment tracking infrastructure
- Analytical SQL practice
- Data validation engineering

It bridges machine learning and data engineering by turning a learning system into a measurable, analyzable platform.

