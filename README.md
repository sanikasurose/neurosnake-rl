# neurosnake-rl 

**From the classic Snake game → Deep Reinforcement Learning → Reproducible RL Research Pipeline**

This project started as a simple Snake game and evolved into a **full reinforcement learning experimentation framework**.

The agent is trained with **Deep Q-Learning (DQN)** using a convolutional neural network and a structured experiment pipeline that enables **reproducible training, experiment tracking, and model evaluation**.

The repository now functions as a **mini RL research lab**.

---

# Current Version - v1.0.0

v1.0.0 marks the transition from **"training a model** to a **complete reinforcement learning experimentation pipeline**.

The system now supports: 

- reproducible experiments
- experiment tracking
- structured analysis
- model checkpointing
- deterministic evaluation
- experiment leaderboards
- CLI-controlled training runs

This makes it possible to **run, compare, and analyze RL experiments systematically**.

---

# Reinforcement Learning System

The agent uses a **Deep Q-Network (DQN)** architecture designed for grid-based spatial reasoning.

Key features:

- Convolutional Neural Network (CNN)
- Frame stacking for temporal awareness
- Double DQN to reduce Q-value overestimation
- Target network stabilization
- Replay buffer training
- Epsilon-greedy exploration
- GPU acceleration via PyTorch

Training optimizes the **Bellman error**:

`Loss = (Q_predicted − Q_target)²`

With Double DQN targets: 

`Loss = (Q_predicted − Q_target)²`

---

# RL Experiment Pipeline

The project now implements a **complete RL research workflow**.


Train Agent  
↓  
Log Experiment (SQLite)  
↓  
Save Best Model Checkpoint  
↓  
Evaluate Models  
↓   
Rank Experiments (Leaderboard)  
↓  
Analyze Results  

This enables **systematic RL experimentation instead of ad-hoc training runs**.

---

# Experiment Tracking & Analytics

Experiments are stored in a **SQLite database**.

Each run logs:

- model version
- learning rate
- gamma
- epsilon schedule
- batch size
- seed
- episode metrics

Episode-level metrics include:

- score
- reward
- loss
- epsilon
- Q-value statistics

The analytics layer enables:

- experiment comparison
- convergence detection
- performance profiling
- data integrity validation

---

# Evaluation System

The repo includes a **deterministic evaluation framework**.

Features:

- evaluate individual checkpoints
- batch evaluation of all trained models
- automatic experiment leaderboard

This allows **objective comparison between experiments**.

---

# Tech Stack

**Core ML**

- Python
- PyTorch
- NumPy

**Experiment Infrastructure**

- SQLite
- Structured experiment logging
- Training analytics
- CLI experiment control

**Visualization**

- Matplotlib

**Original Game**

- Python Turtle (v0.1)

---

# Why I Built This

Most reinforcement learning tutorials stop at **“the model trains.”**

Real ML systems require much more:

- reproducibility
- experiment tracking
- evaluation pipelines
- structured analysis

This project explores **what it takes to turn an RL agent into a research workflow**.

It bridges three areas:

- reinforcement learning
- machine learning engineering
- analytics engineering

---

# Learnings

**Reinforcement Learning**
- DQN and Double DQN
- value stabilization
- replay buffer training
- exploration strategies

**Machine Learning Engineering**
- experiment reproducibility
- checkpoint management
- deterministic evaluation
- structured training pipelines

**Data & Analytics**
- relational schema design
- experiment logging
- SQL analytics
- dataset validation

---

# Version History

- **v0.1** — Snake game (Turtle)
- **v0.2** — Basic MLP DQN
- **v0.3** — CNN + Frame Stacking
- **v0.4** — Double DQN stabilization
- **v0.5** — Experiment database + SQL analytics
- **v1.0** — Full RL experimentation pipeline

---

# Project Goal

The goal of this repository is not just to build a working RL agent, but to demonstrate **how to structure reinforcement learning experiments like a research project**.

The result is a small but complete **reinforcement learning experimentation platform** built from scratch.