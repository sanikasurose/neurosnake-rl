-- analytics/schema.sql

-- Create tables for experiments
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    architecture TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    notes TEXT
);

-- Create tables for hyperparameters
CREATE TABLE IF NOT EXISTS hyperparameters (
    experiment_id INTEGER PRIMARY KEY,
    learning_rate REAL NOT NULL,
    gamma REAL NOT NULL,
    batch_size INTEGER NOT NULL,
    epsilon_start REAL NOT NULL,
    epsilon_decay REAL NOT NULL,
    FOREIGN KEY (experiment_id)
        REFERENCES experiments (experiment_id)
        ON DELETE CASCADE
);

-- Create tables for episodes
CREATE TABLE IF NOT EXISTS episodes (
    episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    episode_number INTEGER NOT NULL,
    score INTEGER,
    total_reward REAL,
    avg_loss REAL,
    epsilon REAL,
    mean_q REAL,
    mean_target_q REAL,
    FOREIGN KEY (experiment_id)
        REFERENCES experiments (experiment_id)
        ON DELETE CASCADE
);