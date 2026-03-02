import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_FILE = "training.db"

def init_db():
    schema_path = Path(__file__).parent / "schema.sql"

    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")

    with open(schema_path) as f:
        schema_sql = f.read()

    conn.executescript(schema_sql)
    conn.commit()
    conn.close()

    print("Database initialized successfully.")

def create_experiment(model_version, learning_rate, gamma, epsilon_start, epsilon_decay, batch_size):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")

    timestamp = datetime.now(timezone.utc).isoformat()

    cursor = conn.execute(
        "INSERT INTO experiments (model_version, architecture, start_time) VALUES (?, ?, ?)",
        (model_version, "DQN", timestamp),
    )
    experiment_id = cursor.lastrowid

    conn.execute(
        "INSERT INTO hyperparameters (experiment_id, learning_rate, gamma, batch_size, epsilon_start, epsilon_decay) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (experiment_id, learning_rate, gamma, batch_size, epsilon_start, epsilon_decay),
    )

    conn.commit()
    conn.close()

    return experiment_id


def log_episode(experiment_id, episode_number, score, total_reward, avg_loss, epsilon, mean_q, mean_target_q):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")

    conn.execute(
        "INSERT INTO episodes "
        "(experiment_id, episode_number, score, total_reward, avg_loss, epsilon, mean_q, mean_target_q) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (experiment_id, episode_number, score, total_reward, avg_loss, epsilon, mean_q, mean_target_q),
    )

    conn.commit()
    conn.close()


def end_experiment(experiment_id):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")

    timestamp = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE experiments SET end_time = ? WHERE experiment_id = ?",
        (timestamp, experiment_id),
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
