import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_FILE = "training.db"

_MISSING_TABLES_MSG = (
    "Database tables not found. Run analytics/db.py to initialize schema."
)


def _connect():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    schema_path = Path(__file__).parent / "schema.sql"

    conn = _connect()

    with open(schema_path) as f:
        schema_sql = f.read()

    conn.executescript(schema_sql)
    conn.commit()
    conn.close()

    print("Database initialized successfully.")


def create_experiment(model_version, learning_rate, gamma, epsilon_start, epsilon_decay, batch_size, notes=None):
    conn = _connect()

    try:
        timestamp = datetime.now(timezone.utc).isoformat()

        cursor = conn.execute(
            "INSERT INTO experiments (model_version, architecture, start_time, notes) VALUES (?, ?, ?, ?)",
            (model_version, "DQN", timestamp, notes),
        )
        experiment_id = cursor.lastrowid

        conn.execute(
            "INSERT INTO hyperparameters (experiment_id, learning_rate, gamma, batch_size, epsilon_start, epsilon_decay) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (experiment_id, learning_rate, gamma, batch_size, epsilon_start, epsilon_decay),
        )

        conn.commit()
        return experiment_id
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            print(_MISSING_TABLES_MSG)
            sys.exit(1)
        raise
    finally:
        conn.close()


def log_episode(experiment_id, episode_number, score, total_reward, avg_loss, epsilon, mean_q, mean_target_q):
    conn = _connect()

    try:
        conn.execute(
            "INSERT INTO episodes "
            "(experiment_id, episode_number, score, total_reward, avg_loss, epsilon, mean_q, mean_target_q) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (experiment_id, episode_number, score, total_reward, avg_loss, epsilon, mean_q, mean_target_q),
        )
        conn.commit()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            print(_MISSING_TABLES_MSG)
            sys.exit(1)
        raise
    finally:
        conn.close()


def end_experiment(experiment_id):
    conn = _connect()

    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE experiments SET end_time = ? WHERE experiment_id = ?",
            (timestamp, experiment_id),
        )
        conn.commit()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            print(_MISSING_TABLES_MSG)
            sys.exit(1)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
