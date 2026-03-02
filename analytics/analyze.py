from analytics.db import _connect

SEPARATOR = "=" * 50


# ---------------------------------------------------------------------------
# SQL queries — kept here as named constants so the functions stay readable.
# ---------------------------------------------------------------------------

PROFILE_TOTAL_EXPERIMENTS = "SELECT COUNT(*) FROM experiments;"
PROFILE_TOTAL_EPISODES = "SELECT COUNT(*) FROM episodes;"
PROFILE_AVG_EPISODES = """
    SELECT AVG(cnt)
    FROM (
        SELECT COUNT(*) AS cnt
        FROM episodes
        GROUP BY experiment_id
    );
"""
PROFILE_SCORE_STATS = """
    SELECT MIN(score), MAX(score), AVG(score)
    FROM episodes;
"""

BEST_EXPERIMENT = """
    SELECT experiment_id, AVG(score) AS avg_score
    FROM episodes
    GROUP BY experiment_id
    ORDER BY avg_score DESC
    LIMIT 1;
"""

EXPERIMENT_SUMMARY = """
    SELECT
        ex.experiment_id,
        ex.model_version,
        COUNT(ep.episode_id)  AS total_episodes,
        AVG(ep.score)         AS avg_score,
        MAX(ep.score)         AS max_score,
        AVG(ep.avg_loss)      AS avg_loss
    FROM experiments ex
    JOIN episodes ep ON ex.experiment_id = ep.experiment_id
    GROUP BY ex.experiment_id
    ORDER BY avg_score DESC;
"""

AVG_SCORE_BY_LR = """
    SELECT h.learning_rate, AVG(ep.score) AS avg_score
    FROM hyperparameters h
    JOIN episodes ep ON h.experiment_id = ep.experiment_id
    GROUP BY h.learning_rate
    ORDER BY avg_score DESC;
"""

AVG_SCORE_BY_GAMMA = """
    SELECT h.gamma, AVG(ep.score) AS avg_score
    FROM hyperparameters h
    JOIN episodes ep ON h.experiment_id = ep.experiment_id
    GROUP BY h.gamma
    ORDER BY avg_score DESC;
"""

AVG_SCORE_BY_BATCH = """
    SELECT h.batch_size, AVG(ep.score) AS avg_score
    FROM hyperparameters h
    JOIN episodes ep ON h.experiment_id = ep.experiment_id
    GROUP BY h.batch_size
    ORDER BY avg_score DESC;
"""

UNDERPERFORMERS = """
    SELECT experiment_id, AVG(score) AS avg_score
    FROM episodes
    GROUP BY experiment_id
    HAVING AVG(score) < ?;
"""

CONVERGENCE = """
    SELECT experiment_id, MIN(episode_number) AS convergence_episode
    FROM (
        SELECT
            experiment_id,
            episode_number,
            AVG(score) OVER (
                PARTITION BY experiment_id
                ORDER BY episode_number
                ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
            ) AS rolling_avg
        FROM episodes
    )
    WHERE rolling_avg > ?
    GROUP BY experiment_id;
"""

# Integrity checks
CHECK_NULL_SCORES = """
    SELECT COUNT(*) FROM episodes WHERE score IS NULL;
"""
CHECK_ORPHAN_EPISODES = """
    SELECT COUNT(*)
    FROM episodes ep
    LEFT JOIN experiments ex ON ep.experiment_id = ex.experiment_id
    WHERE ex.experiment_id IS NULL;
"""
CHECK_DUPLICATE_EPISODES = """
    SELECT experiment_id, episode_number, COUNT(*) AS cnt
    FROM episodes
    GROUP BY experiment_id, episode_number
    HAVING cnt > 1;
"""
CHECK_EMPTY_EXPERIMENTS = """
    SELECT ex.experiment_id
    FROM experiments ex
    LEFT JOIN episodes ep ON ex.experiment_id = ep.experiment_id
    WHERE ep.episode_id IS NULL;
"""


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def profile_dataset(conn):
    print(SEPARATOR)
    print("DATASET PROFILE")
    print(SEPARATOR)

    total_experiments = conn.execute(PROFILE_TOTAL_EXPERIMENTS).fetchone()[0]
    total_episodes = conn.execute(PROFILE_TOTAL_EPISODES).fetchone()[0]
    avg_episodes = conn.execute(PROFILE_AVG_EPISODES).fetchone()[0]

    print(f"  Total Experiments:             {total_experiments}")
    print(f"  Total Episodes:                {total_episodes}")
    print(f"  Average Episodes per Experiment: {avg_episodes or 0:.1f}")

    if total_episodes > 0:
        min_s, max_s, avg_s = conn.execute(PROFILE_SCORE_STATS).fetchone()
        print(f"  Score Range:                   {min_s} – {max_s}")
        print(f"  Overall Average Score:         {avg_s:.2f}")
    else:
        print("  No episode data available.")

    print()


def summarize_performance(conn):
    print(SEPARATOR)
    print("PERFORMANCE SUMMARY")
    print(SEPARATOR)

    rows = conn.execute(EXPERIMENT_SUMMARY).fetchall()
    if not rows:
        print("  No experiment data to summarize.")
        print()
        return

    for exp_id, version, n_eps, avg_sc, max_sc, avg_l in rows:
        print(f"  Experiment {exp_id} ({version})")
        print(f"    Episodes:   {n_eps}")
        print(f"    Avg Score:  {avg_sc:.2f}")
        print(f"    Max Score:  {max_sc}")
        print(f"    Avg Loss:   {avg_l:.4f}")
        print()

    best = conn.execute(BEST_EXPERIMENT).fetchone()
    if best:
        print(f"  Best Experiment: {best[0]}  (avg score {best[1]:.2f})")

    print()


def analyze_hyperparameters(conn):
    print(SEPARATOR)
    print("HYPERPARAMETER ANALYSIS")
    print(SEPARATOR)

    lr_rows = conn.execute(AVG_SCORE_BY_LR).fetchall()
    if lr_rows:
        print("  Avg Score by Learning Rate:")
        for lr, avg in lr_rows:
            print(f"    lr={lr:<10}  avg_score={avg:.2f}")
    else:
        print("  No learning rate data.")

    print()

    gamma_rows = conn.execute(AVG_SCORE_BY_GAMMA).fetchall()
    if gamma_rows:
        print("  Avg Score by Gamma:")
        for gamma, avg in gamma_rows:
            print(f"    gamma={gamma:<8}  avg_score={avg:.2f}")
    else:
        print("  No gamma data.")

    print()

    batch_rows = conn.execute(AVG_SCORE_BY_BATCH).fetchall()
    if batch_rows:
        print("  Avg Score by Batch Size:")
        for bs, avg in batch_rows:
            print(f"    batch_size={bs:<6}  avg_score={avg:.2f}")
    else:
        print("  No batch size data.")

    print()


def detect_underperformers(conn, threshold=2):
    print(SEPARATOR)
    print(f"UNDERPERFORMING EXPERIMENTS  (threshold < {threshold})")
    print(SEPARATOR)

    rows = conn.execute(UNDERPERFORMERS, (threshold,)).fetchall()
    if rows:
        for exp_id, avg in rows:
            print(f"  Experiment {exp_id}:  avg_score={avg:.2f}")
    else:
        print("  All experiments meet the minimum threshold.")

    print()


def detect_convergence(conn, threshold=10):
    print(SEPARATOR)
    print(f"CONVERGENCE DETECTION  (rolling avg > {threshold})")
    print(SEPARATOR)

    rows = conn.execute(CONVERGENCE, (threshold,)).fetchall()
    if rows:
        for exp_id, ep_num in rows:
            print(f"  Experiment {exp_id}:  converged at episode {ep_num}")
    else:
        print("  No experiments have reached convergence yet.")

    print()


def run_data_integrity_checks(conn):
    print(SEPARATOR)
    print("DATA INTEGRITY CHECKS")
    print(SEPARATOR)

    issues_found = False

    null_scores = conn.execute(CHECK_NULL_SCORES).fetchone()[0]
    if null_scores > 0:
        print(f"  \u26a0 WARNING: {null_scores} episodes have NULL scores.")
        issues_found = True

    orphans = conn.execute(CHECK_ORPHAN_EPISODES).fetchone()[0]
    if orphans > 0:
        print(f"  \u26a0 WARNING: {orphans} episodes reference non-existent experiments.")
        issues_found = True

    dupes = conn.execute(CHECK_DUPLICATE_EPISODES).fetchall()
    if dupes:
        print(f"  \u26a0 WARNING: {len(dupes)} duplicate episode numbers detected:")
        for exp_id, ep_num, cnt in dupes:
            print(f"    experiment {exp_id}, episode {ep_num} appears {cnt} times")
        issues_found = True

    empty = conn.execute(CHECK_EMPTY_EXPERIMENTS).fetchall()
    if empty:
        ids = [str(row[0]) for row in empty]
        print(f"  \u26a0 WARNING: {len(empty)} experiments have no episodes: [{', '.join(ids)}]")
        issues_found = True

    if not issues_found:
        print("  \u2713 No integrity issues detected.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    conn = _connect()

    try:
        profile_dataset(conn)
        summarize_performance(conn)
        analyze_hyperparameters(conn)
        detect_underperformers(conn, threshold=2)
        detect_convergence(conn, threshold=10)
        run_data_integrity_checks(conn)
    finally:
        conn.close()

    print(SEPARATOR)
    print("Analysis complete.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
