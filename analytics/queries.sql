-- Query 1: Average episode score by learning rate
-- Joins hyperparameters with episodes to see how each learning rate
-- setting performs on average across all episodes that used it.
SELECT
    h.learning_rate,
    AVG(e.score) AS avg_score
FROM hyperparameters h
JOIN episodes e ON h.experiment_id = e.experiment_id
GROUP BY h.learning_rate;


-- Query 2: Best performing experiment
-- Ranks experiments by their mean episode score and returns only the
-- top performer. Useful for identifying which run to inspect further.
SELECT
    e.experiment_id,
    AVG(e.score) AS avg_score
FROM episodes e
GROUP BY e.experiment_id
ORDER BY avg_score DESC
LIMIT 1;


-- Query 3: Underperforming experiments
-- Flags experiments whose average score falls below a minimum threshold.
-- Helps quickly surface runs that failed to learn or diverged.
SELECT
    e.experiment_id,
    AVG(e.score) AS avg_score
FROM episodes e
GROUP BY e.experiment_id
HAVING AVG(e.score) < 2;


-- Query 4: Rolling average score per episode (window of 10)
-- Computes a 10-episode moving average within each experiment, ordered
-- by episode number. Smooths out noise so training trends are visible.
SELECT
    e.experiment_id,
    e.episode_number,
    e.score,
    AVG(e.score) OVER (
        PARTITION BY e.experiment_id
        ORDER BY e.episode_number
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_score
FROM episodes e
ORDER BY e.experiment_id, e.episode_number;


-- Query 5: Convergence episode detection
-- Finds the earliest episode in each experiment where the 10-episode
-- rolling average score first exceeds a convergence threshold of 10.
-- A NULL result for an experiment means it never converged.
SELECT
    experiment_id,
    MIN(episode_number) AS convergence_episode
FROM (
    SELECT
        e.experiment_id,
        e.episode_number,
        AVG(e.score) OVER (
            PARTITION BY e.experiment_id
            ORDER BY e.episode_number
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS rolling_avg_score
    FROM episodes e
) AS windowed
WHERE rolling_avg_score > 10
GROUP BY experiment_id;
