"""Evaluate every trained checkpoint in models/ and print a leaderboard.

Reuses the deterministic evaluation logic from rl.evaluate so that seeds,
epsilon, eval mode, and the greedy action-selection policy are identical.

Usage:
    python -m rl.evaluate_all --episodes 50
"""

import argparse
import glob
import os
import re

from rl.evaluate import evaluate


def _experiment_name(filename: str) -> str:
    """Derive a short experiment label from a checkpoint filename.

    'exp_7_best.pth' -> 'exp_7'
    'frame_stack_breakthrough.pth' -> 'frame_stack_breakthrough'
    """
    stem = os.path.splitext(filename)[0]
    name = re.sub(r"_best$", "", stem)
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained checkpoints and print a leaderboard",
    )
    parser.add_argument(
        "--models-dir", type=str, default="models",
        help="Directory containing model checkpoints (default: models)",
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of evaluation episodes per model (default: 50)",
    )
    parser.add_argument(
        "--stack-size", type=int, default=4,
        help="Frame-stack size matching the trained models (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.models_dir, "*_best.pth")
    checkpoints = sorted(glob.glob(pattern))

    if not checkpoints:
        print(f"No checkpoints matching '*_best.pth' found in {args.models_dir}/")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) in {args.models_dir}/\n")

    results: list[dict] = []

    for ckpt in checkpoints:
        name = _experiment_name(os.path.basename(ckpt))
        print(f"=== Evaluating {name} ({ckpt}) ===")

        scores = evaluate(
            model_path=ckpt,
            num_episodes=args.episodes,
            stack_size=args.stack_size,
            seed=args.seed,
            verbose=False,
        )

        avg = sum(scores) / len(scores)
        results.append({
            "name": name,
            "avg_score": avg,
            "max_score": max(scores),
            "min_score": min(scores),
        })
        print(f"    avg={avg:.1f}  max={max(scores)}  min={min(scores)}\n")

    results.sort(key=lambda r: r["avg_score"], reverse=True)

    name_width = max(len(r["name"]) for r in results)
    header = "Experiment Leaderboard"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<{name_width}}   "
            f"avg_score={r['avg_score']:<6.1f}   "
            f"max={r['max_score']:<4}   "
            f"min={r['min_score']}"
        )


if __name__ == "__main__":
    main()
