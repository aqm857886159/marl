import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


METRICS = [
    "avg_latency_ms",
    "p99_latency_ms",
    "avg_energy_J",
    "throughput_tps",
    "load_balance_jain",
    "deadline_violation_rate",
]


def load_json(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    if isinstance(data, list):
        return data
    return []


def find_convergence_step(entries: List[Dict]) -> float:
    if not entries:
        return 0.0
    rewards = [entry.get("reward", 0.0) for entry in entries]
    steps = [entry.get("step", 0) for entry in entries]
    best = max(rewards)
    if best == 0.0:
        threshold = 0.0
    else:
        threshold = best - 0.1 * abs(best)
    for step, reward in zip(steps, rewards):
        if reward >= threshold:
            return float(step)
    return float(steps[-1])


def export_metrics(algorithm: str, results_dir: Path, output_dir: Path):
    learning_rows = []
    perf_rows = []
    convergence_rows = []

    for seed_dir in sorted(results_dir.glob("seed_*")):
        if "_" not in seed_dir.name:
            continue
        try:
            seed = int(seed_dir.name.split("_")[1])
        except ValueError:
            continue

        training_entries = load_json(seed_dir / "training_log.json")
        eval_entries = load_json(seed_dir / "eval_log.json")

        if training_entries:
            for entry in training_entries:
                step = entry.get("step", 0)
                reward = entry.get("reward", 0.0)
                learning_rows.append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "timestep": step,
                        "episode_return_mean": reward,
                    }
                )
                for metric in METRICS:
                    if metric in entry:
                        perf_rows.append(
                            {
                                "algorithm": algorithm,
                                "seed": seed,
                                "metric": metric,
                                "value": entry[metric],
                            }
                        )

            convergence_rows.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "steps": find_convergence_step(training_entries),
                }
            )

        for entry in eval_entries:
            step = entry.get("step")
            for metric in METRICS:
                if metric in entry:
                    perf_rows.append(
                        {
                            "algorithm": algorithm,
                            "seed": seed,
                            "metric": metric,
                            "value": entry[metric],
                        }
                    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if learning_rows:
        learning_path = output_dir / f"learning_curves_{algorithm}.csv"
        with learning_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["algorithm", "seed", "timestep", "episode_return_mean"]
            )
            writer.writeheader()
            writer.writerows(learning_rows)

    if perf_rows:
        perf_path = output_dir / f"performance_metrics_{algorithm}.csv"
        with perf_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["algorithm", "seed", "metric", "value"])
            writer.writeheader()
            writer.writerows(perf_rows)

    if convergence_rows:
        conv_path = output_dir / f"convergence_speed_{algorithm}.csv"
        with conv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["algorithm", "seed", "steps"])
            writer.writeheader()
            writer.writerows(convergence_rows)

    print(f"[OK] Exported {algorithm} metrics to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export PyMARL EdgeSimGym metrics to CSV")
    parser.add_argument(
        "--algorithm",
        choices=["MADDPG", "QMIX"],
        required=True,
        help="Algorithm name to export.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing seed_* subfolders (default: results/edge_<algo>).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store exported CSV files (default: results/edge_<algo>/exports).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    algo = args.algorithm.upper()
    default_results = Path(f"results/edge_{algo.lower()}") if args.results_dir is None else Path(args.results_dir)
    default_output = default_results / "exports" if args.output_dir is None else Path(args.output_dir)

    if not default_results.exists():
        raise FileNotFoundError(f"Results directory not found: {default_results}")

    export_metrics(algo, default_results, default_output)


if __name__ == "__main__":
    main()

