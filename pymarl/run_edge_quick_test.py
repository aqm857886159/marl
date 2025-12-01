"""
EdgeSimGym quick test runner for PyMARL.

This script launches a short QMIX training session on the custom edge_marl
environment to verify that the environment parameters match the thesis setup.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_trial(seed: int, t_max: int):
    cmd = [
        sys.executable,
        "src/main.py",
        "--config=edge_qmix",
        "--env-config=edge_marl",
        "with",
        f"seed={seed}",
        f"t_max={t_max}",
        "test_interval=10000",
        "log_interval=1000",
        "runner_log_interval=1000",
        "learner_log_interval=1000",
        "batch_size=32",
        "use_cuda=False",
    ]
    print(f"\n>>> Running QMIX quick test (seed={seed}, t_max={t_max})")
    print("    " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="QMIX quick test on EdgeSimGym.")
    parser.add_argument(
        "--seeds", nargs="+", default=["0"], help="List of seeds (default: 0)"
    )
    parser.add_argument(
        "--t-max", type=int, default=5000, help="Total timesteps for each run."
    )
    args = parser.parse_args()

    for seed_str in args.seeds:
        run_trial(int(seed_str), args.t_max)


if __name__ == "__main__":
    main()

