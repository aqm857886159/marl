"""
临时脚本：跑 60k 步 IPPO 以检查评估指标是否写入日志。
"""

import pandas as pd

import MRRL
from run_ippo_seeds import append_logs


def main():
    seed = 99
    max_steps = 150_000
    eval_interval = 50_000

    print(f"[CHECK] Running IPPO eval test (seed={seed}, max_steps={max_steps})")
    log = MRRL.run_experiment(
        "IPPO",
        seed=seed,
        num_iterations=None,
        max_timesteps=max_steps,
        eval_interval_steps=eval_interval,
    )
    append_logs(log)

    df = pd.DataFrame(log)
    if "avg_latency_ms" not in df.columns:
        print("[CHECK] Eval returned no custom metrics (column missing).")
    else:
        eval_rows = df[df["avg_latency_ms"].notna()]
        if eval_rows.empty:
            print("[CHECK] Eval metrics column exists but contains only NaN.")
        else:
            print("[CHECK] Eval metrics:")
            print(eval_rows[["timestep", "avg_latency_ms", "p99_latency_ms", "avg_energy_J", "throughput_tps", "load_balance_jain"]])


if __name__ == "__main__":
    main()


