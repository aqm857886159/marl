"""
批量运行 IPPO 五个种子，默认训练满 5,000,000 步，并将日志追加到 ray_raw_results_log.csv。
"""
from pathlib import Path

import pandas as pd
import ray
import MRRL

LOG_PATH = Path("ray_raw_results_log.csv")


def append_logs(log_entries):
    """将 run_experiment 返回的日志附加到 CSV。"""
    if not log_entries:
        return

    df_new = pd.DataFrame(log_entries)

    if LOG_PATH.exists():
        existing_cols = pd.read_csv(LOG_PATH, nrows=0).columns.tolist()
        # 确保列顺序与已有文件一致，缺失列填 NA
        for col in existing_cols:
            if col not in df_new:
                df_new[col] = pd.NA
        df_new = df_new[existing_cols]
        df_new.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(LOG_PATH, index=False)

    print(f"[SAVE] 追加 {len(df_new)} 条日志 -> {LOG_PATH.resolve()}")


def main():
    MRRL.USE_MOCK_DATA = False

    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=1)

    try:
        for seed in [0, 1, 2, 3, 4]:
            print(f"\n=== IPPO 训练：种子 {seed} ===")
            log = MRRL.run_experiment("IPPO", seed, num_iterations=None)
            append_logs(log)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()


