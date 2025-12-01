r"""
快速测试 Ray 端增量保存逻辑的脚本。

用途：
- 不修改正式实验配置（5M 步、5 个种子）
- 通过“猴子补丁”临时缩小每次 run_experiment 的步数和迭代次数
- 调用 MRRL.main()，触发我们在 MRRL.PY 中实现的增量保存逻辑

使用方法：
    (marl) C:\Users\23732\Desktop\MARL边缘调度实验> python quick_test_ray_incremental.py

预期现象：
- 终端会打印 MAPPO/IPPO 少量迭代的训练日志
- 每完成一个 (算法, 种子) 会看到 [AUTO-SAVE] 打印
- 根目录下生成 ray_raw_results_log.csv，可用 Excel/记事本查看
"""

import os

import pandas as pd  # 仅用于在最后打印一下结果条数，方便你肉眼确认

import MRRL


def patched_run_experiment(alg_name, seed, num_iterations=1000, max_timesteps=5_000_000, eval_interval_steps=50_000):
    """
    对 MRRL.run_experiment 的轻量包装：
    - 将 num_iterations 和 max_timesteps 收紧到一个很小的规模，用于快速测试
    - 其他逻辑保持不变
    """
    print(f"[DEBUG] run_experiment patched: alg={alg_name}, seed={seed}, "
          f"num_iterations=5, max_timesteps=100_000")
    return MRRL.run_experiment_original(
        alg_name,
        seed,
        num_iterations=5,       # 小规模迭代
        max_timesteps=100_000,  # 最多 10 万步，很快结束
        eval_interval_steps=20_000,
    )


def main():
    # 1. 备份原始 run_experiment
    if not hasattr(MRRL, "run_experiment_original"):
        MRRL.run_experiment_original = MRRL.run_experiment

    # 2. 打补丁：让 main 内部使用我们的小规模版本
    MRRL.run_experiment = patched_run_experiment

    # 3. 确保使用真实实验分支
    MRRL.USE_MOCK_DATA = False

    print("=== QUICK TEST: Ray 增量保存逻辑 ===")
    print("本测试会用很小的步数跑 MAPPO/IPPO，触发 AUTO-SAVE 逻辑。")
    print("运行结束后，请检查当前目录下的 ray_raw_results_log.csv。\n")

    try:
        MRRL.main()
    finally:
        # 4. 恢复原始函数，避免影响后续正式实验
        MRRL.run_experiment = MRRL.run_experiment_original

    # 5. 简单打印一下 CSV 里有多少行，方便你确认
    csv_path = "ray_raw_results_log.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\n[CHECK] {csv_path} 中共有 {len(df)} 条记录。")
        print("前几行示例：")
        print(df.head())
    else:
        print(f"\n[WARN] 未找到 {csv_path}，请确认 AUTO-SAVE 是否正常打印。")


if __name__ == "__main__":
    main()


