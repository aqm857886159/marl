import pandas as pd
import json
import os
import glob
from pathlib import Path

# 配置路径
RAY_CSV_PATH = "ray_raw_results_log.csv"
PYMARL_BASE_DIR = Path("pymarl/results")
OUTPUT_DIR = Path("experiment_outputs_final")

# 定义指标映射 (PyMARL JSON key -> Unified CSV column)
PYMARL_METRIC_MAP = {
    "avg_latency_ms": "avg_latency_ms",
    "p99_latency_ms": "p99_latency_ms",
    "avg_energy_J": "avg_energy_J",
    "deadline_violation_rate": "deadline_violation_rate",
    "throughput_tps": "throughput_tps",
    "load_balance_jain": "load_balance_jain",
}

def load_ray_data():
    """读取 Ray (IPPO/MAPPO) 的数据"""
    if not os.path.exists(RAY_CSV_PATH):
        print(f"[WARN] 未找到 Ray 数据文件: {RAY_CSV_PATH}")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(RAY_CSV_PATH)
    
    # 1. 提取学习曲线 (Reward)
    # 假设 Ray 数据中已有 episode_return_mean
    learning_df = df[['algorithm', 'seed', 'timestep', 'episode_return_mean']].copy()
    learning_df.rename(columns={'episode_return_mean': 'reward'}, inplace=True)
    
    # 2. 提取评估指标
    # Ray 数据是混合的，通常每一步都有指标，但我们只关心收敛后或者周期性的评估
    # 这里为了简单，把所有有数值的行都作为评估数据
    eval_cols = ['algorithm', 'seed', 'timestep'] + list(PYMARL_METRIC_MAP.values())
    # 过滤掉没有这些列的行 (防止旧数据干扰)
    available_cols = [c for c in eval_cols if c in df.columns]
    eval_df = df[available_cols].copy()
    
    print(f"[Ray] 加载了 {len(df)} 行数据 (IPPO/MAPPO)")
    return learning_df, eval_df

def load_pymarl_data(algo_name):
    """读取 PyMARL (MADDPG/QMIX) 的数据"""
    learning_records = []
    eval_records = []
    
    def to_scalar(v):
        if isinstance(v, dict) and "value" in v:
            try:
                return float(v["value"])
            except Exception:
                return None
        try:
            return float(v)
        except Exception:
            return None

    # 对应的目录名，例如 edge_maddpg 或 edge_qmix
    algo_dir = PYMARL_BASE_DIR / f"edge_{algo_name.lower()}"
    if not algo_dir.exists():
        print(f"[WARN] 未找到 PyMARL 算法目录: {algo_dir}")
        return pd.DataFrame(), pd.DataFrame()

    # 遍历 seed_0, seed_1, ...
    seed_dirs = glob.glob(str(algo_dir / "seed_*"))
    for s_dir in seed_dirs:
        try:
            seed = int(os.path.basename(s_dir).split("_")[1])
        except ValueError:
            continue
            
        # A. 读取 Training Log (Reward 曲线)
        train_file = Path(s_dir) / "training_log.json"
        if train_file.exists():
            with open(train_file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    r = to_scalar(entry.get("reward"))
                    if r is None:
                        continue
                    learning_records.append({
                        "algorithm": algo_name,
                        "seed": seed,
                        "timestep": entry.get("step", 0),
                        "reward": r
                    })

        # B. 读取 Eval Log (各项指标)
        eval_file = Path(s_dir) / "eval_log.json"
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    record = {
                        "algorithm": algo_name,
                        "seed": seed,
                        "timestep": entry.get("step", 0)
                    }
                    # 提取各个指标
                    for json_key, csv_col in PYMARL_METRIC_MAP.items():
                        if json_key in entry:
                            val = to_scalar(entry[json_key])
                            if val is not None:
                                record[csv_col] = val
                    eval_records.append(record)

    print(f"[PyMARL] 加载了 {len(learning_records)} 条训练记录, {len(eval_records)} 条评估记录 ({algo_name})")
    return pd.DataFrame(learning_records), pd.DataFrame(eval_records)

def main():
    print("="*40)
    print("   开始合并所有实验数据 (IPPO, MAPPO, MADDPG, QMIX)")
    print("="*40)

    # 1. 加载 Ray 数据
    ray_learning, ray_eval = load_ray_data()

    # 2. 加载 PyMARL 数据
    maddpg_learning, maddpg_eval = load_pymarl_data("MADDPG")
    qmix_learning, qmix_eval = load_pymarl_data("QMIX")

    # 3. 合并
    all_learning = pd.concat([ray_learning, maddpg_learning, qmix_learning], ignore_index=True)
    all_eval = pd.concat([ray_eval, maddpg_eval, qmix_eval], ignore_index=True)

    # 4. 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 4.1 保存学习曲线总表
    learning_path = OUTPUT_DIR / "final_learning_curves.csv"
    all_learning.to_csv(learning_path, index=False)
    print(f"\n[SUCCESS] 学习曲线已保存: {learning_path} (共 {len(all_learning)} 行)")

    # 4.2 保存评估指标总表
    eval_path = OUTPUT_DIR / "final_eval_metrics.csv"
    all_eval.to_csv(eval_path, index=False)
    print(f"[SUCCESS] 评估指标已保存: {eval_path} (共 {len(all_eval)} 行)")

    # 5. 生成一个简单的统计摘要 (Convergence Table)
    # 取最后 10% 的 timestep 的平均值作为最终性能
    print("\n--- 算法最终性能摘要 (Last 10% Steps Mean) ---")
    summary_rows = []
    for algo in all_eval['algorithm'].unique():
        algo_df = all_eval[all_eval['algorithm'] == algo]
        if algo_df.empty:
            continue
        
        max_step = algo_df['timestep'].max()
        cutoff = max_step * 0.9
        final_df = algo_df[algo_df['timestep'] >= cutoff]
        
        stats = final_df[list(PYMARL_METRIC_MAP.values())].mean().to_dict()
        stats['algorithm'] = algo
        summary_rows.append(stats)
    
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        # 调整列顺序
        cols = ['algorithm'] + [c for c in summary_df.columns if c != 'algorithm']
        summary_df = summary_df[cols]
        print(summary_df.to_string(index=False))
        summary_df.to_csv(OUTPUT_DIR / "final_performance_summary.csv", index=False)

if __name__ == "__main__":
    main()

