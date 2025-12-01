import json
import os
from typing import Dict, Optional


def convert_info_to_maddpg_format(sacred_dir: str, output_dir: str) -> None:
    """将 Sacred 的 info.json 转成与 MADDPG 相同的数据结构"""
    info_path = os.path.join(sacred_dir, "info.json")
    if not os.path.exists(info_path):
        print(f"[WARN] info.json not found in {sacred_dir}")
        return

    print(f"Processing {info_path}...")
    with open(info_path, "r") as f:
        info_data = json.load(f)

    # 1. Eval 日志
    eval_entries = []
    if "test_return_mean" not in info_data:
        print("[WARN] No test_return_mean in info.json, skipping eval conversion.")
    else:
        steps = info_data.get("test_return_mean_T", [])
        n_points = len(steps)

        def get_val(key, idx):
            if key in info_data and idx < len(info_data[key]):
                return info_data[key][idx]
            return None

        metric_mapping = {
            "avg_latency_ms": "test_avg_latency_ms_mean",
            "p99_latency_ms": "test_p99_latency_ms_mean",
            "avg_energy_J": "test_avg_energy_J_mean",
            "throughput_tps": "test_throughput_tps_mean",
            "load_balance_jain": "test_load_balance_jain_mean",
            "deadline_violation_rate": "test_deadline_violation_rate_mean",
            "return": "test_return_mean",
        }

        for i in range(n_points):
            entry = {"step": steps[i]}
            for target_key, sacred_key in metric_mapping.items():
                val = get_val(sacred_key, i)
                if val is not None:
                    entry[target_key] = val
            eval_entries.append(entry)

    os.makedirs(output_dir, exist_ok=True)
    if eval_entries:
        eval_path = os.path.join(output_dir, "eval_log.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_entries, f, indent=2)
        print(f"[SUCCESS] Generated {eval_path} ({len(eval_entries)} entries)")

    # 2. Training 日志
    train_entries = []
    if "return_mean" in info_data:
        steps = info_data.get("return_mean_T", [])
        values = info_data.get("return_mean", [])
        for s, v in zip(steps, values):
            train_entries.append({"step": s, "reward": v})

    if train_entries:
        train_path = os.path.join(output_dir, "training_log.json")
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_entries, f, indent=2)
        print(f"[SUCCESS] Generated {train_path} ({len(train_entries)} entries)")


def detect_seed_from_run(run_path: str) -> Optional[int]:
    if not os.path.exists(run_path):
        return None
    try:
        with open(run_path, "r") as f:
            run_data = json.load(f)
        meta = run_data.get("meta", {})
        cfg_updates = meta.get("config_updates", {})
        seed = cfg_updates.get("seed")
        if seed is None:
            seed = run_data.get("config", {}).get("seed")
        return int(seed) if seed is not None else None
    except Exception as exc:
        print(f"[WARN] 无法从 {run_path} 解析 seed: {exc}")
        return None


def detect_seed_from_config(config_path: str) -> Optional[int]:
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        seed = config_data.get("seed")
        return int(seed) if seed is not None else None
    except Exception as exc:
        print(f"[WARN] 无法从 {config_path} 解析 seed: {exc}")
        return None


def auto_map_sacred_runs(sacred_base: str) -> Dict[int, str]:
    numeric_dirs = [d for d in os.listdir(sacred_base) if d.isdigit()]
    if not numeric_dirs:
        print("[WARN] sacred 目录下没有数字命名的运行结果。")
        return {}

    seed_to_run: Dict[int, str] = {}
    for run_id in sorted(numeric_dirs, key=lambda x: int(x)):
        sacred_dir = os.path.join(sacred_base, run_id)
        seed = detect_seed_from_run(os.path.join(sacred_dir, "run.json"))
        if seed is None:
            seed = detect_seed_from_config(os.path.join(sacred_dir, "config.json"))
        if seed is None:
            print(f"[WARN] {sacred_dir} 缺少 seed 信息，已跳过。")
            continue

        previous = seed_to_run.get(seed)
        if previous is not None and int(run_id) <= int(previous):
            continue  # 已有更新的 run
        if previous is not None:
            print(f"[INFO] seed {seed} 的旧运行 {previous} 被较新的 {run_id} 覆盖。")
        seed_to_run[seed] = run_id

    return seed_to_run


def main():
    sacred_base = os.path.join("pymarl", "results", "sacred")
    qmix_base = os.path.join("pymarl", "results", "edge_qmix")
    os.makedirs(qmix_base, exist_ok=True)

    seed_mapping = auto_map_sacred_runs(sacred_base)
    if not seed_mapping:
        print("[ERROR] 未找到任何可用的 Sacred 运行，无法生成 QMIX 数据。")
        return

    for seed, run_id in sorted(seed_mapping.items()):
        sacred_dir = os.path.join(sacred_base, run_id)
        target_dir = os.path.join(qmix_base, f"seed_{seed}")
        print(f"\n[INFO] seed {seed}: Sacred 目录 {run_id} -> {target_dir}")
        convert_info_to_maddpg_format(sacred_dir, target_dir)


if __name__ == "__main__":
    main()

