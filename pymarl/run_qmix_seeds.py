import subprocess
import sys
import os
import json
import glob
from pathlib import Path

def get_latest_sacred_dir(base_path="results/sacred"):
    """找到最近一次 Sacred 实验的目录"""
    dirs = glob.glob(os.path.join(base_path, "*"))
    dirs = [d for d in dirs if os.path.isdir(d) and d.split(os.sep)[-1].isdigit()]
    if not dirs:
        return None
    # 按数字 ID 排序，取最大的（最新的）
    dirs.sort(key=lambda x: int(x.split(os.sep)[-1]))
    return dirs[-1]

def convert_sacred_to_maddpg_format(sacred_dir, output_dir):
    """将 Sacred 的 metrics.json/info.json 转换为 MADDPG 风格的 eval_log.json / training_log.json

    兼容两种来源：
    - metrics.json（sacred 收尾写入；更结构化）
    - info.json（训练过程中持续写入；即使异常退出也通常存在）
    """
    metrics_path = os.path.join(sacred_dir, "metrics.json")
    info_path = os.path.join(sacred_dir, "info.json")

    sacred_data = None
    mode = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            sacred_data = json.load(f)
        mode = "metrics"
    elif os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            sacred_data = json.load(f)
        mode = "info"
    else:
        print(f"[WARN] 未找到 metrics.json 或 info.json: {sacred_dir}")
        return

    def _series(key: str):
        """返回 (steps, values) 或 (None, None)"""
        if mode == "metrics":
            if key not in sacred_data:
                return None, None
            obj = sacred_data[key]
            return obj.get("steps"), obj.get("values")
        # info.json: key -> list, key_T -> list
        if key not in sacred_data or f"{key}_T" not in sacred_data:
            return None, None
        return sacred_data.get(f"{key}_T"), sacred_data.get(key)

    # 1. 转换 Evaluation Logs (eval_log.json)
    # Sacred key: "test_avg_latency_ms_mean" -> {"steps": [], "values": []}
    # Target key: "avg_latency_ms"
    eval_entries = []
    
    # 假设所有 test_ 指标的 step 都是对齐的，我们以 test_return_mean 为基准
    steps, _ = _series("test_return_mean")
    if not steps:
        print("[WARN] 找不到 test_return_mean，跳过 Eval 转换。")
    else:
        n_points = len(steps)
        
        for i in range(n_points):
            entry = {"step": steps[i]}
            # 映射关键指标
            # EdgeMARLEnv 返回 avg_latency_ms -> PyMARL logs test_avg_latency_ms_mean
            mapping = {
                "avg_latency_ms": "test_avg_latency_ms_mean",
                "p99_latency_ms": "test_p99_latency_ms_mean",
                "avg_energy_J": "test_avg_energy_J_mean",
                "throughput_tps": "test_throughput_tps_mean",
                "load_balance_jain": "test_load_balance_jain_mean",
                "deadline_violation_rate": "test_deadline_violation_rate_mean"
            }
            
            for target_k, sacred_k in mapping.items():
                s_steps, s_vals = _series(sacred_k)
                if s_vals is not None and len(s_vals) > i:
                    entry[target_k] = s_vals[i]
            
            eval_entries.append(entry)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_log.json"), "w", encoding="utf-8") as f:
        json.dump(eval_entries, f, indent=2)
        print(f"[CONVERT] 已生成: {output_dir}/eval_log.json ({len(eval_entries)} 条记录)")

    # 2. 转换 Training Logs (training_log.json)
    # 这里的粒度不如 MADDPG 细（MADDPG 记录每条 Episode），这里是 periodic log
    # 但画学习曲线足够了
    train_entries = []
    steps, values = _series("return_mean")
    if steps and values:
        for s, v in zip(steps, values):
            train_entries.append({"step": s, "reward": v})
            
    with open(os.path.join(output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_entries, f, indent=2)
        print(f"[CONVERT] 已生成: {output_dir}/training_log.json ({len(train_entries)} 条记录)")


def run_qmix_seeds():
    # 确保在 pymarl 目录下运行
    if not os.path.exists("src/main.py"):
        print("Error: 请在 pymarl 目录下运行此脚本 (cd pymarl)")
        return

    seeds = [0, 1, 2, 3, 4]
    
    for seed in seeds:
        print(f"\n" + "="*40)
        print(f"   启动 QMIX 训练: 种子 {seed}")
        print(f"="*40 + "\n")
        
        cmd = [
            sys.executable, 
            "src/main.py",
            "--config=edge_qmix", 
            "--env-config=edge_marl",
            "with",
            f"seed={seed}"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # --- 后处理：转换数据格式 ---
            latest_run = get_latest_sacred_dir()
            if latest_run:
                print(f"[INFO] 检测到最新 Sacred 运行目录: {latest_run}")
                # 目标目录: results/edge_qmix/seed_0/
                target_dir = os.path.join("results", "edge_qmix", f"seed_{seed}")
                convert_sacred_to_maddpg_format(latest_run, target_dir)
            else:
                print("[ERROR] 无法找到 Sacred 目录，数据可能未保存。")

        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] 种子 {seed} 训练失败！停止后续任务。")
            print(f"错误详情: {e}")
            break
        except KeyboardInterrupt:
            print(f"\n[STOP] 用户中断训练。")
            break

if __name__ == "__main__":
    run_qmix_seeds()
