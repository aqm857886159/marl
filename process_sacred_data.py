import os
import json
import glob
import numpy as np

def convert_info_to_maddpg_format(sacred_dir, output_dir):
    """Converts Sacred info.json to MADDPG style eval_log.json and training_log.json"""
    info_path = os.path.join(sacred_dir, "info.json")
    if not os.path.exists(info_path):
        print(f"[WARN] info.json not found in {sacred_dir}")
        return

    print(f"Processing {info_path}...")
    with open(info_path, "r") as f:
        info_data = json.load(f)

    # 1. Convert Evaluation Logs (eval_log.json)
    eval_entries = []
    
    # Check for test_return_mean to align steps
    if "test_return_mean" not in info_data:
        print("[WARN] No test_return_mean in info.json, skipping eval conversion.")
    else:
        # Use the _T key for steps
        steps = info_data.get("test_return_mean_T", [])
        n_points = len(steps)
        
        # Helper to get value at index i safely
        def get_val(key, i):
            if key in info_data and i < len(info_data[key]):
                return info_data[key][i]
            return None

        for i in range(n_points):
            step = steps[i]
            entry = {"step": step}
            
            # Mapping: Output Key -> Sacred Key (in info.json)
            # Note: Sacred keys in info.json are like "test_avg_latency_ms_mean"
            mapping = {
                "avg_latency_ms": "test_avg_latency_ms_mean",
                "p99_latency_ms": "test_p99_latency_ms_mean", # Note: p99 might not be logged if not in keys
                "avg_energy_J": "test_avg_energy_J_mean",
                "throughput_tps": "test_throughput_tps_mean",
                "load_balance_jain": "test_load_balance_jain_mean",
                "deadline_violation_rate": "test_deadline_violation_rate_mean",
                "return": "test_return_mean"
            }
            
            for target_k, sacred_k in mapping.items():
                val = get_val(sacred_k, i)
                if val is not None:
                    entry[target_k] = val
            
            eval_entries.append(entry)

    os.makedirs(output_dir, exist_ok=True)
    if eval_entries:
        with open(os.path.join(output_dir, "eval_log.json"), "w", encoding="utf-8") as f:
            json.dump(eval_entries, f, indent=2)
        print(f"[SUCCESS] Generated {output_dir}/eval_log.json ({len(eval_entries)} entries)")

    # 2. Convert Training Logs (training_log.json)
    train_entries = []
    if "return_mean" in info_data:
        steps = info_data.get("return_mean_T", [])
        values = info_data.get("return_mean", [])
        
        for s, v in zip(steps, values):
            train_entries.append({"step": s, "reward": v})
            
    if train_entries:
        with open(os.path.join(output_dir, "training_log.json"), "w", encoding="utf-8") as f:
            json.dump(train_entries, f, indent=2)
        print(f"[SUCCESS] Generated {output_dir}/training_log.json ({len(train_entries)} entries)")

def main():
    # Base paths
    sacred_base = os.path.join("pymarl", "results", "sacred")
    qmix_base = os.path.join("pymarl", "results", "edge_qmix")
    
    # Detect mapping. We assume sacred IDs 1..N map to seeds 0..N? 
    # Or user ran seeds sequentially.
    # Based on ls, we have sacred/1, 2, 3 (failed/small), 4 (big).
    # Let's assume sacred/4 corresponds to Seed 0 (since it's the first successful one).
    
    # Manual mapping for now based on observation
    # Sacred 4 -> Seed 0
    
    mapping = {
        "4": "seed_0"
    }
    
    for sacred_id, seed_dir_name in mapping.items():
        sacred_dir = os.path.join(sacred_base, sacred_id)
        target_dir = os.path.join(qmix_base, seed_dir_name)
        
        if os.path.exists(sacred_dir):
            convert_info_to_maddpg_format(sacred_dir, target_dir)
        else:
            print(f"Sacred dir {sacred_dir} not found.")

if __name__ == "__main__":
    main()

