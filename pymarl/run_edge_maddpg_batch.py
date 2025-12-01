import argparse
import subprocess
from pathlib import Path
from typing import List


def run_seeds(
    seeds: List[int],
    t_max: int,
    device: str,
    max_parallel: int,
):
    """并行/串行启动多个 MADDPG 训练进程。"""
    script = str(Path(__file__).parent / "run_edge_maddpg.py")
    active = []
    seeds = list(seeds)

    while seeds or active:
        # 启动新的进程，直到达到并行上限
        while seeds and len(active) < max_parallel:
            s = seeds.pop(0)
            cmd = ["python", script, "--seed", str(s), "--device", device]
            if t_max is not None:
                cmd.extend(["--t-max", str(t_max)])
            print(f"[LAUNCH] seed={s} cmd={' '.join(cmd)}")
            p = subprocess.Popen(cmd)
            active.append((s, p))

        # 检查已有进程
        still_active = []
        for s, p in active:
            ret = p.poll()
            if ret is None:
                still_active.append((s, p))
            else:
                print(f"[DONE] seed={s} exit_code={ret}")
        active = still_active


def parse_args():
    parser = argparse.ArgumentParser(description="批量运行 EdgeSimGym MADDPG (PyMARL)")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="要运行的随机种子列表，默认 0 1 2 3 4",
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=None,
        help="覆盖总训练步数（默认读取 edge_maddpg.yaml 中的 total_timesteps）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["auto", "cpu", "cuda"],
        help="训练设备：auto/cpu/cuda，默认 cuda",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="最多同时运行的进程数，默认 2（避免显存爆炸）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_seeds(args.seeds, args.t_max, args.device, args.max_parallel)


