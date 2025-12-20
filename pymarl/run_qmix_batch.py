"""
批量运行 QMIX 超参搜索与入围复训脚本（基于 PyMARL）

阶段设计：
- 阶段1（筛选）：每个配置最多跑到 2M 步（通过 t_max 覆盖）。
  - 在达到 1M 步后，若测试指标明显劣于基线（见 baselines.json），则触发早停（真正节省算力）。
  - 若不满足早停条件，则继续跑满 2M。
- 阶段2（入围复训）：对入围配置跑到 3M 步（不早停，保证最终对比口径一致）。

使用方法（在 pymarl 目录下执行）：
  python run_qmix_batch.py            # 运行默认搜索（前 MAX_SEARCH 组）
  python run_qmix_batch.py final      # 运行入围列表（FINAL_CANDIDATES），跑满 3M 步

说明：
- 输出目录：results/edge_qmix/cfg_{id}_seed_{seed}_{phase}/
- 每次运行结束自动调用 convert_sacred_to_maddpg_format 生成 eval/training 日志，便于下游绘图。
- 早停逻辑在 PyMARL 训练循环 `src/run.py` 中实现，本脚本在 search 阶段通过命令行参数启用并传入 baseline。
"""

from __future__ import annotations

import itertools
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_qmix_seeds import convert_sacred_to_maddpg_format, get_latest_sacred_dir

# ======================
# 可调节参数
# ======================

# 搜索空间
RNN = [64, 128]
MIX = [32, 64]
LR = [2e-4, 3e-4, 5e-4]
TGT = [100, 200, 400]
EPS_END = [0.05, 0.1]
EPS_STEPS = [5e5, 1e6]

# 阶段设置（与 MAPPO 口径一致）
# - search：最多 2M；在 1M 后如果明显差于基线则早停，否则继续跑满 2M
# - final：入围后跑满 3M（不早停）
SEARCH_STEPS = 2_000_000
FINAL_STEPS = 3_000_000
MAX_SEARCH = 40  # 方案B：分层抽样 40 组配置做筛选
SEARCH_SAMPLE_SEED = 42  # 固定随机种子，保证抽样可复现
SAMPLING_METHOD = "stratified_lhs"  # 方案B：分层/近似LHS抽样
SAVE_SELECTED_CONFIGS = True  # 落盘保存抽到的 cfg 列表，便于论文对齐

# 入围列表（手动维护 cfg_id），用于 final 模式
FINAL_CANDIDATES = []  # 例如 [2, 5, 9]
AUTO_TOPK = 5  # 入围 Top-K（按 avg_latency_ms 越小越好；若缺失则退化为 reward）

EARLY_STOP_ENABLE = True
# 与 MAPPO 一致：1M 后开始早停判定
EARLY_STOP_STEPS = 1_000_000
EARLY_STOP_WINDOW = 3
EARLY_STOP_RATIO = 0.8
# ✅ 与 MAPPO 对齐：统一用 reward/return 做早停（避免不同算法用不同指标早停导致口径不一致）
EARLY_STOP_METRIC = "return"  # 对应 PyMARL 的 test_return_mean（越大越好）

# 评估口径（与 run_mappo_batch.py 对齐）
# - MAPPO: 每 50k 步评估一次，每次评估 5 个 episode
# - QMIX: 默认 edge_marl.yaml 里是 10 个 episode，这会显著拖慢 wall-clock（评估开销翻倍）
#   这里在 batch 脚本里显式覆盖为 5，保证与 MAPPO 一致且加速。
EVAL_INTERVAL = 50_000
EVAL_EPISODES = 5

# 训练 batch（影响显存占用）。
# 经验：单进程 24GB 显存通常能扛住 128；但你现在常用 2 个 shard 并行跑，
# 两个进程叠加容易在某些配置（例如 rnn_hidden_dim=128）出现 CUDA OOM。
# 因此我们提供“失败自动降档重试”的 batch_size 列表：优先 128，OOM 则降到 64/32。
TRAIN_BATCH_CANDIDATES = [128, 64, 32]

# 日志频率（加速：减少 stdout/sacred 写盘与 console 格式化开销）
# 说明：PyMARL 训练循环里每到 log_interval 会打印一次 Recent Stats；learner/runner 也有各自的日志节奏。
# 原始配置是 5000（非常频繁，Windows 上尤其慢），这里调大到 20000，通常能明显省时。
# 若你还嫌慢，可改成 50000，但会更难观察训练早期是否异常。
LOG_INTERVAL = 20_000
RUNNER_LOG_INTERVAL = 20_000
LEARNER_LOG_INTERVAL = 20_000


def _load_qmix_baseline_return() -> float | None:
    """
    从项目根目录 baselines.json 读取 qmix 的 reward_window_mean（single source of truth）。
    运行目录在 pymarl 下，所以 baselines.json 在 .. 。
    """
    p = Path("..") / "baselines.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        q = data.get("qmix", {})
        # baselines.json 推荐字段：reward_window_mean（更稳）；兼容 fallback
        val = q.get("reward_window_mean", None)
        if val is None:
            val = q.get("reward_at_nearest", None)
        return None if val is None else float(val)
    except Exception as e:
        print(f"[WARN] 读取 ../baselines.json 失败，QMIX 早停将不启用 baseline: {e}")
        return None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _parse_ids_token(token: str) -> list[int]:
    """
    支持:
    - "3" -> [3]
    - "0-10" -> [0..10]
    """
    s = token.strip()
    if not s:
        return []
    if "-" in s:
        a, b = s.split("-", 1)
        a = int(a.strip())
        b = int(b.strip())
        lo, hi = (a, b) if a <= b else (b, a)
        return list(range(lo, hi + 1))
    return [int(s)]


def _parse_ids_arg(ids_arg: str | None) -> set[int] | None:
    """
    解析 --ids 参数。示例：
    - "--ids 0-19" 或 "--ids 0-19,25,30-39"
    返回 set[int]；若未提供则返回 None（表示不过滤）。
    """
    if ids_arg is None:
        return None
    out: list[int] = []
    for part in ids_arg.split(","):
        out.extend(_parse_ids_token(part))
    return set(out)


def _select_by_shard(items: list[dict], shard_idx: int, num_shards: int) -> list[dict]:
    """
    将 cfg_space 按 id 做一致性分片，便于多窗口并行且不冲突：
    - shard_idx: 0..num_shards-1
    - 规则：cfg["id"] % num_shards == shard_idx
    """
    if num_shards <= 1:
        return items
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"invalid shard_idx={shard_idx}, num_shards={num_shards}")
    return [c for c in items if (int(c["id"]) % int(num_shards)) == int(shard_idx)]


def _balanced_choices(values: list[Any], n: int, rng: random.Random) -> list[Any]:
    k = len(values)
    if k == 0:
        raise ValueError("values 为空")
    base = n // k
    rem = n % k
    counts = [base + (1 if i < rem else 0) for i in range(k)]
    out = []
    for v, c in zip(values, counts):
        out.extend([v] * c)
    rng.shuffle(out)
    return out


def _sample_stratified_lhs(n: int, rng: random.Random):
    rnn_seq = _balanced_choices(RNN, n, rng)
    mix_seq = _balanced_choices(MIX, n, rng)
    lr_seq = _balanced_choices(LR, n, rng)
    tgt_seq = _balanced_choices(TGT, n, rng)
    eend_seq = _balanced_choices(EPS_END, n, rng)
    estep_seq = _balanced_choices(EPS_STEPS, n, rng)

    combos = list(zip(rnn_seq, mix_seq, lr_seq, tgt_seq, eend_seq, estep_seq))
    seen = set()
    unique = []
    for c in combos:
        if c in seen:
            continue
        seen.add(c)
        unique.append(c)

    if len(unique) < n:
        full = list(itertools.product(RNN, MIX, LR, TGT, EPS_END, EPS_STEPS))
        rng.shuffle(full)
        for c in full:
            if c in seen:
                continue
            seen.add(c)
            unique.append(c)
            if len(unique) >= n:
                break

    return unique[:n]


def _write_selected_space(space: list[dict]):
    out = Path("results/edge_qmix")
    ensure_dir(out)
    with open(out / "selected_search_space.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sampling_method": SAMPLING_METHOD,
                "seed": SEARCH_SAMPLE_SEED,
                "max_search": MAX_SEARCH,
                "space": space,
            },
            f,
            indent=2,
        )


def _select_topk_from_results(topk: int) -> list[int]:
    """
    从 results/edge_qmix/cfg_*_search/eval_log.json 中选 Top-K。
    主指标：avg_latency_ms 最小；若缺失则用 reward 最大（更接近 0）。
    """
    base = Path("results/edge_qmix")
    if not base.exists():
        return []

    rows = []
    for d in sorted(base.glob("cfg_*_seed_*_search")):
        eval_path = d / "eval_log.json"
        if not eval_path.exists():
            continue
        try:
            eval_entries = json.loads(eval_path.read_text(encoding="utf-8"))
            if not eval_entries:
                continue
            last = eval_entries[-1]
            cfg_id = int(d.name.split("_")[1])
            latency = last.get("avg_latency_ms", None)
            reward = last.get("reward", None)
            rows.append(
                {
                    "cfg_id": cfg_id,
                    "avg_latency_ms": None if latency is None else float(latency),
                    "reward": None if reward is None else float(reward),
                    "dir": str(d),
                }
            )
        except Exception:
            continue

    # 排序规则
    def key(r):
        if r["avg_latency_ms"] is not None:
            return (0, r["avg_latency_ms"])  # 越小越好
        # reward 越大（越接近 0）越好
        return (1, -(r["reward"] if r["reward"] is not None else float("-inf")))

    rows.sort(key=key)
    selected = [r["cfg_id"] for r in rows[:topk]]
    with open(base / "top_candidates.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "topk": topk,
                "metric": "avg_latency_ms (lower is better), fallback: reward (higher is better)",
                "selected_cfg_ids": selected,
                "candidates": rows[: max(topk, 20)],
            },
            f,
            indent=2,
        )
    return selected


def run_one_cfg(cfg: dict, seed: int, total_steps: int, phase: str):
    target_dir = Path(f"results/edge_qmix/cfg_{cfg['id']}_seed_{seed}_{phase}")
    if target_dir.exists():
        print(f"[SKIP] {target_dir} 已存在，跳过。")
        return

    def _build_cmd(train_batch_size: int) -> list[str]:
        cmd = [
            sys.executable,
            "src/main.py",
            "--config=edge_qmix",
            "--env-config=edge_marl",
            "with",
            f"seed={seed}",
            # PyMARL 配置键（注意：不是 agent.* 这种嵌套；否则 sacred 会报 ConfigAddedError）
            # 真实使用处：src/components/action_selectors.py 读取 epsilon_finish / epsilon_anneal_time
            f"rnn_hidden_dim={cfg['rnn']}",
            f"mixing_embed_dim={cfg['mix']}",
            f"lr={cfg['lr']}",
            f"target_update_interval={cfg['tgt']}",
            f"epsilon_finish={cfg['eps_end']}",
            f"epsilon_anneal_time={int(cfg['eps_steps'])}",
            # 训练总步数：PyMARL 主循环以 t_max 为上限（src/run.py）
            f"t_max={int(total_steps)}",
            # 评估设置：与 MAPPO 对齐 + 加速
            f"test_interval={int(EVAL_INTERVAL)}",
            f"test_nepisode={int(EVAL_EPISODES)}",
            # 日志设置：减少打印/写盘频率以加速
            f"log_interval={int(LOG_INTERVAL)}",
            f"runner_log_interval={int(RUNNER_LOG_INTERVAL)}",
            f"learner_log_interval={int(LEARNER_LOG_INTERVAL)}",
            # 训练 batch（显存关键项）
            f"batch_size={int(train_batch_size)}",
        ]
        return cmd

    # search 阶段启用早停（真正省时间）
    if phase == "search" and EARLY_STOP_ENABLE:
        baseline_ret = _load_qmix_baseline_return()
        if baseline_ret is not None:
            early_stop_args = [
                "early_stop_enable=True",
                f"early_stop_steps={int(EARLY_STOP_STEPS)}",
                f"early_stop_window={int(EARLY_STOP_WINDOW)}",
                f"early_stop_ratio={float(EARLY_STOP_RATIO)}",
                f"early_stop_metric={EARLY_STOP_METRIC}",
                f"early_stop_baseline={float(baseline_ret)}",
            ]
        else:
            print("[EARLY STOP] baselines.json 未提供 QMIX reward baseline，本次 search 不启用早停。")
            early_stop_args = []
    else:
        early_stop_args = []

    t0 = time.time()

    # 运行环境：降低 CUDA 内存碎片导致的 OOM（尤其是多进程并行时）
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # OOM 容错：自动降 batch_size 重试，避免整个 batch 因为单个 cfg 直接中断
    last_err: Exception | None = None
    for bs in TRAIN_BATCH_CANDIDATES:
        cmd = _build_cmd(bs) + early_stop_args
        try:
            print(f"[RUN] cfg={cfg['id']} seed={seed} phase={phase} batch_size={bs}")
            subprocess.run(cmd, check=True, env=env)
            last_err = None
            break
        except subprocess.CalledProcessError as e:
            last_err = e
            # 常见：CUDA OOM / 其他运行时错误。这里先提示，再尝试降档。
            print(f"[FAIL] cfg={cfg['id']} seed={seed} phase={phase} batch_size={bs} exit={e.returncode}")
            if bs != TRAIN_BATCH_CANDIDATES[-1]:
                print(f"[RETRY] 尝试降低 batch_size -> {TRAIN_BATCH_CANDIDATES[TRAIN_BATCH_CANDIDATES.index(bs)+1]}")
            continue

    if last_err is not None:
        # 记录失败信息，避免你事后不知道哪个 cfg 因为 OOM/异常没跑完
        ensure_dir(target_dir)
        meta = {
            "cfg": cfg,
            "seed": seed,
            "phase": phase,
            "total_timesteps": total_steps,
            "time_sec": time.time() - t0,
            "status": "failed",
            "error": str(last_err),
        }
        with open(target_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[FAIL-SAVE] {target_dir} 记录失败信息（不会转换 sacred->logs）。继续下一个 cfg。")
        return

    # Sacred -> MADDPG 格式转换
    latest = get_latest_sacred_dir()
    if latest:
        ensure_dir(target_dir)
        convert_sacred_to_maddpg_format(latest, str(target_dir))

    meta = {
        "cfg": cfg,
        "seed": seed,
        "phase": phase,
        "total_timesteps": total_steps,
        "time_sec": time.time() - t0,
    }
    with open(target_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVE] {target_dir} 完成，总步数 {total_steps}")


def build_search_space():
    rng = random.Random(SEARCH_SAMPLE_SEED)
    if SAMPLING_METHOD == "stratified_lhs":
        combos = _sample_stratified_lhs(MAX_SEARCH, rng)
    else:
        combos = list(itertools.product(RNN, MIX, LR, TGT, EPS_END, EPS_STEPS))
        rng.shuffle(combos)
        combos = combos[:MAX_SEARCH]
    space = []
    for idx, (r, m, lr, tgt, eend, estep) in enumerate(combos):
        space.append(
            {
                "id": idx,
                "rnn": r,
                "mix": m,
                "lr": lr,
                "tgt": tgt,
                "eps_end": eend,
                "eps_steps": estep,
            }
        )

    if SAVE_SELECTED_CONFIGS:
        _write_selected_space(space)
    return space


def main():
    args = sys.argv[1:]

    # 轻量参数解析（保持脚本纯标准库、兼容原有用法）
    # 支持：
    # - python run_qmix_batch.py
    # - python run_qmix_batch.py final
    # - python run_qmix_batch.py --ids 0-19
    # - python run_qmix_batch.py --shard 0/2
    final_mode = len(args) > 0 and args[0].lower() == "final"
    ids_arg: str | None = None
    shard_idx: int | None = None
    num_shards: int | None = None

    if "--ids" in args:
        i = args.index("--ids")
        if i + 1 < len(args):
            ids_arg = args[i + 1].strip()

    if "--shard" in args:
        i = args.index("--shard")
        if i + 1 < len(args):
            s = args[i + 1].strip()
            # "0/2" 形式
            if "/" in s:
                a, b = s.split("/", 1)
                shard_idx = int(a.strip())
                num_shards = int(b.strip())
            else:
                raise ValueError("--shard 需要形如 0/2 的参数")

    if final_mode:
        candidates = FINAL_CANDIDATES
        if not candidates:
            candidates = _select_topk_from_results(AUTO_TOPK)
            if candidates:
                print(f"[AUTO FINAL] selected top{AUTO_TOPK} cfg_ids from results:", candidates)
        if not candidates:
            print("[WARN] 未找到入围配置：请先运行 search 或手动填写 FINAL_CANDIDATES。")
            return

        cfg_space = [c for c in build_search_space() if c["id"] in candidates]
        total_steps = FINAL_STEPS
        phase = "final"
    else:
        cfg_space = build_search_space()
        total_steps = SEARCH_STEPS
        phase = "search"

    # 过滤/分片：用于多窗口并行跑不同 cfg 子集（避免目录冲突）
    ids_set = _parse_ids_arg(ids_arg)
    if ids_set is not None:
        cfg_space = [c for c in cfg_space if int(c["id"]) in ids_set]
        print(f"[FILTER] --ids={ids_arg} -> {len(cfg_space)} configs")

    if shard_idx is not None and num_shards is not None:
        cfg_space = _select_by_shard(cfg_space, shard_idx=shard_idx, num_shards=num_shards)
        print(f"[SHARD] --shard {shard_idx}/{num_shards} -> {len(cfg_space)} configs")

    if not cfg_space:
        print("[WARN] 过滤后 cfg_space 为空，退出。")
        return

    for cfg in cfg_space:
        seed = cfg["id"] % 3  # 轮换 3 个种子
        run_one_cfg(cfg, seed, total_steps, phase)


if __name__ == "__main__":
    main()


