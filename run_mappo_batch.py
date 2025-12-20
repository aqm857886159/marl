"""
批量运行 MAPPO 超参搜索与入围复训脚本

阶段设计：
- 阶段1（筛选）：每个配置最多跑 2M 步，在 1M 步后若奖励低于基线的 early_stop_ratio 则提前停止。
- 阶段2（入围复训）：对入围配置关闭早停，跑到 3M 步。

使用方法（示例）：
  python run_mappo_batch.py            # 运行默认搜索（前 MAX_SEARCH 组）
  python run_mappo_batch.py final      # 运行入围列表中的配置（跑满 3M 步）
  python run_mappo_batch.py --tag v2   # 不覆盖旧日志，输出到带 tag 的新文件
  python run_mappo_batch.py final --tag v2

注意：
- 基线奖励请先用默认 MAPPO 跑一条 1M/2M 结果记录后填写 BASELINE_REWARD。
- 日志输出目录：hparam_logs/mappo/cfg_{id}_seed_{seed}_{phase}.json
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from MRRL import ENV_CONFIG
from mappo_trainer import MAPPOTrainer

# ======================
# 可调节参数
# ======================

# 搜索空间
LR_ACT = [1e-4, 2e-4, 3e-4, 5e-4]
LR_CRI = [4e-4, 6e-4, 8e-4, 1e-3]
BATCH = [64, 128]
CLIP = [0.15, 0.2, 0.25]
ENTROPY = [0.0, 0.005, 0.01]
NUM_SGD = [4, 6]

# 阶段设置
SEARCH_MAX_STEPS = 2_000_000
FINAL_MAX_STEPS = 3_000_000  # 入围配置跑到 3M
EARLY_STOP_STEPS = 1_000_000
EARLY_STOP_RATIO = 0.8  # 早停容忍度：允许比基线差 (1-RATIO) 的幅度，见 _early_stop_threshold()
EARLY_STOP_WINDOW = 3  # 建议1：用滑动均值判定早停，窗口大小
EVAL_INTERVAL = 50_000
EVAL_EPISODES = 5
MAX_SEARCH = 40  # 方案B：分层抽样 40 组配置做筛选
SEARCH_SAMPLE_SEED = 42  # 固定随机种子，保证每次抽到的 cfg 可复现
SAMPLING_METHOD = "stratified_lhs"  # 方案B：分层/近似LHS抽样
SAVE_SELECTED_CONFIGS = True  # 将本次抽样到的 cfg 列表落盘，确保论文/代码一致

# 需要先跑一条基线（1M~2M），填入此处；未设置则不触发早停
# ⚠️ 旧方式：手填 BASELINE_REWARD，容易和论文/日志不一致。
# ✅ 新方式：优先从根目录 baselines.json 读取（single source of truth）。
BASELINE_REWARD = -1200  # 兜底值：若 baselines.json 不存在或读取失败则使用

# 入围列表（手动维护 cfg_id），用于 final 模式
FINAL_CANDIDATES = []  # 例如 [3, 7, 12]
AUTO_TOPK = 5  # 入围 Top-K（按 avg_latency_ms 越小越好）


def _load_baseline_reward() -> float | None:
    """
    从 baselines.json 读取 MAPPO 基线 reward。
    返回 None 表示不启用早停基线（需要用户手动设置/检查）。
    """
    p = Path("baselines.json")
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        m = data.get("mappo", {})
        val = m.get("reward_baseline", None)
        return None if val is None else float(val)
    except Exception as e:
        print(f"[WARN] 读取 baselines.json 失败，将使用脚本内 BASELINE_REWARD: {e}")
        return None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _tag_suffix(tag: str | None) -> str:
    return "" if not tag else f"_{tag}"


def _early_stop_threshold(baseline_reward: float) -> float:
    """
    计算早停阈值（适配负 reward 的情况）。

    目标：当“最近 reward 明显比基线差”才早停。
    - 设 baseline = b，容忍差幅 = (1 - ratio) * |b|
    - 阈值 = b - 容忍差幅

    例：b=-1200, ratio=0.8 -> 阈值 = -1200 - 0.2*1200 = -1440
    即：只有 worse than -1440 才早停；不会出现 -1130 这种更好的 reward 反而被早停。
    """
    margin = (1.0 - float(EARLY_STOP_RATIO)) * abs(float(baseline_reward))
    return float(baseline_reward) - margin


def _balanced_choices(values: list[Any], n: int, rng: random.Random) -> list[Any]:
    """
    让每个取值在 n 次抽样中出现次数尽量均匀（差最多 1）。
    """
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
    """
    方案B：分层/近似 LHS 的抽样方式。
    对每个维度生成“均匀覆盖”的取值序列，然后按索引拼装成 n 组配置。
    """
    lr_act_seq = _balanced_choices(LR_ACT, n, rng)
    lr_cri_seq = _balanced_choices(LR_CRI, n, rng)
    batch_seq = _balanced_choices(BATCH, n, rng)
    clip_seq = _balanced_choices(CLIP, n, rng)
    ent_seq = _balanced_choices(ENTROPY, n, rng)
    nsgd_seq = _balanced_choices(NUM_SGD, n, rng)

    combos = list(zip(lr_act_seq, lr_cri_seq, batch_seq, clip_seq, ent_seq, nsgd_seq))

    # 去重：若偶发重复，则从全空间随机补齐
    seen = set()
    unique = []
    for c in combos:
        if c in seen:
            continue
        seen.add(c)
        unique.append(c)

    if len(unique) < n:
        full = list(itertools.product(LR_ACT, LR_CRI, BATCH, CLIP, ENTROPY, NUM_SGD))
        rng.shuffle(full)
        for c in full:
            if c in seen:
                continue
            seen.add(c)
            unique.append(c)
            if len(unique) >= n:
                break

    return unique[:n]


def _write_selected_space(space: list[dict], tag: str | None):
    out = Path("hparam_logs/mappo")
    ensure_dir(out)
    with open(out / f"selected_search_space{_tag_suffix(tag)}.json", "w", encoding="utf-8") as f:
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


def _select_topk_from_logs(topk: int, tag: str | None) -> list[int]:
    """
    从 search 阶段的日志中自动选入围 Top-K（按最后一次评估的 avg_latency_ms）。
    输出 cfg_id 列表，并写入 hparam_logs/mappo/top_candidates.json。
    """
    log_dir = Path("hparam_logs/mappo")
    if not log_dir.exists():
        return []

    rows = []
    pattern = f"cfg_*_search{_tag_suffix(tag)}.json"
    for p in sorted(log_dir.glob(pattern)):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            log = data.get("log", [])
            if not log:
                continue
            # 取最近 3 个评估点的平均（更稳）
            tail = log[-3:] if len(log) >= 3 else log
            latencies = []
            rewards = []
            for x in tail:
                if "avg_latency_ms" in x and x["avg_latency_ms"] is not None:
                    v = float(x["avg_latency_ms"])
                    # latency=0 通常表示该指标缺失（train_episode 缺失时置 0），不能当成最优
                    if v > 0:
                        latencies.append(v)
                if "episode_reward" in x and x["episode_reward"] is not None:
                    rewards.append(float(x["episode_reward"]))

            latency = float("inf") if not latencies else sum(latencies) / len(latencies)
            reward = None if not rewards else sum(rewards) / len(rewards)
            # cfg_id：优先取文件内 cfg.id；如果缺失再从日志项取 cfg_id
            cfg_id = int(data.get("cfg", {}).get("id", tail[-1].get("cfg_id", -1)))
            rows.append({"cfg_id": cfg_id, "avg_latency_ms": latency, "reward": reward, "path": str(p)})
        except Exception:
            continue

    # 优先 latency（越小越好），若 latency 缺失则用 reward（越大越好，越接近 0）
    rows = [r for r in rows if r["cfg_id"] >= 0]

    def sort_key(r):
        if r["avg_latency_ms"] != float("inf"):
            return (0, r["avg_latency_ms"])
        if r.get("reward") is not None:
            return (1, -r["reward"])
        return (2, float("inf"))

    rows.sort(key=sort_key)
    selected = [r["cfg_id"] for r in rows[:topk]]

    ensure_dir(log_dir)
    with open(log_dir / f"top_candidates{_tag_suffix(tag)}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "topk": topk,
                "metric": "avg_latency_ms (lower is better)",
                "selected_cfg_ids": selected,
                "candidates": rows[: max(topk, 20)],
            },
            f,
            indent=2,
        )
    return selected

def run_one_cfg(cfg: dict, seed: int, max_steps: int, early_stop: bool, phase: str, tag: str | None):
    """单个配置的训练与评估循环。"""
    log_dir = Path("hparam_logs/mappo")
    ensure_dir(log_dir)
    out_path = log_dir / f"cfg_{cfg['id']}_seed_{seed}_{phase}{_tag_suffix(tag)}.json"
    if out_path.exists():
        print(f"[SKIP] {out_path} 已存在，跳过。")
        return

    trainer = MAPPOTrainer(ENV_CONFIG, seed=seed)

    # 覆盖超参
    trainer.mappo.actor_optimizer.param_groups[0]["lr"] = cfg["lr_act"]
    trainer.mappo.critic_optimizer.param_groups[0]["lr"] = cfg["lr_cri"]
    trainer.mappo.clip_param = cfg["clip"]
    trainer.mappo.entropy_coeff = cfg["entropy"]
    trainer.mappo.num_sgd_iter = cfg["num_sgd"]
    # ✅ Batch 搜索真正生效：控制 PPO update 的 mini-batch 大小
    trainer.mappo.mini_batch_size = cfg["batch"]

    results = []
    total_steps = 0
    t0 = time.time()
    early_stop_triggered = False
    early_stop_reason = ""
    recent_rewards = deque(maxlen=EARLY_STOP_WINDOW)
    threshold = None if BASELINE_REWARD is None else _early_stop_threshold(BASELINE_REWARD)

    while total_steps < max_steps:
        ep_stats = trainer.train_episode(explore=True, record_buffer=True)
        total_steps += ep_stats["episode_steps"]

        # 定期评估
        if total_steps % EVAL_INTERVAL == 0 or total_steps >= max_steps:
            eval_stats = trainer.evaluate(num_episodes=EVAL_EPISODES)
            log_entry = {
                "algorithm": "MAPPO",
                "cfg_id": cfg["id"],
                "seed": seed,
                "phase": phase,
                "timestep": total_steps,
                "episode_reward": ep_stats["episode_reward"],
                **eval_stats,
            }
            results.append(log_entry)
            # 建议1：保存最近评估的奖励做滑动均值早停
            recent_rewards.append(ep_stats["episode_reward"])
            print(
                f"[{phase}] cfg={cfg['id']} seed={seed} "
                f"steps={total_steps} reward={ep_stats['episode_reward']:.2f} "
                f"latency={eval_stats['avg_latency_ms']:.2f}"
            )

            # 早停判定
            if (
                early_stop
                and BASELINE_REWARD is not None
                and total_steps >= EARLY_STOP_STEPS
                and len(recent_rewards) == EARLY_STOP_WINDOW
                and sum(recent_rewards) / EARLY_STOP_WINDOW < threshold
            ):
                early_stop_triggered = True
                early_stop_reason = (
                    f"moving_avg_reward<threshold (ratio={EARLY_STOP_RATIO}, baseline={BASELINE_REWARD}) over "
                    f"last {EARLY_STOP_WINDOW} evals"
                )
                print(
                    f"[EARLY STOP] cfg={cfg['id']} seed={seed} "
                    f"at {total_steps} (reward {ep_stats['episode_reward']:.2f} "
                    f"< threshold {threshold:.2f}, mean_recent={sum(recent_rewards)/EARLY_STOP_WINDOW:.2f})"
                )
                break

        # 参数更新
        trainer.mappo.update()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cfg": cfg,
                "seed": seed,
                "phase": phase,
                "max_steps": max_steps,
                "total_steps": total_steps,
                "early_stop": early_stop_triggered,
                "early_stop_reason": early_stop_reason,
                "time_sec": time.time() - t0,
                "log": results,
            },
            f,
            indent=2,
        )
    print(f"[SAVE] {out_path} 完成，总步数 {total_steps}")


def build_search_space(tag: str | None):
    rng = random.Random(SEARCH_SAMPLE_SEED)
    if SAMPLING_METHOD == "stratified_lhs":
        combos = _sample_stratified_lhs(MAX_SEARCH, rng)
    else:
        combos = list(itertools.product(LR_ACT, LR_CRI, BATCH, CLIP, ENTROPY, NUM_SGD))
        rng.shuffle(combos)
        combos = combos[:MAX_SEARCH]
    space = []
    for idx, (la, lc, bs, cp, ent, nsgd) in enumerate(combos):
        space.append(
            {
                "id": idx,
                "lr_act": la,
                "lr_cri": lc,
                "batch": bs,
                "clip": cp,
                "entropy": ent,
                "num_sgd": nsgd,
            }
        )

    if SAVE_SELECTED_CONFIGS:
        _write_selected_space(space, tag)
    return space


def main():
    args = sys.argv[1:]
    final_mode = len(args) > 0 and args[0].lower() == "final"
    pick_mode = len(args) > 0 and args[0].lower() in ("pick", "select", "rank")
    tag = None
    if "--tag" in args:
        i = args.index("--tag")
        if i + 1 < len(args):
            tag = args[i + 1].strip()

    # 用 baselines.json 覆盖基线（保持一一对应）
    global BASELINE_REWARD
    loaded = _load_baseline_reward()
    if loaded is not None:
        BASELINE_REWARD = loaded
        print(f"[BASELINE] MAPPO reward_baseline from baselines.json = {BASELINE_REWARD}")
    else:
        print(f"[BASELINE] MAPPO reward_baseline using script default = {BASELINE_REWARD}")

    if pick_mode:
        selected = _select_topk_from_logs(AUTO_TOPK, tag)
        if selected:
            print(f"[PICK] top{AUTO_TOPK} cfg_ids:", selected)
        else:
            print("[PICK] 未找到可用的 search 日志（请先运行 search 阶段）。")
        return

    if final_mode:
        candidates = FINAL_CANDIDATES
        if not candidates:
            # 自动从 search 日志选 Top-K
            candidates = _select_topk_from_logs(AUTO_TOPK, tag)
            if candidates:
                print(f"[AUTO FINAL] selected top{AUTO_TOPK} cfg_ids from logs:", candidates)
        if not candidates:
            print("[WARN] 未找到入围配置：请先运行 search 或手动填写 FINAL_CANDIDATES。")
            return

        cfg_space = [c for c in build_search_space(tag) if c["id"] in candidates]
        max_steps = FINAL_MAX_STEPS
        early_stop_flag = False  # 入围不早停
        phase = "final"
    else:
        cfg_space = build_search_space(tag)
        max_steps = SEARCH_MAX_STEPS
        early_stop_flag = True
        phase = "search"

    for cfg in cfg_space:
        # 轮换 3 个种子：0/1/2
        seed = cfg["id"] % 3
        run_one_cfg(cfg, seed, max_steps=max_steps, early_stop=early_stop_flag, phase=phase, tag=tag)


if __name__ == "__main__":
    main()


