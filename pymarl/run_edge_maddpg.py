import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from maddpg_algo import MADDPG, MADDPGConfig, OrnsteinUhlenbeckNoise, ReplayBuffer
from src.envs.edge_marl_env import EdgeMARLEnv


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env_args(env_config: dict, seed: int, action_mode: str = "hybrid"):
    args = dict(env_config.get("env_args", {}))
    args["seed"] = seed
    args["action_mode"] = action_mode
    return args


def stack_obs(obs_list):
    return np.stack(obs_list, axis=0).astype(np.float32)


def evaluate_policy(env_args: dict, maddpg: MADDPG, episodes: int, device: torch.device, base_seed: int):
    eval_results = []
    for ep in range(episodes):
        eval_env = EdgeMARLEnv(**{**env_args, "seed": base_seed + ep})
        obs = stack_obs(eval_env.get_obs())
        done = False
        while not done:
            actions = maddpg.select_actions(obs, deterministic=True)
            reward, done, _ = eval_env.step(list(actions))
            obs = stack_obs(eval_env.get_obs())
        eval_results.append(eval_env.get_episode_summary())

    keys = eval_results[0].keys()
    aggregated = {k: float(np.mean([res[k] for res in eval_results])) for k in keys}
    return aggregated


def select_device(device_arg: str, force_cpu: bool):
    if force_cpu:
        device = torch.device("cpu")
        print("[INFO] 强制使用 CPU 训练。")
        return device

    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[INFO] Auto 模式：检测到 GPU，可用设备 {torch.cuda.get_device_name(0)}。")
            return device
        else:
            print("[WARN] 未检测到可用 GPU，退回 CPU。")
            return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("已指定 --device cuda，但当前环境未检测到 GPU。请检查 CUDA/驱动。")
        print(f"[INFO] 使用 GPU：{torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    # device_arg == "cpu"
    print("[INFO] 已指定 CPU 训练。")
    return torch.device("cpu")


def run_training(args):
    device = select_device(args.device, args.cpu)
    env_config = load_yaml(Path("src/config/envs/edge_marl.yaml"))
    alg_config = load_yaml(Path("src/config/algs/edge_maddpg.yaml"))

    env_args = build_env_args(env_config, seed=args.seed)
    env = EdgeMARLEnv(**env_args)
    obs = stack_obs(env.get_obs())
    state = env.get_state().astype(np.float32)

    maddpg_cfg = MADDPGConfig(
        n_agents=env.n_agents,
        obs_dim=env.get_obs_size(),
        action_dim=env.get_action_dim(),
        state_dim=env.get_state_size(),
        actor_lr=alg_config["actor_lr"],
        critic_lr=alg_config["critic_lr"],
        gamma=alg_config["gamma"],
        tau=alg_config["tau"],
        hidden_dim=alg_config["hidden_dim"],
        device=device,
    )

    maddpg = MADDPG(maddpg_cfg)
    buffer = ReplayBuffer(
        capacity=alg_config["buffer_size"],
        n_agents=env.n_agents,
        obs_dim=env.get_obs_size(),
        action_dim=env.get_action_dim(),
        state_dim=env.get_state_size(),
        device=device,
    )
    noise = OrnsteinUhlenbeckNoise(
        size=(env.n_agents, env.get_action_dim()),
        theta=alg_config["noise_theta"],
        sigma=alg_config["noise_sigma"],
        dt=alg_config.get("noise_dt", 1.0),
    )

    total_steps = args.t_max or alg_config["total_timesteps"]
    eval_interval = alg_config["eval_interval"]
    eval_episodes = alg_config["eval_episodes"]
    batch_size = alg_config["batch_size"]
    warmup_steps = alg_config["warmup_steps"]
    update_iters = alg_config.get("update_iters", 1)

    results_dir = Path(args.output_dir) / f"seed_{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    training_log = []
    eval_log = []

    episode_reward = 0.0
    episode_count = 0
    noise.reset()

    for step in range(1, total_steps + 1):
        action_noise = noise.sample()
        actions = maddpg.select_actions(obs, noise=action_noise, deterministic=False)
        reward, done, info = env.step(list(actions))
        next_obs = stack_obs(env.get_obs())
        next_state = env.get_state().astype(np.float32)

        buffer.add(obs, state, actions, reward, next_obs, next_state, done)

        if buffer.size >= batch_size and step > warmup_steps:
            batch = buffer.sample(batch_size)
            maddpg.update(batch, update_iters)

        obs = next_obs
        state = next_state
        episode_reward += reward

        if done:
            episode_count += 1
            summary = env.get_episode_summary()
            training_log.append(
                {
                    "step": step,
                    "episode": episode_count,
                    "reward": episode_reward,
                    **summary,
                }
            )
            env_args = build_env_args(env_config, seed=args.seed + episode_count)
            env = EdgeMARLEnv(**env_args)
            obs = stack_obs(env.get_obs())
            state = env.get_state().astype(np.float32)
            episode_reward = 0.0
            noise.reset()

        if step % eval_interval == 0:
            metrics = evaluate_policy(build_env_args(env_config, seed=args.seed + 1000), maddpg, eval_episodes, device, base_seed=args.seed + 2000)
            metrics.update({"step": step})
            eval_log.append(metrics)
            print(f"[Eval] Step {step}: Avg Latency={metrics['avg_latency_ms']:.2f} ms, Throughput={metrics['throughput_tps']:.2f}")

    # Save logs
    with (results_dir / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)
    with (results_dir / "eval_log.json").open("w", encoding="utf-8") as f:
        json.dump(eval_log, f, indent=2)

    print(f"[DONE] MADDPG training finished. Logs saved to {results_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="EdgeSimGym MADDPG trainer (PyMARL)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--t-max", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (override device)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection: auto (default), cpu, or cuda.",
    )
    parser.add_argument("--output-dir", type=str, default="results/edge_maddpg", help="Directory to save logs")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())

