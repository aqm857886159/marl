# --- 0. 实验控制 ---
# 设置为 True: 立即使用模拟数据生成所有图表（用于论文写作）
# 设置为 False: 运行完整的 5M 步 Ray RLlib 训练（用于获取真实数据）
USE_MOCK_DATA = False

# --- 1. 导入库 ---
import os
import math
import gymnasium
import numpy as np
import pandas as pd
import random
import json
import textwrap
from gymnasium.spaces import Box, Discrete, Dict

import ray
try:
    import torch
except ImportError:
    torch = None
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils import deprecation as rllib_deprecation
from ray.tune.registry import register_env

# RLlib 兼容补丁：允许旧版算法在缺失 rllib_contrib 包时继续运行
def _suppress_rllib_contrib_errors():
    if getattr(rllib_deprecation, "_marl_patched", False):
        return

    legacy_paths = {
        "rllib/algorithms/ddpg/",
        "rllib/algorithms/qmix/",
    }
    original = rllib_deprecation.deprecation_warning

    def patched_deprecation_warning(*args, **kwargs):
        old = kwargs.get("old")
        if old is None and args:
            old = args[0]
        if isinstance(old, str) and old in legacy_paths and kwargs.get("error"):
            kwargs = dict(kwargs)
            kwargs["error"] = False
            msg = f"[WARN] RLlib legacy algorithm '{old}' fallback启用（未检测到 rllib_contrib 包）"
            print(msg)
        return original(*args, **kwargs)

    rllib_deprecation.deprecation_warning = patched_deprecation_warning
    rllib_deprecation._marl_patched = True


_suppress_rllib_contrib_errors()

# 导入真正的 MAPPO 实现
try:
    from mappo_trainer import run_mappo_experiment
    MAPPO_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入 MAPPO，将使用 RLlib PPO 模拟。错误: {e}")
    MAPPO_AVAILABLE = False

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- 2. 仿真环境参数 (来自 3.2.1 节, 表 3.2) ---
ENV_CONFIG = {
    "n_nodes": 10,  # 边缘节点数 (N)
    "task_arrival_rate": 10,  # 兼容旧配置：当未设置区间时使用固定值
    "task_arrival_rate_range": (5, 15),  # 动态任务到达率区间 (λ_low, λ_high)
    "task_arrival_mode": "cyclic",  # 负载变化模式：cyclic / random
    "task_arrival_cycle_seconds": 20.0,  # cyclic 模式下的周期长度
    "episode_length": 1000,   # Episode 长度 (模拟 100 秒)

    # 节点异构性 (1-3 GHz)
    "node_cpu_capacity": np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0]) * 1e9,  # Giga-cycles/s

    # 任务特征 (Giga-cycles, MB, ms)
    "task_workload_range": (1.0, 10.0), # Giga-cycles
    "task_data_range": (0.5, 5.0),      # MB
    "task_deadline_range": (0.05, 0.5), # 50-500 ms

    # 网络模拟 (10-100 Mbps, 2-10 ms)
    "network_bw_range": (10, 100),      # Mbps
    "network_latency_range": (0.002, 0.01), # s

    # 奖励权重 (来自 3.2.3 节, B部分)
    "reward_weights": {
        "alpha": 0.5,  # 延迟
        "beta": 0.3,   # 能耗
        "gamma": 0.2   # 违约
    },

    # 连续时间推进控制
    "min_time_step": 0.02,
    "max_time_step": 0.2,
    "initial_observation_window": 0.1,

    # 探索调度 (ε-greedy)
    "exploration_schedule": {
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "decay_steps": 1_000_000
    },

    "disable_env_exploration": False
}

# --- 3. 自定义仿真环境 (EdgeSimGym) ---

class EdgeSimGym(MultiAgentEnv):
    """
    EdgeSimGym 仿真环境 (3.2.1 节)

    (已更新) 遵循 RLlib MultiAgentEnv 接口，
    模拟 POSG 问题 (2.1 节) 并在 info 字典中返回详细的评估指标 (3.2.3 节)。
    """

    _global_step_counter = 0

    def __init__(self, config=ENV_CONFIG):
        super().__init__()

        # 1. 应用环境参数
        self.n_nodes = config["n_nodes"]
        self.task_rate = config.get("task_arrival_rate", 10)
        self.task_rate_range = config.get("task_arrival_rate_range", (self.task_rate, self.task_rate))
        self.task_rate_mode = config.get("task_arrival_mode", "cyclic")
        self.task_rate_cycle = config.get("task_arrival_cycle_seconds", 20.0)
        self.episode_len = config["episode_length"]
        self.cpu_caps = config["node_cpu_capacity"]
        self.task_wl_range = config["task_workload_range"]
        self.task_data_range = config["task_data_range"]
        self.task_deadline_range = config["task_deadline_range"]
        self.bw_range = config["network_bw_range"]
        self.lat_range = config["network_latency_range"]
        self.weights = config["reward_weights"]
        self.min_time_step = config.get("min_time_step", 0.02)
        self.max_time_step = config.get("max_time_step", 0.2)
        self.initial_obs_window = config.get("initial_observation_window", 0.1)
        self.exploration_cfg = config.get("exploration_schedule", {
            "initial_epsilon": 1.0,
            "final_epsilon": 0.05,
            "decay_steps": 1_000_000
        })

        self._agent_ids = set(f"node_{i}" for i in range(self.n_nodes))

        # 2. 定义观测空间 (3.1.2.3.A)
        obs_dim = 1 + 1 + 3 + (self.n_nodes - 1)
        self.observation_space = Dict({
            agent_id: Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent_id in self._agent_ids
        })

        # 3. 定义动作空间 (3.1.2.3.B)
        self.action_space = Dict({
            agent_id: Dict({
                "placement": Discrete(self.n_nodes),
                "resource": Box(low=0.1, high=1.0, shape=(1,), dtype=np.float32)
            })
            for agent_id in self._agent_ids
        })

        # 内部状态
        self.nodes_load = np.zeros(self.n_nodes)
        self.nodes_queue_len = np.zeros(self.n_nodes)
        self.tasks_to_dispatch = {}
        self.current_step = 0
        self.sim_time = 0.0
        self.elapsed_time = 0.0
        self.last_step_duration = self.initial_obs_window
        self.latest_arrival_rate = self.task_rate

        # exploration epsilon (shared across workers)
        self.disable_env_exploration = config.get("disable_env_exploration", False)

        # === 新增：用于度量指标 (3.2.3) ===
        self.episode_latencies = []
        self.episode_energies = []
        self.episode_violations = 0
        self.episode_completed_tasks = 0
        self.episode_cpu_loads = [] # 用于Jain's Index

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.nodes_load = np.zeros(self.n_nodes)
        self.nodes_queue_len = np.zeros(self.n_nodes)
        self.tasks_to_dispatch = {}
        self.current_step = 0
        self.sim_time = 0.0
        self.elapsed_time = 0.0
        self.last_step_duration = self.initial_obs_window
        self.latest_arrival_rate = self.task_rate

        # 重置度量
        self.episode_latencies = []
        self.episode_energies = []
        self.episode_violations = 0
        self.episode_completed_tasks = 0
        self.episode_cpu_loads = []

        obs, _, _ = self._generate_tasks_and_obs(self.sim_time, advance_time=False)
        info = {agent_id: {} for agent_id in obs}

        return obs, info

    def _generate_task(self, arrival_time):
        workload = np.random.uniform(*self.task_wl_range) # Giga-cycles
        data = np.random.uniform(*self.task_data_range)      # MB
        deadline = arrival_time + np.random.uniform(*self.task_deadline_range) # 绝对截止时间 (s)
        return {"workload": workload, "data": data, "deadline": deadline, "arrival": arrival_time}

    def _current_task_rate(self, current_time):
        low, high = self.task_rate_range
        if low == high:
            return low

        if self.task_rate_mode == "cyclic":
            cycle = max(self.task_rate_cycle, 1e-6)
            phase = (current_time % cycle) / cycle
            # 使用正弦波在 [low, high] 间平滑过渡
            return low + (0.5 + 0.5 * np.sin(2 * np.pi * phase)) * (high - low)

        if self.task_rate_mode == "random":
            return np.random.uniform(low, high)

        # 默认退化为平均值
        return 0.5 * (low + high)

    def _sample_time_window(self, arrival_rate):
        """
        根据当前到达率采样下一段时间窗口长度。

        为了与 PyMARL 中的 EdgeMARLEnv 保持一致，这里不再直接使用期望间隔 1/λ，
        而是使用指数分布 Exponential(1/λ) 采样，将任务到达过程建模为真正的 Poisson 过程。
        """
        rate = max(arrival_rate, 1e-6)
        expected = 1.0 / rate
        dt = np.random.exponential(expected)
        return float(np.clip(dt, self.min_time_step, self.max_time_step))

    def _get_current_epsilon(self):
        if self.disable_env_exploration:
            return 0.0
        init_eps = self.exploration_cfg.get("initial_epsilon", 1.0)
        final_eps = self.exploration_cfg.get("final_epsilon", 0.05)
        decay_steps = max(1, self.exploration_cfg.get("decay_steps", 1_000_000))
        progress = min(1.0, EdgeSimGym._global_step_counter / decay_steps)
        return init_eps - (init_eps - final_eps) * progress

    def _maybe_apply_env_exploration(self, action):
        """
        在环境层面实现 ε-greedy，以保证不同算法共享同样的探索调度。
        仅对含混合动作空间的算法生效；QMIX/IPPO 等离散算法在上层采用相同调度。
        """
        if self.disable_env_exploration:
            return action

        epsilon = self._get_current_epsilon()
        if np.random.rand() > epsilon:
            return action

        random_action = {
            "placement": np.random.randint(0, self.n_nodes),
            "resource": np.array([np.random.uniform(0.1, 1.0)], dtype=np.float32)
        }
        return random_action

    def _convert_action(self, action):
        """
        将来自连续策略的动作向量转换为环境所需的 Dict 形式。
        """
        if isinstance(action, dict):
            return action

        vec = np.asarray(action, dtype=np.float32)
        if vec.ndim == 0:
            vec = vec.reshape(1)
        target_dim = self.n_nodes + 1
        if vec.size < target_dim:
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[:vec.size] = vec
            vec = padded

        placement_logits = vec[:-1]
        if np.allclose(placement_logits, 0):
            placement_idx = 0
        else:
            placement_idx = int(np.argmax(placement_logits)) % self.n_nodes

        resource_raw = float(vec[-1])
        resource_norm = (resource_raw + 1.0) / 2.0
        resource_value = 0.1 + 0.9 * np.clip(resource_norm, 0.0, 1.0)

        return {
            "placement": placement_idx,
            "resource": np.array([resource_value], dtype=np.float32)
        }

    def _get_obs(self, agent_id, task, current_time):
        node_idx = int(agent_id.split('_')[1])
        local_load_perc = self.nodes_load[node_idx] / self.cpu_caps[node_idx]
        local_queue_len = self.nodes_queue_len[node_idx]
        task_data = task["data"]
        task_workload = task["workload"]
        task_deadline_remaining = max(0, task["deadline"] - current_time)

        neighbor_loads = []
        for i in range(self.n_nodes):
            if i != node_idx:
                neighbor_loads.append(self.nodes_load[i] / self.cpu_caps[i])

        obs = np.array(
            [local_load_perc, local_queue_len, task_data, task_workload, task_deadline_remaining] + neighbor_loads,
            dtype=np.float32
        )
        return obs

    def _calculate_reward(self, latencies, energies, violations):
        if not latencies:
            return 0.0

        w_lat = self.weights["alpha"]
        w_eng = self.weights["beta"]
        w_vio = self.weights["gamma"]

        cost = (w_lat * np.mean(latencies)) + \
               (w_eng * np.mean(energies)) + \
               (w_vio * np.sum(violations))

        return -cost

    def _generate_tasks_and_obs(self, current_time, advance_time=True):
        self.tasks_to_dispatch = {}
        new_obs = {}

        arrival_rate = self._current_task_rate(current_time)
        if advance_time:
            time_window = self._sample_time_window(arrival_rate)
        else:
            time_window = self.initial_obs_window

        expected_tasks = max(arrival_rate, 1e-6) * time_window
        num_new_tasks = np.random.poisson(expected_tasks)

        for _ in range(num_new_tasks):
            task = self._generate_task(current_time)
            decider_idx = np.random.randint(0, self.n_nodes)
            agent_id = f"node_{decider_idx}"

            if agent_id in self.tasks_to_dispatch:
                self.nodes_queue_len[decider_idx] += 1
            else:
                self.tasks_to_dispatch[agent_id] = task
                new_obs[agent_id] = self._get_obs(agent_id, task, current_time)

        delta_t = time_window if advance_time else 0.0

        # 模拟负载衰减
        decay_window = time_window if advance_time else 0.0
        processed_load = self.cpu_caps * decay_window
        self.nodes_load -= processed_load
        self.nodes_load = np.maximum(0, self.nodes_load)

        # 模拟队列减少
        if advance_time:
            self.nodes_queue_len -= 1
            self.nodes_queue_len = np.maximum(0, self.nodes_queue_len)

        # === 新增：记录节点负载 (3.2.3.A) ===
        # 记录每一步的实际 CPU 利用率
        norm_caps = self.cpu_caps + 1e-9
        current_cpu_usage = np.where(norm_caps > 0, processed_load / norm_caps, 0.0)
        self.episode_cpu_loads.append(current_cpu_usage)

        return new_obs, delta_t, arrival_rate


    def step(self, action_dict):
        self.current_step += 1
        current_time = self.sim_time

        step_latencies = []
        step_energies = []
        step_violations = []
        step_completed_tasks = 0

        infos = {}  # 稍后只为实际返回的 obs keys 填充

        for agent_id, action in action_dict.items():
            if agent_id not in self.tasks_to_dispatch:
                continue

            task = self.tasks_to_dispatch[agent_id]
            src_node_idx = int(agent_id.split('_')[1])

            processed_action = self._convert_action(action)
            adjusted_action = self._maybe_apply_env_exploration(processed_action)
            target_node_idx = adjusted_action["placement"]
            resource_alloc = adjusted_action["resource"][0]

            if target_node_idx == src_node_idx:
                t_trans = 0.0
            else:
                bw = np.random.uniform(*self.bw_range) * 1e6 / 8
                lat = np.random.uniform(*self.lat_range)
                t_trans = (task["data"] / bw) + lat

            t_queue = self.nodes_load[target_node_idx] / self.cpu_caps[target_node_idx]

            allocated_cpu = self.cpu_caps[target_node_idx] * resource_alloc
            t_exec = task["workload"] / (allocated_cpu / 1e9)

            # P_exec ≈ k*f^2, E = P * t = (k*f^2) * (W/f) = k*W*f
            # 简化 k=1，能量单位为 Giga-Joules (假设)
            e_comp = (task["workload"] * (allocated_cpu / 1e9))

            t_total = t_trans + t_queue + t_exec
            e_total = e_comp

            step_latencies.append(t_total)
            step_energies.append(e_total)

            if (current_time + t_total) > task["deadline"]:
                step_violations.append(1)
            else:
                step_violations.append(0)
                step_completed_tasks += 1

            self.nodes_load[target_node_idx] += t_exec * (allocated_cpu / 1e9)
            self.nodes_queue_len[target_node_idx] += 1

        # 5. 生成新任务和观测
        new_obs, delta_t, arrival_rate = self._generate_tasks_and_obs(current_time, advance_time=True)
        rewards = {}
        dones = {"__all__": False}
        if delta_t > 0:
            self.sim_time += delta_t
            self.elapsed_time += delta_t
            self.last_step_duration = delta_t

        self.latest_arrival_rate = arrival_rate

        # 6. 计算全局奖励
        global_reward = self._calculate_reward(step_latencies, step_energies, step_violations)

        for agent_id in new_obs:
            rewards[agent_id] = global_reward
            infos[agent_id] = {}  # 确保 infos 的 keys 与 new_obs 一致

        # 7. 检查 Episode 是否结束
        terminated = dones["__all__"] = self.current_step >= self.episode_len
        truncated = {"__all__": False}

        # === 新增：累积 Episode 指标 ===
        self.episode_latencies.extend(step_latencies)
        self.episode_energies.extend(step_energies)
        self.episode_violations += sum(step_violations)
        self.episode_completed_tasks += step_completed_tasks
        if not self.disable_env_exploration:
            EdgeSimGym._global_step_counter += 1

        if terminated:
            # Episode 结束时，计算并返回所有评估指标 (3.2.3.A)
            # RLlib 会自动平均这些值并放入 result["custom_metrics"]
            total_tasks = len(self.episode_latencies)
            if total_tasks == 0: total_tasks = 1 # 避免除零

            if self.episode_latencies:
                latency_array = np.array(self.episode_latencies) * 1000
                avg_latency = float(np.mean(latency_array))
                p99_latency = float(np.percentile(latency_array, 99))
            else:
                avg_latency = 0.0
                p99_latency = 0.0
            avg_energy = np.mean(self.episode_energies) # G-Joules (假设)
            violation_rate = self.episode_violations / total_tasks
            total_time = max(self.elapsed_time, 1e-6)
            throughput = self.episode_completed_tasks / total_time # task/s

            # 计算 Jain's Index (3.2.3.A)
            if self.episode_cpu_loads:
                avg_cpu_loads_per_node = np.mean(np.array(self.episode_cpu_loads), axis=0)
            else:
                avg_cpu_loads_per_node = np.zeros(self.n_nodes)
            sum_loads = np.sum(avg_cpu_loads_per_node)
            sum_sq_loads = np.sum(avg_cpu_loads_per_node ** 2)
            jain_index = (sum_loads ** 2) / (self.n_nodes * sum_sq_loads + 1e-9)

            metrics = {
                "avg_latency_ms": avg_latency,
                "p99_latency_ms": p99_latency,
                "avg_energy_J": avg_energy,
                "violation_rate": violation_rate,
                "throughput_tps": throughput,
                "load_balance_jain": jain_index
            }
            # 将指标放入 info 字典（优先针对当前存在的 agents；若当前无观测，则至少提供一个占位 agent）
            target_agents = list(new_obs.keys())
            if target_agents:
                for agent_id in target_agents:
                    agent_info = infos.setdefault(agent_id, {})
                    agent_info["episode_metrics"] = metrics.copy()
                    agent_info["custom_metrics"] = metrics.copy()
                    for key, value in metrics.items():
                        agent_info[key] = value
            else:
                common_info = infos.setdefault("__common__", {})
                common_info["episode_metrics"] = metrics.copy()
                common_info["custom_metrics"] = metrics.copy()
                for key, value in metrics.items():
                    common_info[key] = value

        return new_obs, rewards, dones, truncated, infos

class EdgeMetricCallback(DefaultCallbacks):
    """
    自定义回调：用于在 Episode 结束时提取 EdgeSimGym 的自定义指标 (latency, energy, etc.)
    并将其注册到 RLlib 的系统指标中，以便自动评估 (Automatic Evaluation) 能正确记录。
    """
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # 尝试从 info 中提取 custom_metrics 并报告给 RLlib
        metrics_found = False
        
        # 兼容不同版本的 RLlib Episode 对象
        agents = []
        if hasattr(episode, "get_agents"):
            agents = episode.get_agents()
        elif hasattr(episode, "agent_rewards"):
            agents = episode.agent_rewards.keys()

        for agent_id in agents:
            last_info = None
            # 尝试多种方式获取 info
            if hasattr(episode, "last_info_for"):
                last_info = episode.last_info_for(agent_id)
            elif hasattr(episode, "_agent_to_last_info"): # 针对 EpisodeV2 的内部属性回退
                last_info = episode._agent_to_last_info.get(agent_id)
            
            if last_info and "custom_metrics" in last_info:
                for k, v in last_info["custom_metrics"].items():
                    # 将指标写入 episode.custom_metrics
                    # RLlib 会自动处理聚合 (mean, min, max)
                    episode.custom_metrics[k] = v
                metrics_found = True
                break 
        
        if not metrics_found:
            # 极少数情况下（例如所有 agent 同时 done 且 new_obs 为空），metrics 可能在 __common__ 里
            # 或者作为 fallback，我们直接去环境实例里拿（仅限 Local Worker 或单线程 Evaluation Worker）
            try:
                env = base_env.get_sub_environments()[env_index]
                if hasattr(env, "episode_latencies") and env.episode_latencies:
                    # 简单的保底计算
                    latency_arr = np.array(env.episode_latencies) * 1000
                    episode.custom_metrics["avg_latency_ms"] = float(np.mean(latency_arr))
                    episode.custom_metrics["p99_latency_ms"] = float(np.percentile(latency_arr, 99))
                    episode.custom_metrics["avg_energy_J"] = float(np.mean(env.episode_energies))
                    # 其他指标略，保住核心指标即可
            except Exception:
                pass


# --- 4. 主训练流程 (RQ1 实验) ---

def run_experiment(alg_name, seed, num_iterations=None, max_timesteps=5_000_000, eval_interval_steps=50_000):
    """
    运行单个算法和种子的训练实验 (3.2.2.B)
    (已更新) 返回训练和评估结果的 log
    """
    # 特殊处理：MAPPO 使用独立实现
    if alg_name == "MAPPO" and MAPPO_AVAILABLE:
        return run_mappo_experiment(seed, num_iterations, max_total_steps=max_timesteps)
    
    print(f"\n--- 启动训练: 算法={alg_name}, 种子={seed} ---")

    def _normalize_eval_metrics(metric_dict):
        normalized = {}
        for metric_name, metric_val in metric_dict.items():
            if isinstance(metric_val, dict):
                normalized[metric_name] = metric_val.get("mean")
            elif metric_name.endswith("_mean"):
                normalized[metric_name[:-5]] = metric_val
            else:
                normalized[metric_name] = metric_val
        return normalized

    discrete_only = alg_name in ["QMIX", "Greedy"]
    continuous_only = (alg_name == "MADDPG")
    if alg_name in ["MADDPG", "QMIX"]:
        raise NotImplementedError(
            f"{alg_name} 需要在 PyMARL 环境中运行。请参阅 `pymarl_configs/README.md` 获取配置说明。"
        )

    metric_keys = ["avg_latency_ms", "p99_latency_ms", "avg_energy_J", "throughput_tps", "load_balance_jain"]

    base_env_config = {
        **ENV_CONFIG,
        "disable_env_exploration": False
    }
    register_env("EdgeSimGym-v0", lambda config: EdgeSimGym({**base_env_config, **config}))
    dummy_env = EdgeSimGym(base_env_config)

    if alg_name == "IPPO":
        # IPPO (Independent PPO) - 独立学习
        # 每个 agent 有独立的 actor 和 critic
        config = PPOConfig()
        config = config.training(
            sgd_minibatch_size=64, 
            num_sgd_iter=4
        )
        # IPPO：每个 agent 独立策略（无参数共享）
        policies = {}
        for i in range(dummy_env.n_nodes):
            agent_id = f"node_{i}"
            policies[agent_id] = (
                None,
                dummy_env.observation_space[agent_id],
                dummy_env.action_space[agent_id],
                {}
            )

        def ippo_mapping(agent_id, *args, **kwargs):
            return agent_id

        config = config.multi_agent(
            policies=policies,
            policy_mapping_fn=ippo_mapping,
            policies_to_train=list(policies.keys())
        )
        config = config.environment("EdgeSimGym-v0", env_config={})
        
    elif alg_name == "MAPPO":
        # MAPPO (Multi-Agent PPO) - 中心化训练分布式执行
        # 使用 PPO 配置，但通过共享价值网络模拟中心化 Critic
        config = PPOConfig()
        config = config.training(
            sgd_minibatch_size=64, 
            num_sgd_iter=4,
            # MAPPO 关键：启用 GAE 和更大的 batch
            use_gae=True,
            lambda_=0.95,
            vf_loss_coeff=0.5,  # 更重视价值函数学习
            entropy_coeff=0.01  # 鼓励探索
        )
        # MAPPO 的关键区别：通过 policy sharing 实现参数共享
        config = config.multi_agent(
            policies={"shared_policy"},  # 所有 agent 共享一个策略
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        )
        config = config.environment("EdgeSimGym-v0", env_config={})
    else:
        raise ValueError(f"未知算法: {alg_name}")

    # 应用公平对比超参数 (3.2.2.A)
    num_gpus = 1 if (torch and torch.cuda.is_available()) else 0
    config = config.resources(num_gpus=num_gpus)
    config = config.framework("torch")
    train_batch_size = 4096
    config = config.training(
        gamma=0.99,
        lr=3e-4,
        train_batch_size=train_batch_size,
        model={"fcnet_hiddens": [64, 64]}
    )

    # 应用评估设置 (3.2.2.B)
    # 计算 evaluation_interval (以 iteration 为单位)
    # eval_interval_steps (e.g. 50000) / train_batch_size (4096) ~= 13 iterations
    eval_interval_iters = math.ceil(eval_interval_steps / train_batch_size)
    
    config = config.callbacks(EdgeMetricCallback)  # <--- 关键：挂载回调
    config = config.evaluation(
        evaluation_interval=eval_interval_iters, # 启用自动评估
        evaluation_duration=10,                  # 每次评估 10 个 Episodes
        evaluation_duration_unit="episodes",
        evaluation_num_workers=1,                # 使用 1 个 Worker 并行评估 (快速!)
        evaluation_parallel_to_training=False,   # 串行执行，确保结果准确写入
        evaluation_config={
            "explore": False,                    # 关闭探索
            "env_config": {"disable_env_exploration": True}
        },
        always_attach_evaluation_results=True    # 强制返回结果
    )

    rollout_fragment = 1024
    num_rollout_workers = 2
    config = config.rollouts(
        num_rollout_workers=num_rollout_workers,
        rollout_fragment_length=rollout_fragment
    )
    config = config.debugging(seed=seed)

    stop_criteria = {"timesteps_total": max_timesteps}
    
    algo = config.build()

    results_log = []

    expected_iters = math.ceil(max_timesteps / max(train_batch_size, 1))
    iteration_cap = num_iterations
    if iteration_cap is not None and iteration_cap < expected_iters:
        print(
            f"[WARN] 提供的 num_iterations={iteration_cap} 无法覆盖 {max_timesteps:,} 步，"
            f"按 train_batch_size={train_batch_size} 估算至少需要 {expected_iters} 次迭代。"
            f" 将自动提升迭代上限为 {expected_iters}。"
        )
        iteration_cap = expected_iters

    cap_display = iteration_cap if iteration_cap is not None else "∞"
    # 训练以「总交互步数」为主，迭代轮数仅作安全上限
    print(f"开始训练，目标 {max_timesteps:,} 环境步 (迭代上限 {cap_display})...")
    i = 0
    while True:
        result = algo.train()
        i += 1

        # 收集训练曲线数据 (3.3.1)
        log_entry = {
            "algorithm": alg_name,
            "seed": seed,
            "timestep": result["timesteps_total"],
            "episode_return_mean": result["episode_reward_mean"],
        }

        # 从 RLlib 自动评估结果中提取指标
        # 只要 evaluation_interval 到了，result["evaluation"] 就会有数据
        eval_attachment = {}
        if "evaluation" in result and "custom_metrics" in result["evaluation"]:
            eval_metrics = result["evaluation"]["custom_metrics"]
            # RLlib 通常会返回 key_mean, key_min, key_max
            # 我们需要提取 key_mean 并还原为原始 key
            for key in metric_keys:
                mean_key = f"{key}_mean"
                if mean_key in eval_metrics:
                    eval_attachment[key] = eval_metrics[mean_key]
                elif key in eval_metrics: # 某些版本可能直接返回 key
                    eval_attachment[key] = eval_metrics[key]
        
        if eval_attachment:
            log_entry.update(eval_attachment)

        results_log.append(log_entry)

        eval_latency = eval_attachment.get("avg_latency_ms") if eval_attachment else None
        if isinstance(eval_latency, (int, float)):
            eval_display = f"{eval_latency:.2f}"
        else:
            eval_display = "N/A"
        print(
            f"Alg: {alg_name}, Seed: {seed}, Iter: {i}, "
            f"Steps: {log_entry['timestep']}, "
            f"Train Reward: {log_entry['episode_return_mean']:.2f}, "
            f"Eval Latency(ms): {eval_display}"
        )

        # 先按总步数停止，确保与论文 5M 步一致；num_iterations 仅作为保险
        if result["timesteps_total"] >= stop_criteria["timesteps_total"]:
            print(f"达到 {max_timesteps:,} 步，停止训练。")
            break
        if iteration_cap is not None and i >= iteration_cap:
            print(
                f"[WARN] 达到迭代上限 {iteration_cap}，当前步数 {result['timesteps_total']:,}，"
                "未能跑满目标步数。请提高 train_batch_size 或放宽 num_iterations。"
            )
            break

    algo.stop()
    print(f"--- 训练完成: 算法={alg_name}, 种子={seed} ---")
    return results_log


def run_greedy_baseline(seed, num_episodes=100):
    """
    运行贪心基线算法评估 (3.3.2 节)
    贪心策略：总是将任务分配到当前计算延迟最低的节点
    """
    print(f"\n--- 运行贪心基线: 种子={seed} ---")
    
    np.random.seed(seed)
    random.seed(seed)
    
    env_cfg = {**ENV_CONFIG, "disable_env_exploration": True}
    env = EdgeSimGym(env_cfg)
    
    episode_metrics = {
        "avg_latency_ms": [],
        "p99_latency_ms": [],
        "avg_energy_J": [],
        "throughput_tps": [],
        "load_balance_jain": []
    }
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        
        while not done:
            actions = {}
            # 贪心策略：选择当前负载最低的节点
            for agent_id in obs.keys():
                # 找到计算延迟最低的节点（负载最低）
                target_node = int(np.argmin(env.nodes_load / env.cpu_caps))
                actions[agent_id] = {
                    "placement": target_node,
                    "resource": np.array([0.5], dtype=np.float32)  # 固定50%资源分配
                }
            
            obs, rewards, dones, truncated, infos = env.step(actions)
            done = dones.get("__all__", False)
            
            # 收集 episode 结束时的指标
            if done and obs:
                for agent_id in obs.keys():
                    if "episode_metrics" in infos.get(agent_id, {}):
                        metrics = infos[agent_id]["episode_metrics"]
                        episode_metrics["avg_latency_ms"].append(metrics["avg_latency_ms"])
                        episode_metrics["p99_latency_ms"].append(metrics["p99_latency_ms"])
                        episode_metrics["avg_energy_J"].append(metrics["avg_energy_J"])
                        episode_metrics["throughput_tps"].append(metrics["throughput_tps"])
                        episode_metrics["load_balance_jain"].append(metrics["load_balance_jain"])
                        break  # 只需记录一次
    
    # 计算平均值和标准差
    results = {
        "algorithm": "Greedy",
        "seed": seed,
        "avg_latency_ms_mean": np.mean(episode_metrics["avg_latency_ms"]),
        "avg_latency_ms_std": np.std(episode_metrics["avg_latency_ms"]),
        "p99_latency_ms_mean": np.mean(episode_metrics["p99_latency_ms"]),
        "p99_latency_ms_std": np.std(episode_metrics["p99_latency_ms"]),
        "avg_energy_J_mean": np.mean(episode_metrics["avg_energy_J"]),
        "avg_energy_J_std": np.std(episode_metrics["avg_energy_J"]),
        "throughput_tps_mean": np.mean(episode_metrics["throughput_tps"]),
        "throughput_tps_std": np.std(episode_metrics["throughput_tps"]),
        "load_balance_jain_mean": np.mean(episode_metrics["load_balance_jain"]),
        "load_balance_jain_std": np.std(episode_metrics["load_balance_jain"])
    }
    
    print(f"贪心基线评估完成: 延迟={results['avg_latency_ms_mean']:.2f}ms")
    return results


# --- 5. 模拟数据生成 (用于论文图表示例) ---

def generate_mock_data():
    """
    生成与 3.3.1 和 3.3.2 节大纲描述一致的模拟数据。
    """
    print("正在生成模拟数据 (USE_MOCK_DATA=True)...")

    algorithms = ["MAPPO", "MADDPG", "QMIX", "IPPO"]
    seeds = [0, 1, 2, 3, 4]

    # 模拟“表 3.5”的均值和标准差
    # 注意：IPPO 在代码中是 PPO，这里我们用 "IPPO" 字符串
    alg_map = {"PPO": "IPPO", "DDPG": "MADDPG", "QMIX": "QMIX", "MAPPO": "MAPPO"}

    # [均值, 标准差]
    # 我们也加入贪心算法用于对比
    metrics_stats = {
        "MAPPO": {
            "avg_latency_ms": (72.1, 10.8), "avg_energy_J": (2.21, 0.28),
            "throughput_tps": (14.1, 1.1), "load_balance_jain": (0.86, 0.05),
            "p99_latency_ms": (95.0, 8.5),
            "convergence_steps": (2.5e6, 0.1e6)
        },
        "MADDPG": {
            "avg_latency_ms": (78.2, 15.7), "avg_energy_J": (2.38, 0.42),
            "throughput_tps": (13.2, 1.5), "load_balance_jain": (0.79, 0.08),
            "p99_latency_ms": (110.0, 12.0),
            "convergence_steps": (2.0e6, 0.2e6) # 3.3.1: 速度快，方差大
        },
        "QMIX": {
            "avg_latency_ms": (85.3, 12.4), "avg_energy_J": (2.45, 0.31),
            "throughput_tps": (12.5, 1.2), "load_balance_jain": (0.82, 0.06),
            "p99_latency_ms": (125.0, 10.5),
            "convergence_steps": (3.0e6, 0.1e6) # 3.3.1: 速度中，方差小
        },
        "IPPO": {
            "avg_latency_ms": (92.6, 18.3), "avg_energy_J": (2.67, 0.51),
            "throughput_tps": (11.3, 1.8), "load_balance_jain": (0.75, 0.1),
            "p99_latency_ms": (150.0, 15.0),
            "convergence_steps": (4.0e6, 0.3e6) # 3.3.1: 速度慢，性能低
        },
        "Greedy": {
            "avg_latency_ms": (135.4, 22.1), "avg_energy_J": (3.12, 0.58),
            "throughput_tps": (8.7, 1.4), "load_balance_jain": (0.62, 0.12),
            "p99_latency_ms": (210.0, 18.0),
        }
    }

    # 模拟“图 3.3”的学习曲线 (Episode Return)
    # 我们用负的“成本”作为奖励，并模拟收敛行为
    all_results = {"learning_curves": [], "performance_metrics": [], "convergence_speed": []}

    timesteps = np.arange(0, 5_000_001, 50000)

    def sigmoid_convergence(x, L, k, x0):
        # 模拟 S 型收敛曲线
        return L / (1 + np.exp(-k * (x - x0)))

    for alg in algorithms:
        # 匹配 3.3.1 的观察发现
        if alg == "MAPPO":
            L, k, x0 = -150, 2.5, 2.5e6 # 最终性能最高
            stability_noise = 3.0 # 稳定性高
        elif alg == "MADDPG":
            L, k, x0 = -160, 3.0, 2.0e6 # 速度快，性能次高
            stability_noise = 8.0 # 方差较大
        elif alg == "QMIX":
            L, k, x0 = -170, 2.0, 3.0e6 # 速度中，方差小
            stability_noise = 2.0
        else: # IPPO
            L, k, x0 = -250, 1.5, 4.0e6 # 速度慢，性能低
            stability_noise = 5.0

        for seed in seeds:
            base_curve = sigmoid_convergence(timesteps, L, k/1e6, x0)
            # 添加种子间的差异和每一步的噪声
            seed_offset = np.random.normal(0, stability_noise * 5)
            step_noise = np.random.normal(0, stability_noise, size=timesteps.shape)

            returns = base_curve + seed_offset + step_noise

            for t, r in zip(timesteps, returns):
                all_results["learning_curves"].append({
                    "algorithm": alg,
                    "seed": seed,
                    "timestep": t,
                    "episode_return_mean": r
                })

            # 模拟“图 3.6”的 100 个评估 episode 数据
            # (基于表 3.5 的均值和标准差)
            for metric, (mean, std) in metrics_stats[alg].items():
                if metric == "convergence_steps":
                    # 模拟“图 3.4”的收敛数据
                    all_results["convergence_speed"].append({
                        "algorithm": alg,
                        "seed": seed,
                        "steps": np.random.normal(mean, std)
                    })
                    continue

                # 生成 100 个数据点
                samples = np.random.normal(mean, std, size=100)
                for s in samples:
                    all_results["performance_metrics"].append({
                        "algorithm": alg,
                        "seed": seed,
                        "metric": metric,
                        "value": s
                    })

    # 添加贪心算法的性能数据 (图 3.6)
    for seed in seeds:
         for metric, (mean, std) in metrics_stats["Greedy"].items():
            if metric == "convergence_steps": continue
            samples = np.random.normal(mean, std, size=100)
            for s in samples:
                all_results["performance_metrics"].append({
                    "algorithm": "Greedy",
                    "seed": seed,
                    "metric": metric,
                    "value": s
                })

    print("模拟数据生成完毕。")
    return all_results


# --- 6. 绘图与统计分析 (3.3 节) ---

def setup_plotting():
    """设置 Matplotlib 以支持中文显示"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Microsoft JhengHei', 'Arial Unicode MS'] # 尝试使用常见的中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    print("Matplotlib 中文环境设置完毕。")


def plot_learning_curves(df, ax):
    """
    绘制 图 3.3：各算法训练曲线对比
    x轴：训练步数 (0-5M)
    y轴：Episode Return（多目标加权和）
    """
    print("Plotting Figure 3.3: Learning Curves...")
    sns.lineplot(
        data=df,
        x="timestep",
        y="episode_return_mean",
        hue="algorithm",
        style="algorithm",
        ax=ax,
        ci="sd" # 3.3.1: 显示置信区间 (标准差)
    )
    ax.set_title("Figure 3.3: Training Curves Comparison (RQ1 Screening)")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Episode Return")
    ax.legend(title="Algorithm")
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


def plot_convergence_speed(df, ax):
    """
    绘制 图 3.4：收敛速度对比（箱线图）
    """
    print("Plotting Figure 3.4: Convergence Speed...")
    sns.boxplot(
        data=df,
        x="algorithm",
        y="steps",
        ax=ax
    )
    ax.set_title("Figure 3.4: Convergence Speed Comparison (90% Optimum)")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Steps to Converge")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


def plot_performance_radar(df, ax):
    """
    绘制 图 3.5：多维度性能雷达图
    """
    print("Plotting Figure 3.5: Performance Radar...")

    # 1. 提取表 3.5 的平均值
    stats = df.groupby(["algorithm", "metric"])["value"].mean().unstack()

    # 2. 准备雷达图数据
    # 移除收敛速度，因为它不是性能指标
    stats = stats.drop(columns=["convergence_steps", "p99_latency_ms"], errors='ignore')

    # 我们需要规范化指标，因为它们的量纲不同
    # 目标：越高越好
    metrics = list(stats.columns)
    normalized_stats = stats.copy()

    for metric in metrics:
        min_val = stats[metric].min()
        max_val = stats[metric].max()
        if metric in ["avg_latency_ms", "avg_energy_J"]:
            # 成本指标：越低越好 -> 转换
            normalized_stats[metric] = 1 - (stats[metric] - min_val) / (max_val - min_val)
        else:
            # 收益指标：越高越好
            normalized_stats[metric] = (stats[metric] - min_val) / (max_val - min_val)

    # 雷达图绘制
    labels = normalized_stats.columns.to_list()
    num_vars = len(labels)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # 闭合

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)

    for alg_name, row in normalized_stats.iterrows():
        values = row.tolist()
        values += values[:1] # 闭合
        ax.plot(angles, values, label=alg_name, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.1)

    ax.set_title("Figure 3.5: Normalized Multi-Metric Performance Radar", pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))


def plot_performance_distribution(df, fig):
    """
    绘制 图 3.6：各算法性能分布（小提琴图）
    """
    print("Plotting Figure 3.6: Performance Distributions...")

    # 准备数据，移除收敛速度
    df_perf = df[df["metric"] != "convergence_steps"].copy()

    # 将指标名称翻译为中文（用于图表）
    metric_map = {
        "avg_latency_ms": "Avg Latency (ms)",
        "p99_latency_ms": "P99 Latency (ms)",
        "avg_energy_J": "Avg Energy (J)",
        "throughput_tps": "Throughput (task/s)",
        "load_balance_jain": "Load Balance (Jain)"
    }
    df_perf["metric_cn"] = df_perf["metric"].map(metric_map)

    # 使用 FacetGrid 或 catplot 创建 1x4 的子图
    g = sns.catplot(
        data=df_perf,
        x="algorithm",
        y="value",
        col="metric_cn",
        kind="violin", # 小提琴图
        sharey=False, # Y 轴不共享，因为量纲不同
        height=5,
        aspect=0.8,
        cut=0 # 限制在数据范围内
    )

    g.fig.suptitle("Figure 3.6: Performance Distributions across Metrics", y=1.05)
    g.set_axis_labels("Algorithm", "Metric Value")
    g.set_titles("{col_name}")


def calculate_statistics(df):
    """
    计算并打印 表 3.5 (性能) 和 表 3.6 (p值)
    """
    print("\n\n--- 3.3.2 节：性能指标对比 ---")

    # 1. 准备数据 (表 3.5)
    df_perf = df[df["metric"] != "convergence_steps"].copy()
    # 计算均值和标准差
    agg_stats = df_perf.groupby(["algorithm", "metric"])["value"].agg(['mean', 'std']).unstack()

    # 格式化输出 (模拟 表 3.5)
    print("\n表 3.5：各算法性能对比（均值 ± 标准差）")
    print("-" * 80)
    print(f"{'算法':<10} | {'平均延迟(ms)':<18} | {'P99 延迟(ms)':<18} | {'平均能耗(J)':<18} | {'吞吐量(task/s)':<18} | {'负载均衡':<15}")
    print("-" * 80)

    # 获取算法顺序 (MAPPO, MADDPG, QMIX, IPPO, Greedy)
    alg_order = ["MAPPO", "MADDPG", "QMIX", "IPPO", "Greedy"]

    for alg in alg_order:
        if alg not in agg_stats.index: continue
        row = f"{alg:<10} | "
        row += f"{agg_stats.loc[alg, ('mean', 'avg_latency_ms')]:.1f} ± {agg_stats.loc[alg, ('std', 'avg_latency_ms')]:.1f} | "
        row += f"{agg_stats.loc[alg, ('mean', 'p99_latency_ms')]:.1f} ± {agg_stats.loc[alg, ('std', 'p99_latency_ms')]:.1f} | "
        row += f"{agg_stats.loc[alg, ('mean', 'avg_energy_J')]:.2f} ± {agg_stats.loc[alg, ('std', 'avg_energy_J')]:.2f} | "
        row += f"{agg_stats.loc[alg, ('mean', 'throughput_tps')]:.1f} ± {agg_stats.loc[alg, ('std', 'throughput_tps')]:.1f} | "
        row += f"{agg_stats.loc[alg, ('mean', 'load_balance_jain')]:.2f} ± {agg_stats.loc[alg, ('std', 'load_balance_jain')]:.2f}"
        print(row)
    print("-" * 80)

    # 2. 统计显著性检验 (表 3.6)
    print("\n\n--- 3.3.3 节：统计显著性检验 (Welch's t-test on Latency) ---")

    # 提取所有算法（除 Greedy 外）的“平均延迟”数据
    df_latency = df_perf[df_perf["metric"] == "avg_latency_ms"]
    df_latency = df_latency[df_latency["algorithm"] != "Greedy"]

    algorithms = ["QMIX", "MADDPG", "MAPPO", "IPPO"] # 表 3.6 的顺序
    p_value_matrix = pd.DataFrame(index=algorithms, columns=algorithms, dtype=float)

    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            alg1 = algorithms[i]
            alg2 = algorithms[j]

            data1 = df_latency[df_latency["algorithm"] == alg1]["value"]
            data2 = df_latency[df_latency["algorithm"] == alg2]["value"]

            # Welch's t-test (不假设方差相等)
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)

            p_value_matrix.loc[alg1, alg2] = p_value
            p_value_matrix.loc[alg2, alg1] = p_value # 镜像

    print("\n表 3.6：算法间延迟差异的 p 值矩阵")
    print(p_value_matrix.to_string(float_format="%.4f"))
    print("* p<0.05, ** p<0.01")
    print("-" * 80)

    return agg_stats, p_value_matrix


def save_core_data(output_dir, df_curves, df_perf_metrics, df_convergence, agg_stats, p_value_matrix):
    """
    将关键实验数据保存到本地，便于复现与后续分析。
    """
    os.makedirs(output_dir, exist_ok=True)

    df_curves.to_csv(os.path.join(output_dir, "learning_curves.csv"), index=False)
    df_perf_metrics.to_csv(os.path.join(output_dir, "performance_metrics.csv"), index=False)
    df_convergence.to_csv(os.path.join(output_dir, "convergence_speed.csv"), index=False)

    # 展平多层列索引，便于阅读
    flat_cols = [f"{stat}_{metric}" for stat, metric in agg_stats.columns]
    agg_stats_flat = agg_stats.copy()
    agg_stats_flat.columns = flat_cols
    agg_stats_flat = agg_stats_flat.reset_index()
    agg_stats_flat.to_csv(os.path.join(output_dir, "table3_5_summary.csv"), index=False)

    p_value_matrix.to_csv(os.path.join(output_dir, "table3_6_latency_pvalues.csv"))


# --- 7. 主函数 ---

def main():
    """主执行函数"""

    all_results = {}

    if USE_MOCK_DATA:
        # 1. 使用模拟数据
        all_results = generate_mock_data()

    else:
        # 2. 运行真实实验
        if not ray.is_initialized():
            ray.init(num_cpus=4, num_gpus=1) # 根据您的 3.2.2.C 节调整

        # RLlib 阶段仅运行 MAPPO/IPPO（MADDPG/QMIX 已迁移至 PyMARL）
        algorithms_to_run = ["MAPPO", "IPPO"]
        random_seeds = [0, 1, 2, 3, 4]

        # 累积所有 (算法, 种子) 的训练与评估日志
        raw_results_log = []
        # 增量保存路径：即使中途停止，也能保留已完成种子的日志
        incremental_log_path = "ray_raw_results_log.csv"
        greedy_results = []
        all_results = {
            "learning_curves": [],
            "performance_metrics": [],
            "convergence_speed": []
        }

        try:
            # 训练 MARL 算法（支持增量保存）
            for alg in algorithms_to_run:
                for seed in random_seeds:
                    # 运行实验（默认按 5M 步跑满）
                    log = run_experiment(alg, seed, num_iterations=None)
                    raw_results_log.extend(log)

                    # 每完成一个 (算法, 种子) 就立即增量保存一次原始日志，防止中途终止导致数据丢失
                    try:
                        df_partial = pd.DataFrame(raw_results_log)
                        df_partial.to_csv(incremental_log_path, index=False)
                        print(f"[AUTO-SAVE] 已增量保存 raw log 至 {os.path.abspath(incremental_log_path)} "
                              f"(当前记录数: {len(df_partial)})")
                    except Exception as e:
                        print(f"[WARN] 增量保存 raw log 失败: {e}")
            
            # 运行贪心基线（不需要训练）
            print("\n=== 开始评估贪心基线 ===")
            for seed in random_seeds:
                greedy_result = run_greedy_baseline(seed, num_episodes=100)
                greedy_results.append(greedy_result)

            # 处理 MARL 算法的训练数据
            df_log = pd.DataFrame(raw_results_log)

            # 1. 学习曲线数据 (只有 MARL 算法有)
            df_curves = df_log.dropna(subset=["episode_return_mean"])

            # 2. 性能指标数据 (取最后一次评估)
            df_perf_marl = df_log.dropna(subset=["avg_latency_ms"]).drop_duplicates(subset=["algorithm", "seed"], keep="last")
            
            # 添加贪心基线的性能数据
            greedy_perf_rows = []
            for gr in greedy_results:
                greedy_perf_rows.append({
                    "algorithm": "Greedy",
                    "seed": gr["seed"],
                    "avg_latency_ms": gr["avg_latency_ms_mean"],
                    "p99_latency_ms": gr["p99_latency_ms_mean"],
                    "avg_energy_J": gr["avg_energy_J_mean"],
                    "throughput_tps": gr["throughput_tps_mean"],
                    "load_balance_jain": gr["load_balance_jain_mean"]
                })
            df_greedy_perf = pd.DataFrame(greedy_perf_rows)
            
            # 合并 MARL 和 Greedy 的性能数据
            df_perf_all = pd.concat([df_perf_marl, df_greedy_perf], ignore_index=True)
            
            # 转换为长格式用于绘图
            df_perf_long = pd.melt(
                df_perf_all,
                id_vars=["algorithm", "seed"],
                value_vars=["avg_latency_ms", "p99_latency_ms", "avg_energy_J", "throughput_tps", "load_balance_jain"],
                var_name="metric",
                value_name="value"
            )

            # 3. 收敛数据 (在真实模式下难以自动计算，留空)
            df_converge = pd.DataFrame(columns=["algorithm", "seed", "steps"])

            all_results = {
                "learning_curves": df_curves.to_dict('records'),
                "performance_metrics": df_perf_long.to_dict('records'),
                "convergence_speed": df_converge.to_dict('records')
            }

        except Exception as e:
            print(f"实验出错: {e}")
        finally:
            ray.shutdown()

    # --- 8. 处理数据并绘图 ---
    if not all_results["learning_curves"]:
        print("未找到可供绘图的数据。")
        return

    # 转换数据为 DataFrame
    df_curves = pd.DataFrame(all_results["learning_curves"])
    df_perf_metrics = pd.DataFrame(all_results["performance_metrics"])
    df_convergence = pd.DataFrame(all_results["convergence_speed"])

    output_dir = "experiment_outputs"

    # 设置中文显示
    setup_plotting()

    # --- 绘制图 3.3 和 3.4 ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # 图 3.3
    plot_learning_curves(df_curves, ax1)

    # 图 3.4
    if not df_convergence.empty:
        plot_convergence_speed(df_convergence, ax2)
    else:
        ax2.set_title("Figure 3.4: Convergence Speed (No Data)")

    fig1.tight_layout()
    fig1.savefig("figure_3.3_3.4_learning_curves.png", dpi=300)

    # --- 绘制图 3.5 (雷达图) ---
    fig2, ax3 = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    plot_performance_radar(df_perf_metrics, ax3)
    fig2.tight_layout()
    fig2.savefig("figure_3.5_radar_chart.png", dpi=300)

    # --- 绘制图 3.6 (小提琴图) ---
    # catplot 会创建自己的 Figure 对象
    plot_performance_distribution(df_perf_metrics, fig2)
    plt.savefig("figure_3.6_violin_plots.png", dpi=300)

    # --- 打印统计表格 (表 3.5, 3.6) ---
    agg_stats, p_value_matrix = calculate_statistics(df_perf_metrics)

    # --- 保存关键数据 ---
    save_core_data(
        output_dir=output_dir,
        df_curves=df_curves,
        df_perf_metrics=df_perf_metrics,
        df_convergence=df_convergence,
        agg_stats=agg_stats,
        p_value_matrix=p_value_matrix
    )
    print(f"核心数据已保存至: {os.path.abspath(output_dir)}")

    print("\n--- 所有图表和表格已生成 (模拟数据) ---")
    plt.show()


if __name__ == "__main__":
    main()