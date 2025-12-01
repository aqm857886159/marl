import math
from typing import Dict, List

import numpy as np

from .multiagentenv import MultiAgentEnv


class EdgeMARLEnv(MultiAgentEnv):
    """
    Edge computing task scheduling environment for PyMARL.

    This is a lightweight port of the EdgeSimGym environment used in the RLlib pipeline.
    Only the discrete action mode (placement-only) is exposed so that algorithms such as
    QMIX can operate without any modification to PyMARL's discrete action assumptions.
    """

    def __init__(self, **env_args):
        self.n_agents = env_args.get("n_nodes", 10)
        self.episode_limit = env_args.get("episode_length", 1000)
        self.node_cpu_capacity = np.asarray(
            env_args.get("node_cpu_capacity", np.ones(self.n_agents) * 1e9), dtype=np.float64
        )
        self.task_workload_range = env_args.get("task_workload_range", (1.0, 10.0))
        self.task_data_range = env_args.get("task_data_range", (0.5, 5.0))
        self.task_deadline_range = env_args.get("task_deadline_range", (0.05, 0.5))
        self.bw_range = env_args.get("network_bw_range", (10.0, 100.0))
        self.latency_range = env_args.get("network_latency_range", (0.002, 0.01))
        self.reward_weights = env_args.get(
            "reward_weights", {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
        )
        self.task_rate = env_args.get("task_arrival_rate", 10.0)
        self.task_rate_range = env_args.get("task_arrival_rate_range", (5.0, 15.0))
        self.task_rate_mode = env_args.get("task_arrival_mode", "cyclic")
        self.task_rate_cycle = env_args.get("task_arrival_cycle_seconds", 20.0)
        self.min_time_step = env_args.get("min_time_step", 0.02)
        self.max_time_step = env_args.get("max_time_step", 0.2)
        self.initial_obs_window = env_args.get("initial_observation_window", 0.1)
        self.action_mode = env_args.get("action_mode", "discrete")
        if self.action_mode not in {"discrete", "hybrid"}:
            raise ValueError("EdgeMARLEnv action_mode must be 'discrete' or 'hybrid'.")
        self.hybrid_action_dim = self.n_agents + 1  # placement logits + resource scalar

        self.random_seed = env_args.get("seed", None)
        self.rng = np.random.default_rng(self.random_seed)

        self.current_step = 0
        self.sim_time = 0.0
        self.nodes_load = np.zeros(self.n_agents, dtype=np.float64)
        self.nodes_queue = np.zeros(self.n_agents, dtype=np.float64)
        self.tasks_to_dispatch: Dict[int, Dict] = {}
        self.last_time_window = self.initial_obs_window
        self.episode_metrics = self._init_metrics()

        self.obs_dim = 5 + (self.n_agents - 1)
        self.state_dim = self.obs_dim * self.n_agents

        self.reset()

    def seed(self, seed: int):
        self.random_seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # PyMARL MultiAgentEnv interface
    # ------------------------------------------------------------------
    def reset(self):
        self.current_step = 0
        self.sim_time = 0.0
        self.nodes_load.fill(0.0)
        self.nodes_queue.fill(0.0)
        self.last_time_window = self.initial_obs_window
        self.episode_metrics = self._init_metrics()

        self.tasks_to_dispatch = {}
        for agent_idx in range(self.n_agents):
            self.tasks_to_dispatch[agent_idx] = self._sample_task(agent_idx)

    def step(self, actions: List[np.ndarray]):
        if len(actions) != self.n_agents:
            raise ValueError("Actions must be provided for each agent.")

        time_window = self._sample_time_window()
        self.sim_time += time_window
        self.last_time_window = time_window
        self.current_step += 1

        latencies = []
        energies = []
        violations = []
        completed = 0

        for agent_idx, raw_action in enumerate(actions):
            task = self.tasks_to_dispatch.get(agent_idx)
            if task is None:
                continue

            if self.action_mode == "discrete":
                if hasattr(raw_action, "cpu"):
                    raw_action = raw_action.cpu().numpy()
                target_node = int(np.clip(raw_action, 0, self.n_agents - 1))
                resource_share = 0.5
            else:
                target_node, resource_share = self._convert_hybrid_action(raw_action)

            placement_load = self.nodes_load[target_node]

            if target_node == agent_idx:
                transfer_latency = 0.0
            else:
                bw = self.rng.uniform(*self.bw_range) * 1e6 / 8.0  # MBps
                net_lat = self.rng.uniform(*self.latency_range)
                transfer_latency = (task["data"] / bw) + net_lat

            queue_latency = placement_load / (self.node_cpu_capacity[target_node] + 1e-9)
            effective_cpu = self.node_cpu_capacity[target_node] * resource_share
            exec_latency = task["workload"] / (effective_cpu / 1e9)

            total_latency = transfer_latency + queue_latency + exec_latency
            energy = task["workload"] * (effective_cpu / 1e9)

            latencies.append(total_latency)
            energies.append(energy)
            completed += 1
            violations.append(1 if (self.sim_time + total_latency) > task["deadline"] else 0)

            self.nodes_load[target_node] += task["workload"]
            self.nodes_queue[target_node] += 1

        self._advance_system(time_window)
        self._spawn_new_tasks()

        if completed == 0:
            avg_latency = 0.0
            avg_energy = 0.0
            violation_rate = 0.0
        else:
            avg_latency = float(np.mean(latencies))
            avg_energy = float(np.mean(energies))
            violation_rate = float(np.mean(violations))

        throughput = (completed / time_window) if time_window > 0 else 0.0
        load_balance = self._compute_jain_index(self.nodes_load + 1e-6)

        reward = self._compute_reward(avg_latency, avg_energy, violation_rate)
        terminated = self.current_step >= self.episode_limit

        self.episode_metrics["latency"].append(avg_latency)
        self.episode_metrics["energy"].append(avg_energy)
        self.episode_metrics["violations"].append(violation_rate)
        self.episode_metrics["throughput"].append(throughput)
        self.episode_metrics["load_balance"].append(load_balance)

        info = {
            "avg_latency_ms": avg_latency * 1000.0,
            "avg_energy_J": avg_energy,
            "deadline_violation_rate": violation_rate,
            "throughput_tps": throughput,
            "load_balance_jain": load_balance,
            "episode_limit": terminated and self.current_step >= self.episode_limit,
        }
        return reward, terminated, info

    def get_obs(self):
        return [self._build_obs(agent_idx) for agent_idx in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        return self._build_obs(agent_id)

    def get_obs_size(self):
        return self.obs_dim

    def get_state(self):
        return np.concatenate(self.get_obs(), axis=0)

    def get_state_size(self):
        return self.state_dim

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.get_total_actions()), dtype=np.int32)

    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.get_total_actions(), dtype=np.int32)

    def get_total_actions(self):
        return self.n_agents

    def get_action_dim(self):
        if self.action_mode == "discrete":
            return 1
        return self.hybrid_action_dim

    def get_episode_summary(self):
        latency_ms = np.array(self.episode_metrics["latency"], dtype=np.float32) * 1000.0
        energy = np.array(self.episode_metrics["energy"], dtype=np.float32)
        throughput = np.array(self.episode_metrics["throughput"], dtype=np.float32)
        violations = np.array(self.episode_metrics["violations"], dtype=np.float32)
        load_balance = np.array(self.episode_metrics["load_balance"], dtype=np.float32)
        summary = {
            "avg_latency_ms": float(latency_ms.mean()) if latency_ms.size else 0.0,
            "p99_latency_ms": float(np.percentile(latency_ms, 99)) if latency_ms.size else 0.0,
            "avg_energy_J": float(energy.mean()) if energy.size else 0.0,
            "throughput_tps": float(throughput.mean()) if throughput.size else 0.0,
            "deadline_violation_rate": float(violations.mean()) if violations.size else 0.0,
            "load_balance_jain": float(load_balance.mean()) if load_balance.size else 0.0,
        }
        return summary

    def close(self):
        return

    def render(self):
        return

    def save_replay(self):
        return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_metrics(self):
        return {
            "latency": [],
            "energy": [],
            "violations": [],
            "throughput": [],
            "load_balance": [],
        }

    def _sample_time_window(self):
        rate = self._current_task_rate()
        expected = 1.0 / max(rate, 1e-6)
        dt = self.rng.exponential(expected)
        return float(np.clip(dt, self.min_time_step, self.max_time_step))

    def _current_task_rate(self):
        if self.task_rate_mode == "cyclic":
            phase = (self.sim_time % self.task_rate_cycle) / self.task_rate_cycle
            min_rate, max_rate = self.task_rate_range
            return min_rate + (max_rate - min_rate) * 0.5 * (1 + math.sin(2 * math.pi * phase - math.pi / 2))
        if self.task_rate_mode == "random":
            return self.rng.uniform(*self.task_rate_range)
        return self.task_rate

    def _sample_task(self, agent_idx: int):
        workload = self.rng.uniform(*self.task_workload_range)  # Giga cycles
        data = self.rng.uniform(*self.task_data_range)  # MB
        deadline = self.sim_time + self.rng.uniform(*self.task_deadline_range)
        return {
            "agent": agent_idx,
            "workload": workload,
            "data": data,
            "deadline": deadline,
        }

    def _advance_system(self, time_window: float):
        processed = self.node_cpu_capacity * time_window
        self.nodes_load = np.maximum(0.0, self.nodes_load - processed)
        self.nodes_queue = np.maximum(0.0, self.nodes_queue - 1)

    def _spawn_new_tasks(self):
        for agent_idx in range(self.n_agents):
            self.tasks_to_dispatch[agent_idx] = self._sample_task(agent_idx)

    def _build_obs(self, agent_idx: int):
        load_ratio = self.nodes_load[agent_idx] / (self.node_cpu_capacity[agent_idx] + 1e-9)
        queue_len = self.nodes_queue[agent_idx]
        task = self.tasks_to_dispatch.get(agent_idx)
        if task is None:
            task_data = 0.0
            task_workload = 0.0
            task_deadline_remaining = 0.0
        else:
            task_data = task["data"]
            task_workload = task["workload"]
            task_deadline_remaining = max(0.0, task["deadline"] - self.sim_time)

        neighbor_loads = []
        for idx in range(self.n_agents):
            if idx == agent_idx:
                continue
            neighbor_loads.append(self.nodes_load[idx] / (self.node_cpu_capacity[idx] + 1e-9))

        obs = np.array(
            [load_ratio, queue_len, task_data, task_workload, task_deadline_remaining] + neighbor_loads,
            dtype=np.float32,
        )
        return obs

    def _compute_reward(self, latency, energy, violations):
        alpha = self.reward_weights.get("alpha", 0.5)
        beta = self.reward_weights.get("beta", 0.3)
        gamma = self.reward_weights.get("gamma", 0.2)
        return -(alpha * latency + beta * energy + gamma * violations)

    @staticmethod
    def _compute_jain_index(values: np.ndarray):
        numerator = np.square(np.sum(values))
        denominator = values.size * np.sum(np.square(values))
        if denominator <= 0:
            return 0.0
        return float(numerator / denominator)

    def _convert_hybrid_action(self, action_vec):
        vec = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        if vec.size < self.hybrid_action_dim:
            padded = np.zeros(self.hybrid_action_dim, dtype=np.float32)
            padded[: vec.size] = vec
            vec = padded
        placement_logits = vec[: self.n_agents]
        if np.allclose(placement_logits, 0.0):
            placement_idx = 0
        else:
            placement_idx = int(np.argmax(placement_logits)) % self.n_agents
        resource_raw = float(vec[-1])
        resource_norm = (resource_raw + 1.0) / 2.0  # map from [-1,1] to [0,1]
        resource_share = 0.1 + 0.9 * np.clip(resource_norm, 0.0, 1.0)
        return placement_idx, resource_share

