"""
MAPPO 训练器 - 与 EdgeSimGym (OpenAI Gym) 集成
"""

import numpy as np
import torch
from mappo_algorithm import MAPPO

# 延迟导入，避免循环依赖
EdgeSimGym = None
ENV_CONFIG = None

def _lazy_import():
    """延迟导入 MRRL 模块"""
    global EdgeSimGym, ENV_CONFIG
    if EdgeSimGym is None:
        from MRRL import EdgeSimGym as _EdgeSimGym, ENV_CONFIG as _ENV_CONFIG
        EdgeSimGym = _EdgeSimGym
        ENV_CONFIG = _ENV_CONFIG
    return EdgeSimGym, ENV_CONFIG


class MAPPOTrainer:
    """
    MAPPO 训练器，适配 OpenAI Gym MultiAgentEnv 接口
    """
    
    def __init__(self, env_config, seed=0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            env_config: 环境配置（来自 ENV_CONFIG）
            seed: 随机种子
            device: 'cuda' 或 'cpu'
        """
        _EdgeSimGym, _ = _lazy_import()
        # MAPPO 不需要 QMIX 格式
        env_config_copy = {**env_config}
        self.env = _EdgeSimGym(env_config_copy)
        self.seed = seed
        self.device = device
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # 获取环境信息
        self.n_agents = env_config["n_nodes"]
        
        # 获取观测和动作空间维度
        # 由于泊松分布可能生成0个任务，需要多次尝试
        sample_obs = None
        max_attempts = 10
        for attempt in range(max_attempts):
            sample_obs, _ = self.env.reset(seed=seed + attempt)
            if sample_obs:
                break
        
        if sample_obs:
            first_agent = list(sample_obs.keys())[0]
            self.obs_dim = len(sample_obs[first_agent])

            # 动作空间：混合动作 = placement_logits (n_agents) + resource_scalar (1)
            # 在 MAPPO 算法内部视为连续动作向量，由高斯策略输出
            self.action_dim = env_config["n_nodes"] + 1
        else:
            # 如果仍然获取不到观测，使用固定值
            # 观测维度 = 5 (本地特征) + 9 (其他节点) = 14
            print("[WARNING] 无法从环境获取观测，使用固定维度")
            self.obs_dim = 5 + (self.n_agents - 1)  # 5个本地特征 + n-1个邻居
            self.action_dim = env_config["n_nodes"] + 1
        
        # 全局状态维度（所有 agent 的观测拼接）
        self.state_dim = self.obs_dim * self.n_agents
        
        # 创建 MAPPO 算法
        self.mappo = MAPPO(
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            n_agents=self.n_agents,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_param=0.2,
            value_loss_coeff=0.5,
            entropy_coeff=0.01,
            max_grad_norm=0.5,
            num_sgd_iter=4,
            device=device
        )
        
        print(f"[OK] MAPPO 初始化完成:")
        print(f"  - 观测维度: {self.obs_dim}")
        print(f"  - 状态维度: {self.state_dim}")
        print(f"  - 动作维度: {self.action_dim}")
        print(f"  - 智能体数: {self.n_agents}")
        print(f"  - 设备: {self.device}")
    
    def obs_dict_to_state(self, obs_dict):
        """
        将观测字典转换为全局状态向量
        
        Args:
            obs_dict: {agent_id: obs_array}
        
        Returns:
            state: 全局状态向量 (拼接所有 agent 的观测)
        """
        # 关键：始终为所有 agent 生成观测，缺失的用零向量填充
        # 这确保状态维度始终一致
        state_list = []
        for i in range(self.n_agents):
            agent_id = f"node_{i}"
            if agent_id in obs_dict:
                state_list.append(obs_dict[agent_id])
            else:
                # 用零向量填充缺失的观测
                state_list.append(np.zeros(self.obs_dim, dtype=np.float32))
        
        state = np.concatenate(state_list)
        return state
    
    def _format_env_actions(self, actions_dict):
        """
        将 MAPPO 连续动作向量映射为环境期望的混合动作：
        - 动作向量 a ∈ R^{n_agents+1}
          - 前 n_agents 维作为 placement logits，取 argmax 得到离散节点
          - 最后一维作为 resource_raw，经 tanh 压到 [-1,1] 后映射到 [0.1,1.0]
        """
        formatted = {}
        for agent_id, action_vec in actions_dict.items():
            vec = np.asarray(action_vec, dtype=np.float32).reshape(-1)
            if vec.size < (self.n_agents + 1):
                # 不足时补零，保证维度一致
                padded = np.zeros(self.n_agents + 1, dtype=np.float32)
                padded[: vec.size] = vec
                vec = padded

            placement_logits = vec[: self.n_agents]
            # 若 logits 全为 0，则默认选择 0 号节点
            if np.allclose(placement_logits, 0.0):
                placement = 0
            else:
                placement = int(np.argmax(placement_logits) % self.n_agents)

            resource_raw = float(vec[-1])
            # 先用 tanh 压缩到 [-1,1]，再映射到 [0,1]
            resource_norm = (np.tanh(resource_raw) + 1.0) / 2.0
            # 映射到 [0.1, 1.0] 的资源占比（与 EdgeMARLEnv 一致风格）
            resource_share = 0.1 + 0.9 * np.clip(resource_norm, 0.0, 1.0)

            formatted[agent_id] = {
                "placement": placement,
                "resource": np.array([resource_share], dtype=np.float32),
            }
        return formatted

    def _store_transition(self, obs_dict, actions_dict, log_probs_dict, state, reward, dones, value):
        done_flag = dones.get("__all__", False)
        for agent_id, obs in obs_dict.items():
            if agent_id not in actions_dict or agent_id not in log_probs_dict:
                continue
            action = actions_dict[agent_id]
            log_prob = log_probs_dict[agent_id]
            self.mappo.buffer.add(
                obs=obs,
                state=state,
                action=action,
                reward=reward,
                done=done_flag,
                log_prob=log_prob,
                value=value
            )

    def train_episode(self, explore=True, record_buffer=True):
        """
        训练一个 episode
        
        Returns:
            episode_stats: episode 统计信息
        """
        obs_dict, _ = self.env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        episode_metrics = {
            "avg_latency_ms": [],
            "p99_latency_ms": [],
            "avg_energy_J": [],
            "throughput_tps": [],
            "load_balance_jain": []
        }
        
        while not done:
            # 获取全局状态
            state = self.obs_dict_to_state(obs_dict)
            
            # 获取动作
            actions_dict, log_probs_dict = self.mappo.get_actions(obs_dict, explore=explore)
            env_actions = self._format_env_actions(actions_dict)
            
            # 执行动作
            next_obs_dict, rewards_dict, dones, truncated, infos = self.env.step(env_actions)
            
            # 获取全局奖励（所有 agent 共享）
            if rewards_dict:
                global_reward = list(rewards_dict.values())[0]
            else:
                global_reward = 0
            
            # 获取价值估计
            value = self.mappo.get_values(state)
            
            # 存储经验（每个时间步）
            if record_buffer and obs_dict:
                self._store_transition(obs_dict, actions_dict, log_probs_dict, state, global_reward, dones, value)
            
            # 收集 episode 指标
            if next_obs_dict:
                for agent_id in next_obs_dict.keys():
                    if "episode_metrics" in infos.get(agent_id, {}):
                        metrics = infos[agent_id]["episode_metrics"]
                        episode_metrics["avg_latency_ms"].append(metrics["avg_latency_ms"])
                        episode_metrics["p99_latency_ms"].append(metrics["p99_latency_ms"])
                        episode_metrics["avg_energy_J"].append(metrics["avg_energy_J"])
                        episode_metrics["throughput_tps"].append(metrics["throughput_tps"])
                        episode_metrics["load_balance_jain"].append(metrics["load_balance_jain"])
                        break
            
            obs_dict = next_obs_dict
            done = dones.get("__all__", False)
            episode_reward += global_reward
            episode_steps += 1
        
        # 计算平均指标
        avg_metrics = {}
        for key in episode_metrics:
            if episode_metrics[key]:
                avg_metrics[key] = np.mean(episode_metrics[key])
            else:
                avg_metrics[key] = 0.0
        
        return {
            "episode_reward": episode_reward,
            "episode_steps": episode_steps,
            **avg_metrics
        }
    
    def train(self, num_iterations=1000, update_interval=1, eval_interval=10, eval_episodes=10, max_total_steps=5_000_000):
        """
        训练 MAPPO
        
        Args:
            num_iterations: 训练迭代次数
            update_interval: 每隔多少个 episode 更新一次
            eval_interval: 每隔多少次更新评估一次
            eval_episodes: 评估时运行的 episode 数
        
        Returns:
            results_log: 训练日志
        """
        results_log = []
        total_steps = 0
        episode_count = 0

        # 训练以“总环境步数”为主；num_iterations 仅保留作接口兼容，不再作为硬停止条件
        print(f"\n开始 MAPPO 训练 (目标: {max_total_steps:,} 环境步)...")

        iteration = 0
        while True:
            iteration += 1
            # 训练 episode
            episode_stats = self.train_episode(explore=True, record_buffer=True)
            total_steps += episode_stats["episode_steps"]
            episode_count += 1

            # 定期更新
            if episode_count % update_interval == 0:
                update_info = self.mappo.update()

            # 定期评估
            if (iteration) % eval_interval == 0:
                eval_stats = self.evaluate(num_episodes=eval_episodes)
                
                log_entry = {
                    "algorithm": "MAPPO",
                    "seed": self.seed,
                    "iteration": iteration,
                    "timestep": total_steps,
                    "episode_return_mean": episode_stats["episode_reward"],
                    "avg_latency_ms": eval_stats["avg_latency_ms"],
                    "p99_latency_ms": eval_stats["p99_latency_ms"],
                    "avg_energy_J": eval_stats["avg_energy_J"],
                    "throughput_tps": eval_stats["throughput_tps"],
                    "load_balance_jain": eval_stats["load_balance_jain"]
                }
                
                results_log.append(log_entry)
                
                print(
                    f"Iter {iteration}/{num_iterations} | "
                    f"Steps: {total_steps} | "
                    f"Reward: {episode_stats['episode_reward']:.2f} | "
                    f"Latency: {eval_stats['avg_latency_ms']:.2f}ms"
                )

            # 按总步数停止，确保与论文 5M 步一致
            if total_steps >= max_total_steps:
                print(f"达到 {max_total_steps:,} 步，停止训练。")
                break
        
        print(f"[OK] MAPPO 训练完成")
        return results_log
    
    def evaluate(self, num_episodes=100):
        """
        评估当前策略
        
        Args:
            num_episodes: 评估的 episode 数
        
        Returns:
            eval_stats: 评估统计
        """
        all_metrics = {
            "avg_latency_ms": [],
            "p99_latency_ms": [],
            "avg_energy_J": [],
            "throughput_tps": [],
            "load_balance_jain": []
        }
        
        prev_flag = self.env.disable_env_exploration
        self.env.disable_env_exploration = True

        try:
            for _ in range(num_episodes):
                episode_stats = self.train_episode(explore=False, record_buffer=False)  # 确定性策略

                for key in all_metrics:
                    all_metrics[key].append(episode_stats[key])
        finally:
            self.env.disable_env_exploration = prev_flag

        # 计算平均值
        eval_stats = {key: np.mean(values) for key, values in all_metrics.items()}

        return eval_stats
    
    def save_model(self, path):
        """保存模型"""
        self.mappo.save(path)
        print(f"[OK] 模型已保存到 {path}")
    
    def load_model(self, path):
        """加载模型"""
        self.mappo.load(path)
        print(f"[OK] 模型已从 {path} 加载")


def run_mappo_experiment(seed, num_iterations=1000, max_total_steps=5_000_000):
    """
    运行 MAPPO 实验（与 run_experiment 接口一致）
    
    Args:
        seed: 随机种子
        num_iterations: 训练迭代次数
    
    Returns:
        results_log: 训练日志
    """
    _, _ENV_CONFIG = _lazy_import()
    print(f"\n--- 启动训练: 算法=MAPPO (真实实现), 种子={seed} ---")
    
    # 创建训练器
    trainer = MAPPOTrainer(_ENV_CONFIG, seed=seed)
    
    # 训练
    results_log = trainer.train(
        num_iterations=num_iterations,
        update_interval=1,
        eval_interval=10,
        eval_episodes=10,
        max_total_steps=max_total_steps
    )
    
    # 保存模型
    trainer.save_model(f"mappo_seed_{seed}.pt")
    
    print(f"--- 训练完成: 算法=MAPPO, 种子={seed} ---")
    return results_log


if __name__ == "__main__":
    # 测试 MAPPO 训练
    print("=== MAPPO 集成测试 ===")
    
    # 延迟导入
    _lazy_import()
    
    # 快速测试（10 次迭代）
    results = run_mappo_experiment(seed=0, num_iterations=10)
    
    print(f"\n测试完成！收集了 {len(results)} 条记录")
    if results:
        print(f"最后一条记录: {results[-1]}")

