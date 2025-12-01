"""
真正的 MAPPO (Multi-Agent PPO) 实现
基于论文: "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
参考实现: https://github.com/marlbenchmark/on-policy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from collections import deque
import random


class MAPPOActor(nn.Module):
    """MAPPO 的 Actor 网络（策略网络）

    为了支持混合动作空间，我们让 Actor 输出一个连续向量 a∈R^{action_dim}，
    后续由上层（如 MAPPOTrainer）将该向量解释为 [placement_logits, resource_scalar]：
    - placement_logits: 前 n_agents 维，用 argmax 得到离散放置节点
    - resource_scalar: 最后一维，经非线性映射到 [0.1, 1.0] 的资源占比

    在 PPO 中，我们将该连续向量视为对角高斯分布 Normal(μ, σ) 的采样，
    log_prob 为各维 log_prob 之和。
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        # 使用可学习的对角协方差，对所有 agent 共享
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        """返回连续动作的均值向量 μ(obs)"""
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        return mu

    def _dist(self, obs):
        mu = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mu)
        # 独立高斯分布，每一维一个 Normal
        dist = Normal(mu, std)
        return dist

    def get_action_log_prob(self, obs, action=None):
        """获取动作和对数概率

        Args:
            obs: [B, obs_dim] 或 [1, obs_dim]
            action: 若为 None，则采样；否则使用给定动作计算 log_prob

        Returns:
            action: 采样或给定的动作张量，形状与 μ 相同
            log_prob: 每个样本对应的标量 log_prob（各维求和）
            entropy: 每个样本对应的标量熵（各维熵求和）
        """
        dist = self._dist(obs)

        if action is None:
            action = dist.rsample()

        # 对每个样本，按维度求和得到标量 log_prob / entropy
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy


class MAPPOCritic(nn.Module):
    """MAPPO 的 Critic 网络（中心化价值函数）"""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        # 关键：接收全局状态（所有 agent 的观测拼接）
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """
        state: 全局状态（中心化）
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value_out(x)
        return value


class RolloutBuffer:
    """经验回放缓冲区"""
    def __init__(self):
        self.observations = []
        self.states = []  # 全局状态
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, obs, state, action, reward, done, log_prob, value):
        self.observations.append(obs)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.observations = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def get(self):
        return {
            'observations': self.observations,
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'log_probs': self.log_probs,
            'values': self.values
        }


class MAPPO:
    """
    真正的 MAPPO 算法实现
    
    核心特性：
    1. Centralized Critic: 价值函数使用全局状态
    2. Decentralized Actor: 每个 agent 的策略只依赖局部观测
    3. Parameter Sharing: 所有 agent 共享相同的 Actor 参数
    4. Value Normalization: 对价值函数进行归一化
    """
    
    def __init__(
        self,
        obs_dim,
        state_dim,
        action_dim,
        n_agents,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        value_loss_coeff=0.5,
        entropy_coeff=0.01,
        max_grad_norm=0.5,
        num_sgd_iter=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.num_sgd_iter = num_sgd_iter
        self.device = device
        
        # 创建网络（所有 agent 共享）
        self.actor = MAPPOActor(obs_dim, action_dim).to(device)
        self.critic = MAPPOCritic(state_dim).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 经验缓冲区
        self.buffer = RolloutBuffer()
        
        # 价值归一化（MAPPO 的关键技巧）
        self.value_normalizer = ValueNormalizer(1, device=device)
    
    def get_actions(self, observations, explore=True):
        """
        获取所有 agent 的动作
        
        Args:
            observations: dict {agent_id: obs}
            explore: 是否探索
        
        Returns:
            actions: dict {agent_id: action}
            log_probs: dict {agent_id: log_prob}
        """
        actions = {}
        log_probs = {}
        
        self.actor.eval()
        with torch.no_grad():
            for agent_id, obs in observations.items():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                if explore:
                    # 探索时从高斯分布采样
                    action_tensor, log_prob_tensor, _ = self.actor.get_action_log_prob(obs_tensor)
                    action_np = action_tensor.squeeze(0).cpu().numpy()
                    log_prob = float(log_prob_tensor.item())
                else:
                    # 确定性策略（评估时）：直接取均值 μ(obs)
                    mu = self.actor.forward(obs_tensor)
                    action_np = mu.squeeze(0).cpu().numpy()
                    log_prob = 0.0

                actions[agent_id] = action_np
                log_probs[agent_id] = log_prob
        
        self.actor.train()
        return actions, log_probs
    
    def get_values(self, state):
        """
        获取全局状态的价值估计
        
        Args:
            state: 全局状态（所有 agent 观测的拼接）
        """
        self.critic.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.critic(state_tensor)
            value = self.value_normalizer.denormalize(value)
        self.critic.train()
        return value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        计算 Generalized Advantage Estimation (GAE)
        
        这是 MAPPO 论文中强调的关键技术
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update(self):
        """
        更新 Actor 和 Critic 网络
        
        使用 PPO-clip 目标函数
        """
        data = self.buffer.get()
        
        if len(data['rewards']) == 0:
            return {}
        
        # 转换为 tensor
        observations = torch.FloatTensor(np.array(data['observations'])).to(self.device)
        states = torch.FloatTensor(np.array(data['states'])).to(self.device)
        # 连续动作向量 [T, action_dim]
        actions = torch.FloatTensor(np.array(data['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(data['log_probs'])).to(self.device)
        values = data['values']
        
        # 计算 GAE
        next_value = 0  # 假设 episode 结束
        advantages, returns = self.compute_gae(
            data['rewards'], values, data['dones'], next_value
        )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 归一化 advantages（MAPPO 的关键技巧）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮 SGD 更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for _ in range(self.num_sgd_iter):
            # 重新计算 log probs 和 values（使用连续动作分布）
            _, new_log_probs, entropy = self.actor.get_action_log_prob(observations, actions)
            new_values = self.critic(states).squeeze(-1)
            
            # Actor loss (PPO-clip)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss (MSE)
            new_values_normalized = self.value_normalizer(new_values)
            returns_normalized = self.value_normalizer(returns)
            critic_loss = 0.5 * ((new_values_normalized - returns_normalized) ** 2).mean()
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = actor_loss + self.value_loss_coeff * critic_loss + self.entropy_coeff * entropy_loss
            
            # 更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
        
        # 清空缓冲区
        self.buffer.clear()
        
        return {
            'actor_loss': total_actor_loss / self.num_sgd_iter,
            'critic_loss': total_critic_loss / self.num_sgd_iter,
            'entropy': total_entropy / self.num_sgd_iter
        }
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'value_normalizer': self.value_normalizer.state_dict()
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.value_normalizer.load_state_dict(checkpoint['value_normalizer'])


class ValueNormalizer(nn.Module):
    """
    价值函数归一化
    
    MAPPO 论文强调这对训练稳定性至关重要
    """
    def __init__(self, input_shape, device='cpu'):
        super().__init__()
        self.device = device
        self.register_buffer('running_mean', torch.zeros(input_shape).to(device))
        self.register_buffer('running_var', torch.ones(input_shape).to(device))
        self.register_buffer('count', torch.ones(()).to(device))
    
    def forward(self, x):
        """归一化"""
        return (x - self.running_mean) / torch.sqrt(self.running_var + 1e-8)
    
    def denormalize(self, x):
        """反归一化"""
        return x * torch.sqrt(self.running_var + 1e-8) + self.running_mean
    
    def update(self, x):
        """更新统计量"""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """从批次统计更新运行统计"""
        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count
        
        new_mean = self.running_mean + delta * batch_count / tot_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.running_mean = new_mean
        self.running_var = new_var
        self.count = tot_count

