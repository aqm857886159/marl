import copy
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dim, action_dim)
        self.output_activation = nn.Tanh()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.net(obs))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        input_dim = state_dim + action_dim
        self.net = mlp(input_dim, hidden_dim, 1)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, actions], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros_like(self.state)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.state[self.ptr] = state
        self.next_state[self.ptr] = next_state
        self.actions[self.ptr] = actions
        self.rewards[self.ptr, 0] = reward
        self.dones[self.ptr, 0] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.choice(self.size, batch_size, replace=False)
        batch = dict(
            obs=torch.as_tensor(self.obs[idx], device=self.device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=self.device),
            state=torch.as_tensor(self.state[idx], device=self.device),
            next_state=torch.as_tensor(self.next_state[idx], device=self.device),
            actions=torch.as_tensor(self.actions[idx], device=self.device),
            rewards=torch.as_tensor(self.rewards[idx], device=self.device),
            dones=torch.as_tensor(self.dones[idx], device=self.device),
        )
        return batch


class OrnsteinUhlenbeckNoise:
    def __init__(self, size: Tuple[int, ...], theta: float, sigma: float, dt: float = 1.0):
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.size = size
        self.state = np.zeros(size, dtype=np.float32)

    def reset(self):
        self.state.fill(0.0)

    def sample(self):
        dx = self.theta * (-self.state) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(*self.size)
        self.state += dx.astype(np.float32)
        return self.state


@dataclass
class MADDPGConfig:
    n_agents: int
    obs_dim: int
    action_dim: int
    state_dim: int
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.01
    hidden_dim: int = 64
    device: torch.device = torch.device("cpu")


class MADDPG:
    def __init__(self, config: MADDPGConfig):
        self.config = config
        self.device = config.device

        self.actors = nn.ModuleList(
            [Actor(config.obs_dim, config.action_dim, config.hidden_dim).to(self.device) for _ in range(config.n_agents)]
        )
        self.critics = Critic(config.state_dim, config.n_agents * config.action_dim, config.hidden_dim).to(self.device)

        self.target_actors = copy.deepcopy(self.actors)
        self.target_critic = copy.deepcopy(self.critics)

        self.actor_optim = [
            optim.Adam(actor.parameters(), lr=config.actor_lr) for actor in self.actors
        ]
        self.critic_optim = optim.Adam(self.critics.parameters(), lr=config.critic_lr)

    def to(self, device: torch.device):
        self.device = device
        self.actors.to(device)
        self.critics.to(device)
        self.target_actors.to(device)
        self.target_critic.to(device)

    def select_actions(self, obs: np.ndarray, noise: np.ndarray = None, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = []
        with torch.no_grad():
            for agent_idx, actor in enumerate(self.actors):
                action = actor(obs_tensor[agent_idx])
                actions.append(action)
        actions_tensor = torch.stack(actions, dim=0)
        actions_np = actions_tensor.cpu().numpy()
        if not deterministic and noise is not None:
            actions_np += noise
        return np.clip(actions_np, -1.0, 1.0)

    def update(self, batch, update_iters: int = 1):
        for _ in range(update_iters):
            self._update_step(batch)

    def _update_step(self, batch):
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        state = batch["state"]
        next_state = batch["next_state"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions = []
            for agent_idx, target_actor in enumerate(self.target_actors):
                next_actions.append(target_actor(next_obs[:, agent_idx]))
            next_actions_tensor = torch.cat(next_actions, dim=-1)
            target_q = self.target_critic(next_state, next_actions_tensor)
            y = rewards + self.config.gamma * (1.0 - dones) * target_q

        current_actions = actions.view(actions.size(0), -1)
        q_value = self.critics(state, current_actions)
        critic_loss = nn.MSELoss()(q_value, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=1.0)
        self.critic_optim.step()

        actor_actions = []
        for agent_idx, actor in enumerate(self.actors):
            actor_actions.append(actor(obs[:, agent_idx]))
        actor_actions_tensor = torch.cat(actor_actions, dim=-1)
        actor_loss = -self.critics(state, actor_actions_tensor).mean()

        for optim_i in self.actor_optim:
            optim_i.zero_grad()
        actor_loss.backward()
        for actor, optim_i in zip(self.actors, self.actor_optim):
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optim_i.step()

        self._soft_update()

    def _soft_update(self):
        tau = self.config.tau
        for target_param, param in zip(self.target_critic.parameters(), self.critics.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_actor, actor in zip(self.target_actors, self.actors):
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


