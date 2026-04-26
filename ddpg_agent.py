from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer


@dataclass
class DDPGConfig:
    state_dim: int = 10
    action_dim: int = 2
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    replay_size: int = 10**6
    batch_size: int = 256
    device: str = "cpu"
    noise_std: float = 0.1


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class DDPGAgent:
    def __init__(self, cfg: DDPGConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = Actor(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.replay = ReplayBuffer(cfg.replay_size)

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state_t).detach().cpu().numpy()[0]

        if not deterministic:
            action += np.random.normal(0, self.cfg.noise_std, size=action.shape)

        return np.clip(action, -1, 1)

    def update(self) -> Dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {}

        batch = self.replay.sample(self.cfg.batch_size)

        s = torch.FloatTensor(batch.states).to(self.device)
        a = torch.FloatTensor(batch.actions).to(self.device)
        r = torch.FloatTensor(batch.rewards).to(self.device)
        ns = torch.FloatTensor(batch.next_states).to(self.device)
        d = torch.FloatTensor(batch.dones).to(self.device)

        with torch.no_grad():
            next_a = self.actor_target(ns)
            target_q = self.critic_target(ns, next_a)
            y = r + (1 - d) * self.cfg.gamma * target_q

        # critic
        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # actor
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def _soft_update(self):
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.cfg.tau * p.data + (1 - self.cfg.tau) * tp.data)

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.cfg.tau * p.data + (1 - self.cfg.tau) * tp.data)

    def save(self, actor_path: str, prefix: str) -> None:
        torch.save(self.actor.state_dict(), actor_path)
        # 🔥 FIX: save critic để có thể resume training (đồng nhất với SAC)
        torch.save(self.critic.state_dict(), prefix + "_critic.pt")