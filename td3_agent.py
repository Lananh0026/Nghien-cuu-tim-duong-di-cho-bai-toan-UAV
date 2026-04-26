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
class TD3Config:
    state_dim: int = 10
    action_dim: int = 2
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    replay_size: int = 10**6
    device: str = "cpu"
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    # 🔥 FIX: thêm exploration noise std (tương tự DDPG)
    expl_noise: float = 0.1


class Actor(nn.Module):
    def __init__(self, s, a, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, a),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, s, a, h):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(s + a, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1),
        )
        self.q2 = copy.deepcopy(self.q1)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)


class TD3Agent:
    def __init__(self, cfg: TD3Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = Actor(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.replay = ReplayBuffer(cfg.replay_size)
        self.total_it = 0

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state_t).detach().cpu().numpy()[0]

        # 🔥 FIX CRITICAL: add exploration noise khi train (deterministic=False)
        # Trước đây bỏ qua flag này → TD3 không explore → học kém
        if not deterministic:
            noise = np.random.normal(0, self.cfg.expl_noise, size=action.shape)
            action = np.clip(action + noise, -1, 1)

        return action

    def update(self) -> Dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {}

        self.total_it += 1

        batch = self.replay.sample(self.cfg.batch_size)

        s = torch.FloatTensor(batch.states).to(self.device)
        a = torch.FloatTensor(batch.actions).to(self.device)
        r = torch.FloatTensor(batch.rewards).to(self.device)
        ns = torch.FloatTensor(batch.next_states).to(self.device)
        d = torch.FloatTensor(batch.dones).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.cfg.policy_noise).clamp(
                -self.cfg.noise_clip, self.cfg.noise_clip
            )
            next_a = (self.actor_target(ns) + noise).clamp(-1, 1)

            q1_t, q2_t = self.critic_target(ns, next_a)
            target_q = torch.min(q1_t, q2_t)
            y = r + (1 - d) * self.cfg.gamma * target_q

        q1, q2 = self.critic(s, a)
        loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

        if self.total_it % self.cfg.policy_delay == 0:
            actor_loss = -self.critic.q1(torch.cat([s, self.actor(s)], dim=-1)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update()

        return {"critic_loss": loss.item()}

    def _soft_update(self):
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.cfg.tau * p.data + (1 - self.cfg.tau) * tp.data)

        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.cfg.tau * p.data + (1 - self.cfg.tau) * tp.data)

    def save(self, actor_path: str, prefix: str) -> None:
        torch.save(self.actor.state_dict(), actor_path)
        # 🔥 FIX: save cả critic như SAC/DDPG để có thể resume
        torch.save(self.critic.state_dict(), prefix + "_critic.pt")