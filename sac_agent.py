from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer


LOG_STD_MIN = -20
LOG_STD_MAX = 2


@dataclass
class SACConfig:
    state_dim: int = 10
    action_dim: int = 2
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    replay_size: int = 10**6
    batch_size: int = 256
    device: str = "cpu"
    target_entropy: float = -2.0


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        x = self.backbone(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP(state_dim + action_dim, hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class SACAgent:
    def __init__(self, cfg: SACConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = GaussianPolicy(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.q1 = QNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.q2 = QNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.q1_target = QNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.q2_target = QNetwork(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        self.log_alpha = torch.tensor(math.log(0.5), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = cfg.target_entropy

        self.replay = ReplayBuffer(cfg.replay_size)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(state_t)
            else:
                action, _, _ = self.actor.sample(state_t)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self) -> Dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            return {}

        batch = self.replay.sample(self.cfg.batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions, next_logp, _ = self.actor.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_logp
            target_q = rewards + (1.0 - dones) * self.cfg.gamma * q_next

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_actions, logp, _ = self.actor.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * logp - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def _soft_update(self, src: nn.Module, dst: nn.Module) -> None:
        for src_param, dst_param in zip(src.parameters(), dst.parameters()):
            dst_param.data.copy_(self.cfg.tau * src_param.data + (1.0 - self.cfg.tau) * dst_param.data)

    def save(self, actor_path: str, critic_prefix: str) -> None:
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.q1.state_dict(), critic_prefix + "_q1.pt")
        torch.save(self.q2.state_dict(), critic_prefix + "_q2.pt")

    def load_actor(self, actor_path: str) -> None:
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.actor.eval()
