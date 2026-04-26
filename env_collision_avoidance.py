from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class EnvConfig:
    map_width: float = 50.0
    n_agents: int = 5
    vmax: float = 10.0
    amax: float = 5.0
    dsense: float = 15.0
    dcolli: float = 1.0
    dmin: float = 1.0
    dscale: float = 25.0
    dt: float = 1.0 / 50.0
    max_steps: int = 1500
    goal_threshold: float = 1.0


class MultiUAVCollisionEnv:
    """Paper-faithful training environment for decentralized multi-UAV collision avoidance.

    Each UAV flies at constant altitude in a 2D square map.
    Observation per agent is 10-D:
      [v_agent/vmax, psi_agent/pi, min(1,d_goal/dscale), theta_goal/pi,
       d_obs1/dsense, theta_obs1/pi, psi_obs1/pi,
       d_obs2/dsense, theta_obs2/pi, psi_obs2/pi]
    Action per agent is 2-D continuous in [-1, 1]:
      [2v/vmax - 1, theta/pi]
    """

    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.step_count = 0
        self.pos = np.zeros((self.cfg.n_agents, 2), dtype=np.float32)
        self.vel = np.zeros((self.cfg.n_agents, 2), dtype=np.float32)
        self.heading = np.zeros((self.cfg.n_agents,), dtype=np.float32)
        self.depots = np.zeros((self.cfg.n_agents, 2), dtype=np.float32)
        self.goals = np.zeros((self.cfg.n_agents, 2), dtype=np.float32)
        self.done_agents = np.zeros((self.cfg.n_agents,), dtype=bool)
        self.initial_goal_dist = np.ones((self.cfg.n_agents,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.step_count = 0
        half = self.cfg.map_width / 2.0

        while True:
            depots = np.random.uniform(-half, half, size=(self.cfg.n_agents, 2)).astype(np.float32)
            goals = np.random.uniform(-half, half, size=(self.cfg.n_agents, 2)).astype(np.float32)
            if self._all_separated(depots, 3.0) and self._all_separated(goals, 3.0):
                break

        self.depots[:] = depots
        self.goals[:] = goals
        self.pos[:] = depots
        self.vel[:] = 0.0
        self.heading[:] = np.random.uniform(-math.pi, math.pi, size=self.cfg.n_agents)
        self.done_agents[:] = False
        self.initial_goal_dist[:] = np.linalg.norm(self.goals - self.pos, axis=1).clip(min=1e-6)
        return self._get_all_states()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        actions = np.clip(actions, -1.0, 1.0)
        self.step_count += 1

        prev_goal_dists = np.linalg.norm(self.goals - self.pos, axis=1)

        for i in range(self.cfg.n_agents):
            if self.done_agents[i]:
                continue
            speed = (actions[i, 0] + 1.0) * 0.5 * self.cfg.vmax
            theta = actions[i, 1] * math.pi
            new_v = np.array([speed * math.cos(theta), speed * math.sin(theta)], dtype=np.float32)
            accel = (new_v - self.vel[i]) / self.cfg.dt
            accel_norm = np.linalg.norm(accel)
            if accel_norm > self.cfg.amax:
                accel = accel / accel_norm * self.cfg.amax
                new_v = self.vel[i] + accel * self.cfg.dt
            self.vel[i] = new_v
            self.heading[i] = math.atan2(float(new_v[1]), float(new_v[0])) if np.linalg.norm(new_v) > 1e-6 else self.heading[i]
            self.pos[i] += self.vel[i] * self.cfg.dt

        next_goal_dists = np.linalg.norm(self.goals - self.pos, axis=1)
        min_obs_dists = self._pairwise_min_dists()

        rewards = np.zeros((self.cfg.n_agents,), dtype=np.float32)
        for i in range(self.cfg.n_agents):
            rewards[i] = self._reward_for_agent(i, prev_goal_dists[i], next_goal_dists[i], min_obs_dists[i])

        for i in range(self.cfg.n_agents):
            out_of_bounds = np.any(np.abs(self.pos[i]) > self.cfg.map_width / 2.0)
            reached = next_goal_dists[i] <= self.cfg.goal_threshold
            self.done_agents[i] = self.done_agents[i] or reached or out_of_bounds

        done = bool(np.all(self.done_agents) or self.step_count >= self.cfg.max_steps)
        states = self._get_all_states()
        collision_mask = (min_obs_dists < self.cfg.dcolli)  # shape (n_agents,)
        info = {
            "success_rate": float(np.mean(np.linalg.norm(self.goals - self.pos, axis=1) <= self.cfg.goal_threshold)),
            "collision_rate": float(np.mean(min_obs_dists < self.cfg.dcolli)),
            "collision_mask": collision_mask,  # ← THÊM DÒNG NÀY
            "done_agents": self.done_agents.copy(),
        }
        return states, rewards, done, info

    def _reward_for_agent(self, i: int, dgoal_t: float, dgoal_tp1: float, d_obs_tp1: float) -> float:
        # Faithful implementation of Algorithm 1 from the paper (exact match)
        # Input: d_goal,t, d_goal,t+1, theta_goal,t+1, d_obs,t+1, d_init, vmax, d_colli, d_min
        theta_goal_tp1 = self._goal_angle_local(i)
        d_init = float(self.initial_goal_dist[i])

        # Khởi tạo r_t theo relative velocity 
        rel_velocity_to_goal = (dgoal_t - dgoal_tp1) / max(self.cfg.dt * self.cfg.vmax, 1e-6)
        rt = float(rel_velocity_to_goal)

        if d_obs_tp1 <= self.cfg.dcolli:
            rt = -2.0
        else:
            if rt > 0:
                rt = rt * (1.0 - dgoal_tp1 / d_init)
            else:
                rt = rt * (1.0 + dgoal_tp1 / d_init)
            
            rt = rt - 0.01 * abs(theta_goal_tp1)
            rt = rt - 0.01 * min(self.cfg.vmax / d_init, 1.0)   # paper viết d_max → là lỗi đánh máy, phải là vmax

        # KHÔNG có bonus +2.0 khi đạt goal (paper không có)
        return float(rt)

    def _all_separated(self, points: np.ndarray, min_dist: float) -> bool:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if np.linalg.norm(points[i] - points[j]) < min_dist:
                    return False
        return True

    def _pairwise_min_dists(self) -> np.ndarray:
        out = np.full((self.cfg.n_agents,), np.inf, dtype=np.float32)
        for i in range(self.cfg.n_agents):
            for j in range(self.cfg.n_agents):
                if i == j:
                    continue
                d = np.linalg.norm(self.pos[i] - self.pos[j])
                out[i] = min(out[i], d)
        return out

    def _goal_angle_local(self, i: int) -> float:
        rel = self.goals[i] - self.pos[i]
        goal_angle_world = math.atan2(float(rel[1]), float(rel[0]))
        theta_goal = self._wrap_to_pi(goal_angle_world - float(self.heading[i]))
        return theta_goal

    def _get_agent_info(self, i: int) -> np.ndarray:
        speed = float(np.linalg.norm(self.vel[i]))
        d_goal = float(np.linalg.norm(self.goals[i] - self.pos[i]))
        theta_goal = self._goal_angle_local(i)
        return np.array([
            speed / self.cfg.vmax,
            float(self.heading[i]) / math.pi,
            min(1.0, d_goal / self.cfg.dscale),
            theta_goal / math.pi,
        ], dtype=np.float32)

    def _get_obs_info(self, i: int) -> np.ndarray:
        vals: List[Tuple[float, np.ndarray]] = []
        for j in range(self.cfg.n_agents):
            if i == j:
                continue
            rel = self.pos[j] - self.pos[i]
            d = float(np.linalg.norm(rel))
            angle_world = math.atan2(float(rel[1]), float(rel[0]))
            theta_obs = self._wrap_to_pi(angle_world - float(self.heading[i]))
            psi_obs = self._wrap_to_pi(float(self.heading[j]) - float(self.heading[i]))
            vals.append((d, np.array([
                min(d / self.cfg.dsense, 1.0),
                theta_obs / math.pi,
                psi_obs / math.pi,
            ], dtype=np.float32)))

        vals.sort(key=lambda x: x[0])
        selected = [x[1] for x in vals[:2] if x[0] <= self.cfg.dsense]
        while len(selected) < 2:
            selected.append(np.array([1.0, 1.0, 0.0], dtype=np.float32))
        return np.concatenate(selected, axis=0)

    def _get_state(self, i: int) -> np.ndarray:
        return np.concatenate([self._get_agent_info(i), self._get_obs_info(i)], axis=0).astype(np.float32)

    def _get_all_states(self) -> np.ndarray:
        return np.vstack([self._get_state(i) for i in range(self.cfg.n_agents)]).astype(np.float32)

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
