from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np


@dataclass
class Batch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=self.capacity)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: float) -> None:
        self.buffer.append((state.astype(np.float32), action.astype(np.float32), float(reward), next_state.astype(np.float32), float(done)))

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idx))
        return Batch(
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
            next_states=np.asarray(next_states, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self) -> int:
        return len(self.buffer)
