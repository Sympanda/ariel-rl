"""
Abstract base class for all baseline schedulers.

The interface mirrors what an RL agent receives:
  act(obs, info) → int action index

``obs``  is the dict from ArielEnv: {"events": float32 (K×D), "global": float32 (G,)}
``info`` is the dict from ArielEnv: {"action_mask": bool (K,), "step_result": …, …}

Baselines may also accept an optional ``state`` argument (the raw MissionState)
for baselines that need richer information than the obs arrays provide.
However, any baseline that relies on ``state`` is not directly comparable to a
policy-gradient agent that only sees obs/info.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaselineAgent(ABC):
    """Abstract baseline agent."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def act(self, obs: dict, info: dict) -> int:
        """Choose an action given the current observation and info dict.

        Parameters
        ----------
        obs:
            ``{"events": float32 (K, D), "global": float32 (G,)}``
        info:
            Env info dict containing at minimum ``"action_mask": bool (K,)``.

        Returns
        -------
        int
            Action index.  Must be within [0, K) and should be valid
            (i.e. ``info["action_mask"][action] == True``).
        """

    def reset(self) -> None:
        """Called at the start of each episode.  Override if stateful."""

    def _valid_indices(self, info: dict) -> np.ndarray:
        """Return indices of valid actions from the action mask."""
        mask: np.ndarray = info["action_mask"]
        return np.where(mask)[0]

    def _fallback(self, info: dict) -> int:
        """Return the first valid action, or 0 if none exist."""
        valid = self._valid_indices(info)
        return int(valid[0]) if len(valid) > 0 else 0
