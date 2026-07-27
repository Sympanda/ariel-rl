"""
SmartGreedy baseline: pick the valid action that maximises science return
per unit of time cost, explicitly accounting for slew overhead.

Score = science_weight × (1 + α × progress_in_tier)
        ─────────────────────────────────────────────
        slew_time_norm + duration_norm + ε

Numerator
---------
  ``science_weight``   — intrinsic science priority of the target
  ``progress_in_tier`` — how close the target is to its next tier;
                          scaled by ``alpha`` (set to 0 for pure value/cost)

Denominator
-----------
  ``slew_time_norm``   — normalised slew time to this target
  ``duration_norm``    — normalised observation duration (T14)
  ``ε``                — small constant to avoid division by zero when the
                          telescope is already pointing at the target

Unlike ``GreedyValue`` (ignores timing) and ``GreedyBalanced`` (ignores
cost), SmartGreedy explicitly penalises events that require long slews.
It will prefer a slightly lower-value target that is nearby over a
high-value target on the opposite side of the sky.
"""

from __future__ import annotations

import numpy as np

from ariel_rl.baselines.base import BaselineAgent
from ariel_rl.utils.config import ObservationConfig


class SmartGreedy(BaselineAgent):
    """Pick the valid candidate maximising science-per-unit-cost.

    Score = science_weight × (1 + α × progress_in_tier)
            / (slew_time_norm + duration_norm + ε)

    Parameters
    ----------
    obs_cfg:
        ``ObservationConfig`` from the env (to locate feature columns).
        Can be omitted; falls back to positional defaults that match the
        standard feature list.
    alpha:
        Weight on the near-completion progress bonus.  Set to 0 to score
        purely on science_weight / cost.  Default 1.0.
    eps:
        Small denominator offset preventing division by zero when slew
        and duration are both near zero.  Default 0.01 (in normalised units).
    """

    def __init__(
        self,
        obs_cfg: ObservationConfig | None = None,
        alpha: float = 1.0,
        eps: float = 0.01,
        seed: int = 0,
    ) -> None:
        super().__init__(seed)
        self.alpha = alpha
        self.eps   = eps

        # Feature indices — resolved from config if available
        self._slew_idx: int | None = None
        self._dur_idx:  int | None = None
        self._sw_idx:   int | None = None
        self._prog_idx: int | None = None

        if obs_cfg is not None:
            feats = obs_cfg.event_features
            def _idx(name: str) -> int | None:
                return feats.index(name) if name in feats else None
            self._slew_idx = _idx("slew_time_days")
            self._dur_idx  = _idx("duration_days")
            self._sw_idx   = _idx("science_weight")
            self._prog_idx = _idx("progress_in_tier")

    def act(self, obs: dict, info: dict) -> int:
        valid = self._valid_indices(info)
        if len(valid) == 0:
            return 0

        events: np.ndarray = obs["events"]   # (K, D)
        K = events.shape[0]

        # ── feature extraction (fall back to safe defaults) ──────────────
        slew  = events[:, self._slew_idx] if self._slew_idx is not None else np.zeros(K)
        dur   = events[:, self._dur_idx]  if self._dur_idx  is not None else np.ones(K)
        sw    = events[:, self._sw_idx]   if self._sw_idx   is not None else np.ones(K)
        prog  = events[:, self._prog_idx] if self._prog_idx is not None else np.zeros(K)

        # ── score = value / cost ──────────────────────────────────────────
        value = sw * (1.0 + self.alpha * prog)
        cost  = np.abs(slew) + np.abs(dur) + self.eps   # abs: guard against negative normalised values
        scores = value / cost

        # ── apply mask and pick best ──────────────────────────────────────
        masked = np.full(K, -np.inf)
        masked[valid] = scores[valid]
        return int(np.argmax(masked))
