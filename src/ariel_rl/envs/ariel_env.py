"""
ArielEnv: Gymnasium environment for Ariel mission target scheduling.

Episode flow
------------
    obs, info = env.reset()
    while True:
        action = agent.act(obs, info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

Action spaces
-------------
``topk`` (default):
    Discrete(K).  The agent picks an index 0…K-1 into the K upcoming events.
    ``info["action_mask"]`` is a boolean array of shape (K,).

``target``:
    Discrete(N).  The agent picks a target index 0…N-1.
    The env schedules the next available event for that target.
    ``info["action_mask"]`` is a boolean array of shape (N,).

Observation space
-----------------
Dict with two Box spaces:
    "events"  Box(shape=(K_or_N, n_event_features), dtype=float32)
    "global"  Box(shape=(n_global_features,),        dtype=float32)

Reward
------
Per-step reward is computed by ``rewards.compute_reward`` and includes:

* Sparse tier-completion bonuses (T1=1, T2=3, T3=10, scaled by science_weight × diversity_mult)
* Dense progress shaping (proportional to Δprogress_in_tier; 3× boost when near a tier boundary)
* Dense efficiency reward (obs_duration / total_cost; penalises long slews)
* Missed-event penalty (if agent arrives after window_end)

In addition, ``check_milestone_reward`` fires one-shot bonuses when T1 coverage
crosses 25/50/75/90/100 % of the catalogue, and ``compute_terminal_reward`` fires
a quadratic end-of-episode bonus based on final T1 coverage fraction.

The ``info`` dict always includes the raw ``step_result`` from
``execute_observation`` (tier changes, slew cost, etc.) for external analysis.

Configuration
-------------
Pass an EnvConfig (or a path to a YAML) to the constructor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from ariel_rl.data.preprocess_targets import build_target_table
from ariel_rl.envs.action_mask import any_valid, compute_mask
from ariel_rl.envs.observation_builder import build as build_obs, observation_shapes
from ariel_rl.simulator.event_backend import EventBackend, TableBackend
from ariel_rl.simulator.event_generator import generate_events
from ariel_rl.simulator.mission_state import MissionState
from ariel_rl.simulator.slew import SLEW_RATE_DEG_PER_MIN, MIN_SLEW_S, MAX_SLEW_S
from ariel_rl.rewards.compute_reward import (
    compute_reward,
    check_milestone_reward,
    compute_terminal_reward,
)
from ariel_rl.utils.config import (
    EnvConfig,
    default_env_config,
    load_env_config,
)


class ArielEnv(gym.Env):
    """Gymnasium environment for Ariel exoplanet target scheduling.

    Parameters
    ----------
    config:
        An ``EnvConfig`` instance, a path to a YAML config file, or ``None``
        for all defaults.
    csv_path:
        Path to the raw MCS CSV.  Only used when *targets* is not provided.
    targets:
        Pre-built target DataFrame (skips CSV loading if provided).
    events:
        Pre-built event DataFrame for the ``TableBackend``.  Ignored when
        *backend* is provided explicitly.
    backend:
        Optional pre-constructed ``EventBackend`` instance.  Use this to
        select ``DynamicBackend`` without a pre-computed event table::

            from ariel_rl.simulator.event_backend import DynamicBackend
            env = ArielEnv(config, targets=targets,
                           backend=DynamicBackend(targets))

        When *None*, a ``TableBackend`` is created automatically from *events*
        (or a freshly generated event table if *events* is also ``None``).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | str | Path | None = None,
        csv_path: str | Path | None = None,
        targets: Optional[pd.DataFrame] = None,
        events: Optional[pd.DataFrame] = None,
        backend: Optional[EventBackend] = None,
    ) -> None:
        super().__init__()

        # ---- config ----
        if config is None:
            self.cfg = default_env_config()
        elif isinstance(config, (str, Path)):
            self.cfg = load_env_config(config)
        else:
            self.cfg = config

        self._slew_rate = self.cfg.slew.rate_deg_per_min
        self._min_slew_s = self.cfg.slew.min_slew_seconds
        self._max_slew_s = self.cfg.slew.max_slew_seconds

        # ---- static tables ----
        if targets is not None:
            self._targets = targets.copy()
        else:
            self._targets = build_target_table(csv_path)

        # Apply global max_tier_cap: clip each target's max_tier downward.
        cap = self.cfg.mission.max_tier_cap
        if cap < 3 and "max_tier" in self._targets.columns:
            self._targets["max_tier"] = self._targets["max_tier"].clip(upper=cap)

        # ---- event backend ----
        if backend is not None:
            # Explicit backend supplied — use it directly.
            self._backend: EventBackend = backend
            self._events: pd.DataFrame = events if events is not None else pd.DataFrame()
        else:
            # Default: TableBackend wrapping a pre-computed (or freshly generated) table.
            if events is not None:
                self._events = events
            else:
                self._events = generate_events(
                    self._targets,
                    mission_start=self.cfg.mission.start_bjd,
                    mission_end=self.cfg.mission.start_bjd + self.cfg.mission.lifetime_days,
                )
            self._backend = TableBackend(self._events)

        # ---- determine action space size ----
        if self.cfg.action.type == "topk":
            self._n_actions = self.cfg.action.topk.k
        elif self.cfg.action.type == "target":
            if isinstance(self._backend, TableBackend):
                self._n_actions = len(self._targets)
            else:
                raise ValueError(
                    "The 'target' action space type requires TableBackend. "
                    "Use action.type='topk' with DynamicBackend."
                )
        else:
            raise ValueError(f"Unknown action type: {self.cfg.action.type!r}")

        # ---- bootstrap a dummy state to measure observation shapes ----
        _dummy_state = MissionState.from_tables(
            self._targets,
            self._events,
            mission_start=self.cfg.mission.start_bjd,
            mission_end=self.cfg.mission.start_bjd + self.cfg.mission.lifetime_days,
            backend=self._backend,
        )
        shapes = observation_shapes(_dummy_state, self.cfg.observation, self._n_actions)

        # ---- Gymnasium spaces ----
        self.observation_space = spaces.Dict({
            "events": spaces.Box(
                low=-3.0, high=3.0,
                shape=shapes["events"],
                dtype=np.float32,
            ),
            "global": spaces.Box(
                low=0.0, high=1.0,
                shape=shapes["global"],
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.Discrete(self._n_actions)

        # ---- episode state (initialised in reset) ----
        self._state: Optional[MissionState] = None
        self._candidates: Optional[pd.DataFrame] = None
        self._action_mask: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._milestones_hit: set[float] = set()   # tracks one-shot T1-coverage bonuses

        # ---- relative reward mode: baseline trajectory ----
        self._baseline_traj: dict = {}
        if self.cfg.reward.reward_mode == "relative":
            traj_path_str = self.cfg.reward.baseline_trajectory_path
            if not traj_path_str:
                raise ValueError(
                    "reward_mode='relative' requires baseline_trajectory_path to be set. "
                    "Run scripts/generate_baseline_trajectory.py first, then point "
                    "baseline_trajectory_path at the resulting JSON."
                )
            traj_path = Path(traj_path_str)
            if not traj_path.exists():
                raise FileNotFoundError(
                    f"Baseline trajectory not found: {traj_path}. "
                    "Run scripts/generate_baseline_trajectory.py to generate it."
                )
            with open(traj_path) as _f:
                self._baseline_traj = json.load(_f)

        # Relative reward episode accumulators (initialised in reset)
        self._rel_interval_acc: float = 0.0
        self._rel_total_acc: float = 0.0
        self._rel_comparison_idx: int = 0
        self._rel_compound_idx: int = 0
        self._next_comparison_bjd: float = 0.0
        self._next_compound_bjd: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        self._backend.reset()
        self._state = MissionState.from_tables(
            self._targets,
            self._events,
            mission_start=self.cfg.mission.start_bjd,
            mission_end=self.cfg.mission.start_bjd + self.cfg.mission.lifetime_days,
            backend=self._backend,
        )
        self._step_count = 0
        self._milestones_hit = set()

        # Reset relative-reward accumulators
        self._rel_interval_acc = 0.0
        self._rel_total_acc = 0.0
        self._rel_comparison_idx = 0
        self._rel_compound_idx = 0
        self._next_comparison_bjd = (
            self.cfg.mission.start_bjd + self.cfg.reward.comparison_interval_days
        )
        self._next_compound_bjd = (
            self.cfg.mission.start_bjd + self.cfg.reward.compound_interval_days
        )

        self._candidates, self._action_mask = self._get_candidates_and_mask()

        obs = build_obs(self._state, self._candidates, self.cfg.observation)
        info = self._make_info(step_result=None)
        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self._state is not None, "Call reset() before step()."

        mask = self._action_mask
        if not mask[action]:
            # Invalid action — penalise and don't advance the clock
            penalty = -self.cfg.reward.invalid_action_penalty
            obs = build_obs(self._state, self._candidates, self.cfg.observation)
            info = self._make_info(step_result=None)
            info["invalid_action"] = True
            info["abs_reward"] = penalty
            return obs, penalty, False, False, info

        # Map action index → event_id
        event_id = self._action_to_event_id(action)

        # Execute the observation in the simulator
        step_result = self._state.execute_observation(event_id)
        self._step_count += 1

        # ---- compute full absolute reward for this step ----
        abs_reward = self._compute_reward(step_result)

        # One-shot milestone bonus (fires when a T1 coverage threshold is crossed)
        n_total = self._state.total_targets
        milestone_bonus, self._milestones_hit = check_milestone_reward(
            tier1_completed=self._state.tier1_completed,
            total_reachable=n_total,
            milestones_hit=self._milestones_hit,
            cfg=self.cfg.reward,
        )
        abs_reward += milestone_bonus

        # Check episode termination
        terminated = self._state.is_done()
        self._candidates, self._action_mask = self._get_candidates_and_mask()

        # If no valid actions remain, try advancing the window by looking further
        # ahead (up to _MAX_SKIP_ATTEMPTS) rather than terminating immediately.
        if not terminated and not any_valid(self._action_mask):
            self._candidates, self._action_mask = self._skip_to_next_feasible()

        if not terminated and not any_valid(self._action_mask):
            terminated = True

        # Terminal bonus: fired once at episode end
        if terminated:
            abs_reward += compute_terminal_reward(
                tier1_completed=self._state.tier1_completed,
                total_reachable=n_total,
                cfg=self.cfg.reward,
            )

        # ---- apply reward mode ----
        if self.cfg.reward.reward_mode == "relative":
            reward = self._apply_relative_reward(abs_reward, terminated)
        else:
            reward = abs_reward

        obs = build_obs(self._state, self._candidates, self.cfg.observation)
        info = self._make_info(step_result=step_result)
        info["abs_reward"] = abs_reward
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Candidate selection helpers
    # ------------------------------------------------------------------

    def _get_candidates_and_mask(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Return (candidate_events, action_mask) for the current state."""
        if self.cfg.action.type == "topk":
            return self._candidates_topk()
        elif self.cfg.action.type == "target":
            return self._candidates_target()
        else:
            raise ValueError(f"Unknown action type: {self.cfg.action.type!r}")

    def _candidates_topk(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Top-K upcoming events, delegated to the active EventBackend."""
        k = self.cfg.action.topk.k
        t_now = self._state.clock.current_time

        candidates = self._backend.candidates(t_now, k)

        # Pad to exactly K rows with zero/invalid dummies if the backend
        # returned fewer candidates than requested (e.g. near end of mission).
        if len(candidates) < k:
            cols = candidates.columns if len(candidates) else pd.Index(
                ["event_id", "target_id", "event_type",
                 "window_start", "window_mid", "window_end",
                 "duration", "duration_days", "tier_goal",
                 "base_science_value", "visibility_valid",
                 "ephemeris_uncertainty", "event_index"]
            )
            padding = _make_padding_rows(k - len(candidates), cols)
            candidates = pd.concat([candidates, padding], ignore_index=True)

        mask = compute_mask(self._state, candidates, self.cfg.action)
        return candidates.reset_index(drop=True), mask

    def _candidates_target(self) -> tuple[pd.DataFrame, np.ndarray]:
        """One next-event per target, ordered to match target table index."""
        rows = []
        for _, trow in self._targets.iterrows():
            tid = trow["target_id"]
            nxt = self._state.next_event_for_target(tid)
            if nxt is not None:
                rows.append(nxt.to_dict())
            else:
                rows.append(_sentinel_event(tid, self._events.columns))

        candidates = pd.DataFrame(rows)
        mask = compute_mask(self._state, candidates, self.cfg.action)
        return candidates.reset_index(drop=True), mask

    def _skip_to_next_feasible(
        self, max_lookahead: int = 10
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """When the current k candidates are all infeasible, look further ahead.

        Progressively asks the backend for larger candidate windows until a
        valid action is found or ``max_lookahead`` attempts are exhausted.
        The clock does NOT advance here — it advances only when the agent
        executes the chosen action via ``execute_observation``.
        """
        k = self.cfg.action.topk.k
        t_now = self._state.clock.current_time

        for multiplier in range(2, max_lookahead + 2):
            bigger_k = k * multiplier
            candidates = self._backend.candidates(t_now, bigger_k)
            if len(candidates) == 0:
                break
            # Pad to bigger_k for mask computation
            if len(candidates) < bigger_k:
                cols = candidates.columns
                padding = _make_padding_rows(bigger_k - len(candidates), cols)
                candidates = pd.concat([candidates, padding], ignore_index=True)

            from ariel_rl.envs.action_mask import compute_mask
            mask = compute_mask(self._state, candidates, self.cfg.action)

            if mask.any():
                # Found feasible events — trim back to k, keeping the valid ones first
                valid_idx = np.where(mask)[0][:k]
                invalid_idx = np.where(~mask)[0]
                keep = np.concatenate([valid_idx, invalid_idx])[:k]
                candidates = candidates.iloc[keep].reset_index(drop=True)
                mask = mask[keep]
                # Pad back to k if needed
                if len(candidates) < k:
                    cols = candidates.columns
                    padding = _make_padding_rows(k - len(candidates), cols)
                    candidates = pd.concat([candidates, padding], ignore_index=True)
                    mask = np.concatenate([mask, np.zeros(k - len(mask), dtype=bool)])
                return candidates, mask

        # Nothing found — return current (all invalid) candidates
        return self._candidates, self._action_mask

    def _action_to_event_id(self, action: int) -> int:
        """Convert an action index to an event_id in the event table."""
        if self.cfg.action.type in ("topk", "target"):
            row = self._candidates.iloc[action]
            return int(row["event_id"])
        raise ValueError(f"Unknown action type: {self.cfg.action.type!r}")

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, step_result: dict) -> float:
        """Compute the per-step reward for the current step using the rewards module."""
        if step_result is None:
            return 0.0
        return compute_reward(
            step_result=step_result,
            cfg=self.cfg.reward,
            bin_totals=self._state._bin_totals,
            bin_observed=self._state.population_bin_counts,
        )

    def _apply_relative_reward(self, abs_reward: float, terminated: bool) -> float:
        """Convert an absolute reward into a checkpoint-based relative reward.

        Accumulates ``abs_reward`` internally and emits rewards only at two
        types of mission-time checkpoints:

        * **Comparison intervals** (every ``comparison_interval_days``):
          ``comparison_scale × (agent_interval_acc − baseline_interval_mean)``
          Measures how much better the agent did in this short window compared
          to the baseline.

        * **Compound checkpoints** (every ``compound_interval_days``):
          ``compound_scale × (agent_total_acc − baseline_cumulative_at_checkpoint)``
          Measures cumulative advantage over the baseline so far — compounds as
          the agent consistently outperforms.

        At episode termination any remaining partial comparison interval is
        flushed so the agent always receives a signal for the final stretch.
        """
        cfg = self.cfg.reward
        traj = self._baseline_traj

        self._rel_interval_acc += abs_reward
        self._rel_total_acc += abs_reward

        reward = 0.0
        t_now = self._state.clock.current_time

        interval_rewards: list = traj.get("interval_rewards", [])
        compound_rewards: list = traj.get("compound_cumulative_rewards", [])

        # ---- emit comparison intervals that the clock has crossed ----
        while t_now >= self._next_comparison_bjd:
            baseline_iv = (
                float(interval_rewards[self._rel_comparison_idx])
                if self._rel_comparison_idx < len(interval_rewards)
                else 0.0
            )
            reward += cfg.comparison_scale * (self._rel_interval_acc - baseline_iv)
            self._rel_interval_acc = 0.0
            self._rel_comparison_idx += 1
            self._next_comparison_bjd += cfg.comparison_interval_days

        # ---- emit compound checkpoints that the clock has crossed ----
        while t_now >= self._next_compound_bjd:
            baseline_cum = (
                float(compound_rewards[self._rel_compound_idx])
                if self._rel_compound_idx < len(compound_rewards)
                else float(traj.get("total_mean_reward", 0.0))
            )
            reward += cfg.compound_scale * (self._rel_total_acc - baseline_cum)
            self._rel_compound_idx += 1
            self._next_compound_bjd += cfg.compound_interval_days

        # ---- at termination flush the remaining partial comparison interval ----
        if terminated and self._rel_interval_acc != 0.0:
            baseline_iv = (
                float(interval_rewards[self._rel_comparison_idx])
                if self._rel_comparison_idx < len(interval_rewards)
                else 0.0
            )
            reward += cfg.comparison_scale * (self._rel_interval_acc - baseline_iv)
            self._rel_interval_acc = 0.0

        return reward

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _make_info(self, step_result: Optional[dict]) -> dict:
        info: dict[str, Any] = {
            "action_mask":    self._action_mask,
            "step_count":     self._step_count,
            "mission_summary": self._state.summary() if self._state else {},
            "invalid_action": False,
        }
        if step_result is not None:
            info["step_result"] = step_result
        return info

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def state(self) -> Optional[MissionState]:
        """Direct access to the simulator state (useful for debugging)."""
        return self._state

    @property
    def action_mask(self) -> Optional[np.ndarray]:
        return self._action_mask


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_padding_rows(n: int, columns: pd.Index) -> pd.DataFrame:
    """Create n zero-filled dummy rows to pad the candidate table to K."""
    dummy = {col: [0] * n for col in columns}
    dummy["visibility_valid"] = [False] * n
    dummy["event_id"] = [-1] * n
    dummy["target_id"] = [""] * n
    dummy["window_end"] = [0.0] * n
    dummy["duration_days"] = [0.0] * n
    dummy["duration"] = [0.0] * n
    return pd.DataFrame(dummy)[columns]


def _sentinel_event(target_id: str, columns: pd.Index) -> dict:
    """A dummy event row for a target with no upcoming events."""
    row = {col: 0 for col in columns}
    row["target_id"] = target_id
    row["event_id"] = -1
    row["visibility_valid"] = False
    row["window_end"] = 0.0
    row["duration_days"] = 0.0
    row["duration"] = 0.0
    return row
