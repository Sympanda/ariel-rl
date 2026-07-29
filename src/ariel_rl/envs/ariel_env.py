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
from ariel_rl.envs.planet_feature_builder import (
    build_planet_features,
    build_static_features,
    N_PLANET_FEATURES,
    PLANET_FEATURE_NAMES,
)
from ariel_rl.simulator.event_backend import EventBackend, DynamicBackend
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
        Unused — retained for backward-compatibility only.  The environment
        now defaults to ``DynamicBackend`` which computes events on-the-fly.
    backend:
        Optional pre-constructed ``EventBackend`` instance.  Defaults to
        ``DynamicBackend(targets)`` when *None*.
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
            self._targets = build_target_table(
                csv_path,
                science_weight_floor=self.cfg.reward.science_weight_floor,
            )

        # Apply global max_tier_cap: clip each target's max_tier downward.
        cap = self.cfg.mission.max_tier_cap
        if cap < 3 and "max_tier" in self._targets.columns:
            self._targets["max_tier"] = self._targets["max_tier"].clip(upper=cap)

        # ---- event backend ----
        self._events: pd.DataFrame = events if events is not None else pd.DataFrame()
        if backend is not None:
            self._backend: EventBackend = backend
        else:
            self._backend = DynamicBackend(self._targets)

        # ---- determine action space size ----
        if self.cfg.action.type == "topk":
            self._n_actions = self.cfg.action.topk.k
        elif self.cfg.action.type == "target":
            self._n_actions = len(self._targets)
        elif self.cfg.action.type == "full_set":
            cfg_k_filter = self.cfg.action.full_set.k_filter
            cfg_n_max    = self.cfg.action.full_set.n_max
            if cfg_k_filter > 0:
                # k_filter: only top-K planets reach the policy → action space = K
                self._n_actions = cfg_k_filter
            elif cfg_n_max > 0:
                # n_max is a hard ceiling — the catalogue must fit inside it.
                if len(self._targets) > cfg_n_max:
                    raise ValueError(
                        f"Catalogue has {len(self._targets)} targets but "
                        f"action.full_set.n_max={cfg_n_max}.  "
                        "Increase n_max (or reduce the catalogue)."
                    )
                self._n_actions = cfg_n_max
            else:
                # n_max=0 → use catalogue size (backward-compatible default)
                self._n_actions = len(self._targets)
        else:
            raise ValueError(f"Unknown action type: {self.cfg.action.type!r}")

        # Number of actual catalogue targets (may be < _n_actions when padding active)
        self._n_targets = len(self._targets)

        # Dynamic active-planet set (full_set mode only; all targets start active).
        # Maintained as an ordered list so action_index i maps to _active_target_ids[i].
        # Populated in reset(); updated after each step when targets complete.
        self._active_target_ids: list[str] = []
        # Reverse index: target_id → position in _active_target_ids (for O(1) lookup)
        self._active_tid_to_idx: dict[str, int] = {}
        # target_id → row-index in the full static-feature cache (built at reset)
        self._tid_to_cache_idx: dict[str, int] = {}

        # ---- static per-planet feature cache (full_set mode) ----
        self._static_planet_features: np.ndarray | None = None

        # ---- bootstrap a dummy state to measure observation shapes ----
        _dummy_state = MissionState.from_backend(
            self._targets,
            backend=self._backend,
            mission_start=self.cfg.mission.start_bjd,
            mission_end=self.cfg.mission.start_bjd + self.cfg.mission.lifetime_days,
        )

        # ---- Gymnasium spaces ----
        if self.cfg.action.type == "full_set":
            # Observation: per-planet features (N × F) + global
            shapes = observation_shapes(_dummy_state, self.cfg.observation, self._n_actions)
            self.observation_space = spaces.Dict({
                "planets": spaces.Box(
                    low=-3.0, high=3.0,
                    shape=(self._n_actions, N_PLANET_FEATURES),
                    dtype=np.float32,
                ),
                "global": spaces.Box(
                    low=0.0, high=1.0,
                    shape=shapes["global"],
                    dtype=np.float32,
                ),
            })
        else:
            shapes = observation_shapes(_dummy_state, self.cfg.observation, self._n_actions)
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
        self._state = MissionState.from_backend(
            self._targets,
            backend=self._backend,
            mission_start=self.cfg.mission.start_bjd,
            mission_end=self.cfg.mission.start_bjd + self.cfg.mission.lifetime_days,
            overhead_days_per_obs=self.cfg.mission.overhead_days_per_obs,
        )
        self._step_count = 0
        self._milestones_hit = set()

        if self.cfg.action.type == "full_set":
            # Initialise dynamic active set — at reset all targets begin at tier 0
            # so every target is active.
            self._active_target_ids = list(self._targets["target_id"].astype(str))
            self._active_tid_to_idx = {tid: i for i, tid in enumerate(self._active_target_ids)}

            # Pre-compute static planet features (time-invariant for the episode).
            if self.cfg.action.full_set.cache_static:
                self._static_planet_features = build_static_features(self._state)
                # Build reverse index: target_id → row in the (N, n_static) cache
                all_tids = list(self._targets["target_id"].astype(str))
                self._tid_to_cache_idx = {tid: i for i, tid in enumerate(all_tids)}

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

        obs = self._build_observation()
        info = self._make_info(step_result=None)
        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self._state is not None, "Call reset() before step()."

        mask = self._action_mask
        if not mask[action]:
            # Invalid action — penalise and don't advance the clock
            penalty = -self.cfg.reward.invalid_action_penalty
            obs = self._build_observation()
            info = self._make_info(step_result=None)
            info["invalid_action"] = True
            info["abs_reward"] = penalty
            return obs, penalty, False, False, info

        # Map action index → event_id
        event_id = self._action_to_event_id(action)

        # Snapshot per-bin and per-host counts BEFORE executing the observation
        # (needed to compute the marginal coverage and host-diversity rewards).
        bin_observed_before = dict(self._state.population_bin_counts)
        host_tier1_before   = self._host_tier1_counts()

        # Execute the observation in the simulator
        step_result = self._state.execute_observation(event_id)
        self._step_count += 1

        # ---- compute full absolute reward for this step ----
        abs_reward = self._compute_reward(step_result, bin_observed_before, host_tier1_before)

        # One-shot milestone bonus (fires when a T1 coverage threshold is crossed)
        n_total = self._state.total_targets
        milestone_bonus, self._milestones_hit = check_milestone_reward(
            tier1_completed=self._state.tier1_completed,
            total_reachable=n_total,
            milestones_hit=self._milestones_hit,
            cfg=self.cfg.reward,
        )
        abs_reward += milestone_bonus

        # Update dynamic active set: remove planets that just completed max_tier.
        if self.cfg.action.type == "full_set":
            self._update_active_set()

        # Check episode termination
        terminated = self._state.is_done()
        self._candidates, self._action_mask = self._get_candidates_and_mask()

        # If no valid actions remain, try to recover a feasible action set.
        # For topk: ask the backend for a larger window (the K nearest events
        #   may all be expired; looking further ahead usually finds a valid one).
        # For target/full_set: candidates are fixed (one per target) — if none
        #   are valid the episode terminates; no lookahead is attempted.
        if not terminated and not any_valid(self._action_mask):
            if self.cfg.action.type == "topk":
                self._candidates, self._action_mask = self._skip_to_next_feasible_topk()

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

        obs = self._build_observation()
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
        elif self.cfg.action.type == "full_set":
            return self._candidates_full_set()
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
                 "duration", "duration_days", "block_duration_days", "tier_goal",
                 "base_science_value", "visibility_valid",
                 "ephemeris_uncertainty", "event_index"]
            )
            padding = _make_padding_rows(k - len(candidates), cols)
            candidates = pd.concat([candidates, padding], ignore_index=True)

        mask = compute_mask(self._state, candidates, self.cfg.action)
        return candidates.reset_index(drop=True), mask

    # ------------------------------------------------------------------
    # Dynamic active set management (full_set mode)
    # ------------------------------------------------------------------

    def _update_active_set(self) -> None:
        """Remove targets that have reached max_tier from the active set.

        Called after each observation step in ``full_set`` mode.  After removal
        the active set index is rebuilt so that ``_active_tid_to_idx`` remains
        consistent.
        """
        if not self._active_target_ids:
            return
        new_active: list[str] = []
        for tid in self._active_target_ids:
            prog   = self._state._progress_dict.get(tid)
            target = self._state._target_lookup.get(tid)
            if prog is None or target is None:
                continue
            if int(prog["current_tier"]) < int(target["max_tier"]):
                new_active.append(tid)
        self._active_target_ids = new_active
        self._active_tid_to_idx = {tid: i for i, tid in enumerate(new_active)}

    # ------------------------------------------------------------------
    # Candidate selection helpers (continued)
    # ------------------------------------------------------------------

    def _candidates_full_set(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Full-set mode: one *first-reachable* event per active target + padding.

        Each active planet token is associated with the first upcoming event
        that the telescope can reach from its current position (i.e.
        ``t_now + slew < block_end``).  If the current block has expired or
        the slew would miss it, we look further ahead — so the agent sees a
        real, meaningful action for every active planet rather than a
        routinely-masked stale event.

        Candidate row ordering matches ``_active_target_ids`` so that
        ``action_index i → _active_target_ids[i]``.

        Padding rows (indices ``n_active … n_max-1``) are always masked False.

        Implementation note
        -------------------
        Previously this method ran two nested Python loops (one over 16 k pool
        rows via ``iterrows()``, one over all active planets).  It now uses:

        1. A vectorised NumPy slew computation over all active targets at once.
        2. A pandas merge → filter → groupby to find the first-reachable event
           per target — no Python loop at all for the common case.
        3. A small Python fallback loop only for the rare long-period targets
           not covered by the pool.

        This gives ~10–50× speedup on the env-step bottleneck.
        """
        from ariel_rl.simulator.slew import slew_time_days_vec
        from ariel_rl.data.schemas import COST_FACTOR

        t_now    = self._state.clock.current_time
        n_active = len(self._active_target_ids)

        _FALLBACK_COLS = pd.Index([
            "event_id", "target_id", "event_type",
            "window_start", "window_mid", "window_end",
            "duration", "duration_days", "block_duration_days", "tier_goal",
            "base_science_value", "visibility_valid",
            "ephemeris_uncertainty", "event_index",
        ])

        if n_active == 0:
            pad  = _make_padding_rows(self._n_actions, _FALLBACK_COLS)
            mask = np.zeros(self._n_actions, dtype=bool)
            return pad, mask

        # ------------------------------------------------------------------
        # Step 1: vectorised slew → t_arrive for every active target
        # ------------------------------------------------------------------
        active_tids = self._active_target_ids          # list[str], len = n_active
        target_rows = [self._state._target_lookup.get(tid) for tid in active_tids]
        valid_flags = np.array([r is not None for r in target_rows], dtype=bool)

        ra_arr  = np.array([float(r["ra"])  if r is not None else 0.0 for r in target_rows])
        dec_arr = np.array([float(r["dec"]) if r is not None else 0.0 for r in target_rows])

        slews     = slew_time_days_vec(
            self._state.current_ra, self._state.current_dec, ra_arr, dec_arr
        )
        t_arrives = t_now + slews          # shape (n_active,)

        # ------------------------------------------------------------------
        # Step 2: populate the backend step-cache with a large pool
        # ------------------------------------------------------------------
        pool_size = max(n_active * 20, 200)
        pool_df   = self._backend.candidates(t_now, pool_size)

        fallback_cols = pool_df.columns if len(pool_df) > 0 else _FALLBACK_COLS

        # ------------------------------------------------------------------
        # Step 3: vectorised first-reachable-event selection via merge+groupby
        # ------------------------------------------------------------------
        best_dict: dict[str, dict] = {}   # tid → best event row dict

        if len(pool_df) > 0:
            # Build a t_arrive lookup table (one row per active target)
            t_arr_df = pd.DataFrame({
                "target_id": active_tids,
                "t_arrive":  t_arrives,
            })

            # Add block_end to pool (vectorised)
            p = pool_df.copy()
            p["block_end"] = (
                p["window_mid"].to_numpy(float)
                + p["block_duration_days"].to_numpy(float) / 2.0
            )
            p["target_id"] = p["target_id"].astype(str)

            # Merge: each pool event gets its target's t_arrive
            p = p.merge(t_arr_df, on="target_id", how="inner")

            # Filter to reachable events, then take the earliest per target
            reachable = p[p["block_end"] > p["t_arrive"]]
            if len(reachable) > 0:
                best_rows = (
                    reachable
                    .sort_values("window_mid")
                    .groupby("target_id", sort=False)
                    .first()
                    .reset_index()
                )
                # to_dict('records') is ~8× faster than iterrows() + to_dict()
                for rec in best_rows.to_dict("records"):
                    best_dict[str(rec["target_id"])] = rec

        # ------------------------------------------------------------------
        # Step 4: fallback for long-period targets not found in the pool
        # ------------------------------------------------------------------
        tid_to_idx = {tid: i for i, tid in enumerate(active_tids)}
        for i, tid in enumerate(active_tids):
            if not valid_flags[i] or tid in best_dict:
                continue
            t_arrive = t_arrives[i]
            future = self._backend.events_for_target(tid, t_now, n=20)
            for fev in future:
                bd    = float(fev.get("block_duration_days", 0.0))
                wm    = float(fev.get("window_mid", 0.0))
                if wm + bd / 2.0 > t_arrive:
                    eid = self._backend.register_event({**fev, "target_id": tid})
                    if eid >= 0:
                        best_ev = {**fev, "event_id": eid, "target_id": tid}
                        best_ev.setdefault("tier_goal",            1)
                        best_ev.setdefault("base_science_value",   1.0)
                        best_ev.setdefault("visibility_valid",     True)
                        best_ev.setdefault("ephemeris_uncertainty", 0.0)
                        best_ev.setdefault("event_index",          -1)
                        best_ev.setdefault("duration",
                                           best_ev.get("duration_days", 0.0) * 86400.0)
                        best_dict[tid] = best_ev
                    break

        # ------------------------------------------------------------------
        # Step 5: assemble rows in _active_target_ids order → DataFrame
        # ------------------------------------------------------------------
        rows: list[dict] = []
        for i, tid in enumerate(active_tids):
            if not valid_flags[i]:
                rows.append(_sentinel_event(tid, fallback_cols))
            elif tid in best_dict:
                rows.append(best_dict[tid])
            else:
                rows.append(_sentinel_event(tid, fallback_cols))

        candidates = pd.DataFrame(rows)
        for col in fallback_cols:
            if col not in candidates.columns:
                candidates[col] = 0 if col != "target_id" else ""
        candidates = candidates.reindex(columns=fallback_cols, fill_value=0)

        # ------------------------------------------------------------------
        # Step 5.5 (optional): fast top-K pre-filter by event urgency
        # ------------------------------------------------------------------
        # When k_filter > 0, keep only the K planets whose first-reachable event
        # has the soonest window_mid — i.e. the K most urgent opportunities.
        # This is the same principle as top-K event mode: "these windows are
        # closing soon, so the agent must reason about them right now."
        # The ISAB then decides which of the K to actually observe.
        # Reduces ISAB token count from N_max (~814) → K for ~(N/K)× GPU speedup.
        k_filter = self.cfg.action.full_set.k_filter
        if k_filter > 0 and len(candidates) > k_filter:
            wm = candidates["window_mid"].to_numpy(float)
            # argpartition finds the K smallest window_mids in O(N)
            top_k_idx = np.argpartition(wm, k_filter)[:k_filter]
            # Sort ascending so slot 0 = most urgent
            top_k_idx = top_k_idx[np.argsort(wm[top_k_idx])]
            candidates = candidates.iloc[top_k_idx].reset_index(drop=True)

        # ------------------------------------------------------------------
        # Step 6: mask + pad to _n_actions
        # ------------------------------------------------------------------
        mask  = compute_mask(self._state, candidates, self.cfg.action)
        n_real = len(candidates)
        if n_real < self._n_actions:
            extra = _make_padding_rows(self._n_actions - n_real, candidates.columns)
            candidates = pd.concat([candidates, extra], ignore_index=True)
            mask = np.concatenate([mask, np.zeros(self._n_actions - n_real, dtype=bool)])

        return candidates.reset_index(drop=True), mask

    def _candidates_target(self) -> tuple[pd.DataFrame, np.ndarray]:
        """One next-event per target, ordered to match target table index.

        Works with both TableBackend (queries self.events) and DynamicBackend
        (calls backend.candidates for a large window, then picks first per target).
        """
        from ariel_rl.simulator.event_backend import TableBackend
        t_now = self._state.clock.current_time
        n_targets = len(self._targets)

        if isinstance(self._backend, TableBackend) and len(self._events) > 0:
            # Original table-based path
            rows = []
            fallback_cols = self._events.columns
            for _, trow in self._targets.iterrows():
                tid = trow["target_id"]
                nxt = self._state.next_event_for_target(tid)
                if nxt is not None:
                    rows.append(nxt.to_dict())
                else:
                    rows.append(_sentinel_event(tid, fallback_cols))
        else:
            # DynamicBackend path: fetch a large window and pick first per target
            # Fetch 3× the target count to ensure coverage of all targets
            pool = self._backend.candidates(t_now, n_targets * 3)
            # Build target_id → first upcoming event mapping
            seen: dict[str, dict] = {}
            for _, ev in pool.iterrows():
                tid = ev["target_id"]
                if tid not in seen:
                    seen[tid] = ev.to_dict()

            fallback_cols = pd.Index(
                ["event_id", "target_id", "event_type",
                 "window_start", "window_mid", "window_end",
                 "duration", "duration_days", "block_duration_days", "tier_goal",
                 "base_science_value", "visibility_valid",
                 "ephemeris_uncertainty", "event_index"]
            )
            rows = []
            for _, trow in self._targets.iterrows():
                tid = trow["target_id"]
                if tid in seen:
                    rows.append(seen[tid])
                else:
                    rows.append(_sentinel_event(tid, fallback_cols))

        candidates = pd.DataFrame(rows)
        mask = compute_mask(self._state, candidates, self.cfg.action)
        return candidates.reset_index(drop=True), mask

    def _skip_to_next_feasible_topk(
        self, max_lookahead: int = 10
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Top-K mode only: look further ahead when all K candidates are expired.

        Progressively asks the backend for larger candidate windows until a
        valid action is found or ``max_lookahead`` attempts are exhausted.
        The clock does NOT advance here — it advances only when the agent
        executes the chosen action via ``execute_observation``.

        This method must NOT be called in ``target`` or ``full_set`` modes
        because those modes return exactly one candidate per target and there
        is no "larger window" concept.  An all-invalid mask in those modes
        means the episode should terminate.
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
        if self.cfg.action.type in ("topk", "target", "full_set"):
            row = self._candidates.iloc[action]
            return int(row["event_id"])
        raise ValueError(f"Unknown action type: {self.cfg.action.type!r}")

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _build_observation(self) -> dict:
        """Build the agent observation dict for the current state.

        In ``full_set`` mode the observation contains ``"planets"`` (N × F)
        and ``"global"``.  In ``topk`` / ``target`` modes it contains
        ``"events"`` (K × E) and ``"global"``.
        """
        from ariel_rl.envs.observation_builder import _build_global
        if self.cfg.action.type == "full_set":
            # When k_filter is active, _candidates already contains only the
            # top-K filtered rows (real + padding).  Derive active_ids from
            # those rows so that the planet feature array matches the reduced
            # observation space shape (_n_actions = k_filter, not N_max).
            # Without k_filter, use the full _active_target_ids list as before.
            _k_filter = self.cfg.action.full_set.k_filter
            if _k_filter > 0 and self._candidates is not None and len(self._candidates) > 0:
                active_ids = [
                    str(r["target_id"])
                    for r in self._candidates.to_dict("records")
                    if r.get("target_id") and str(r["target_id"]) not in ("", "0")
                ]
            else:
                active_ids = self._active_target_ids   # full set (no pre-filter)

            # Build target_id → event dict from the pre-computed candidates
            # (candidates are in active-target order so rows 0…n_active-1 are real).
            per_target_events: dict[str, dict] | None = None
            if self._candidates is not None and len(self._candidates) > 0:
                n_real = len(active_ids)
                # Only look at real (non-padding) rows; to_dict('records') is
                # ~8× faster than iterrows() + to_dict() on 814-row DataFrames.
                per_target_events = {
                    str(rec["target_id"]): rec
                    for rec in self._candidates.iloc[:n_real].to_dict("records")
                    if rec.get("target_id")
                }

            # Prepare correctly-shaped static feature slice for active targets only.
            static_for_active: np.ndarray | None = None
            if self._static_planet_features is not None and self._tid_to_cache_idx:
                indices = [
                    self._tid_to_cache_idx[tid]
                    for tid in active_ids
                    if tid in self._tid_to_cache_idx
                ]
                if indices:
                    static_for_active = self._static_planet_features[indices]

            if active_ids:
                planet_arr = build_planet_features(
                    self._state,
                    static_features=static_for_active,
                    per_target_events=per_target_events,
                    target_ids=active_ids,
                )
            else:
                planet_arr = np.zeros((0, N_PLANET_FEATURES), dtype=np.float32)

            # Pad to _n_actions rows with zeros (padding positions)
            n_real = planet_arr.shape[0]
            if n_real < self._n_actions:
                padding = np.zeros(
                    (self._n_actions - n_real, N_PLANET_FEATURES), dtype=np.float32
                )
                planet_arr = np.concatenate([planet_arr, padding], axis=0)
            global_arr = _build_global(self._state, self.cfg.observation)
            return {"planets": planet_arr, "global": global_arr}
        return build_obs(self._state, self._candidates, self.cfg.observation)

    def _compute_reward(
        self,
        step_result: dict,
        bin_observed_before: dict[str, int] | None = None,
        host_tier1_before: dict[str, int] | None = None,
    ) -> float:
        """Compute the per-step reward for the current step using the rewards module."""
        if step_result is None:
            return 0.0
        bin_observed_after = self._state.population_bin_counts
        return compute_reward(
            step_result=step_result,
            cfg=self.cfg.reward,
            bin_totals=self._state._bin_totals,
            bin_observed_before=bin_observed_before or bin_observed_after,
            bin_observed_after=bin_observed_after,
            host_tier1_counts=host_tier1_before,
        )

    def _host_tier1_counts(self) -> dict[str, int]:
        """Return a dict of {host_id: n_tier1_completed_targets} for the current state."""
        counts: dict[str, int] = {}
        if "host_id" not in self._targets.columns:
            return counts
        for tid, prog in self._state._progress_dict.items():
            if int(prog.get("current_tier", 0)) >= 1:
                target_row = self._state._target_lookup.get(tid)
                if target_row is not None:
                    hid = str(target_row.get("host_id", ""))
                    if hid:
                        counts[hid] = counts.get(hid, 0) + 1
        return counts

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
    dummy["window_mid"] = [0.0] * n
    dummy["duration_days"] = [0.0] * n
    dummy["block_duration_days"] = [0.0] * n
    dummy["duration"] = [0.0] * n
    return pd.DataFrame(dummy)[columns]


def _sentinel_event(target_id: str, columns: pd.Index) -> dict:
    """A dummy event row for a target with no upcoming events."""
    row = {col: 0 for col in columns}
    row["target_id"] = target_id
    row["event_id"] = -1
    row["visibility_valid"] = False
    row["window_end"] = 0.0
    row["window_mid"] = 0.0
    row["duration_days"] = 0.0
    row["block_duration_days"] = 0.0
    row["duration"] = 0.0
    return row
