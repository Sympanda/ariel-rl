"""
MissionState: the single mutable object that the environment and agent act on.

It holds:
  - The static target and event tables (read-only)
  - A per-target progress table (mutable)
  - The mission clock
  - A pointer to the "current position" in the sky (for slew costs)
  - Summary statistics used for reward and observation construction

Typical episode flow
--------------------
    state = MissionState.from_tables(targets, events)
    while not state.is_done():
        obs = observation_builder.build(state, candidate_events)
        action = agent.act(obs)
        reward, info = state.execute_action(action)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ariel_rl.data.observation_requirements import compute_progress, initialise_progress_table
from ariel_rl.data.schemas import (
    COST_FACTOR,
    MISSION_END_BJD,
    MISSION_LIFETIME_DAYS,
    MISSION_START_BJD,
    TIER_1,
    TIER_2,
    TIER_3,
    TIER_NONE,
)
from ariel_rl.simulator.mission_clock import MissionClock
from ariel_rl.simulator.slew import slew_time_days


@dataclass
class MissionState:
    """Full mutable state of the Ariel mission.

    Parameters
    ----------
    targets:
        Processed target table (static, not modified).
    events:
        Full event table sorted by window_mid (static, not modified).
    clock:
        MissionClock instance (mutable).
    progress:
        Per-target progress DataFrame (mutable), indexed by target_id.
    current_ra:
        Current telescope pointing RA (degrees).
    current_dec:
        Current telescope pointing Dec (degrees).
    """

    targets: pd.DataFrame
    events: pd.DataFrame
    clock: MissionClock
    progress: pd.DataFrame                     # indexed by target_id
    current_ra: float = 0.0
    current_dec: float = 0.0

    # Observation history for diagnostics / Gantt chart
    obs_log: list = field(default_factory=list, repr=False)

    # Convenience index: target_id → row in targets
    _target_lookup: dict[str, pd.Series] = field(default_factory=dict, repr=False)

    # Pluggable event backend — provides candidates() and get_event().
    # Set by from_tables() or from_backend(); defaults to TableBackend.
    _backend: object = field(default=None, repr=False)

    # Normalisation constant: max tier-3 observations required across all targets
    _max_obs_rem_val: int = field(default=1, repr=False)

    # Cached population_bin_counts — invalidated when any tier-1 is completed
    _pop_bin_cache: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._target_lookup = {
            row["target_id"]: row for _, row in self.targets.iterrows()
        }
        if self._backend is None:
            from ariel_rl.simulator.event_backend import TableBackend
            self._backend = TableBackend(self.events)
        max_t3 = self.targets["tier3_required_obs"].max()
        self._max_obs_rem_val = int(max_t3) if pd.notna(max_t3) else 1
        # Plain-dict cache for progress rows — much faster than progress.loc
        self._progress_dict: dict = self.progress.to_dict(orient="index")
        # Static per-bin target counts for diversity reward computation
        self._bin_totals: dict[str, int] = (
            self.targets["population_bin"].value_counts().to_dict()
            if "population_bin" in self.targets.columns
            else {}
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_tables(
        cls,
        targets: pd.DataFrame,
        events: pd.DataFrame,
        mission_start: float = MISSION_START_BJD,
        mission_end: float = MISSION_END_BJD,
        backend=None,
    ) -> "MissionState":
        """Construct a fresh MissionState using a pre-computed event table.

        Parameters
        ----------
        backend:
            Optional pre-constructed ``EventBackend``.  When *None*, a
            ``TableBackend`` wrapping *events* is created automatically.
        """
        clock = MissionClock(mission_start=mission_start, mission_end=mission_end)
        progress = initialise_progress_table(targets)
        ra0 = float(targets["ra"].iloc[0]) if len(targets) else 0.0
        dec0 = float(targets["dec"].iloc[0]) if len(targets) else 0.0
        return cls(
            targets=targets,
            events=events,
            clock=clock,
            progress=progress,
            current_ra=ra0,
            current_dec=dec0,
            _backend=backend,
        )

    @classmethod
    def from_backend(
        cls,
        targets: pd.DataFrame,
        backend,
        mission_start: float = MISSION_START_BJD,
        mission_end: float = MISSION_END_BJD,
    ) -> "MissionState":
        """Construct a fresh MissionState using a ``DynamicBackend``.

        No pre-computed event table is needed; pass the backend directly.
        The ``events`` DataFrame stored on the state will be empty — methods
        that depend on the table (``events_for_target``, ``next_event_for_target``,
        ``upcoming_events``) are not available with this factory.
        """
        from ariel_rl.simulator.event_backend import _EMPTY_EVENTS
        clock = MissionClock(mission_start=mission_start, mission_end=mission_end)
        progress = initialise_progress_table(targets)
        ra0 = float(targets["ra"].iloc[0]) if len(targets) else 0.0
        dec0 = float(targets["dec"].iloc[0]) if len(targets) else 0.0
        return cls(
            targets=targets,
            events=_EMPTY_EVENTS.copy(),
            clock=clock,
            progress=progress,
            current_ra=ra0,
            current_dec=dec0,
            _backend=backend,
        )

    # ------------------------------------------------------------------
    # Core action: execute an observation
    # ------------------------------------------------------------------

    def execute_observation(
        self,
        event_id: int,
    ) -> dict:
        """Execute the observation for the given event_id.

        Advances the clock by (slew + obs_duration).
        Updates target progress.
        Returns an info dict.

        Parameters
        ----------
        event_id:
            Row in ``self.events`` to observe.

        Returns
        -------
        dict with keys:
            target_id, event_type, obs_duration_days, slew_days,
            total_cost_days, tier_before, tier_after, tier_completed,
            obs_number, missed (bool — arrived after window_end)
        """
        event = self._backend.get_event(event_id)
        target_id = event["target_id"]
        target = self._target_lookup[target_id]

        # ---- slew ----
        slew_days = slew_time_days(
            ra1=self.current_ra,
            dec1=self.current_dec,
            ra2=float(target["ra"]),
            dec2=float(target["dec"]),
        )

        # ---- check if we need to wait for window_start ----
        window_start = float(event["window_start"])
        if self.clock.current_time < window_start:
            # Jump ahead to window start (idle wait)
            self.clock.skip_to(window_start)

        # ---- check if we've missed the window ----
        window_end = float(event["window_end"])
        missed = (self.clock.current_time + slew_days) > window_end

        obs_duration_days = float(event["duration_days"])

        tier_before   = int(self._progress_dict[target_id]["current_tier"])
        progress_before = float(self._progress_dict[target_id]["progress_in_tier"])

        if missed:
            self.clock.record_miss()
            # Still pay the slew cost so the clock advances
            self.clock.advance(obs_duration_days=0.0, slew_days=slew_days)
        else:
            # Pay slew then observation
            self.clock.advance(obs_duration_days=obs_duration_days, slew_days=slew_days)
            # Update pointing
            self.current_ra = float(target["ra"])
            self.current_dec = float(target["dec"])
            # Update progress
            self._increment_progress(target_id, target)

        tier_after    = int(self._progress_dict[target_id]["current_tier"])
        progress_after = float(self._progress_dict[target_id]["progress_in_tier"])

        info = {
            "target_id":         target_id,
            "event_id":          event_id,
            "event_type":        event["event_type"],
            "obs_duration_days": obs_duration_days,
            "slew_days":         slew_days,
            "total_cost_days":   obs_duration_days + slew_days,
            "tier_before":       tier_before,
            "tier_after":        tier_after,
            "tier_completed":    tier_after > tier_before,
            "obs_number":        int(self._progress_dict[target_id]["obs_completed"]),
            "missed":            missed,
            # Additional context consumed by the reward function:
            "progress_before":   progress_before,
            "progress_after":    progress_after,
            "science_weight":    float(target.get("science_weight", 0.5)),
            "population_bin":    str(target.get("population_bin", "")),
            # Orbital period (days) — used by the rarity bonus in compute_reward.
            "period":            float(target.get("period", 0.0)),
        }

        # Append to observation log (used for Gantt / diagnostic plots)
        self.obs_log.append({
            "step":              len(self.obs_log),
            "mission_day":       float(event["window_mid"]) - self.clock.mission_start,
            "target_id":         target_id,
            "event_type":        str(event["event_type"]),
            "window_mid":        float(event["window_mid"]),
            "obs_duration_days": obs_duration_days,
            "slew_days":         slew_days,
            "tier_before":       tier_before,
            "tier_after":        tier_after,
            "missed":            missed,
            "population_bin":    str(target.get("population_bin", "")),
            "science_weight":    float(target.get("science_weight", 0.0)),
            "ra":                float(target.get("ra", 0.0)),
            "dec":               float(target.get("dec", 0.0)),
        })

        return info

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------

    def _increment_progress(self, target_id: str, target: pd.Series) -> None:
        old_completed = int(self._progress_dict[target_id]["obs_completed"])
        new_completed = old_completed + 1
        new_state = compute_progress(new_completed, target)
        # Update both the fast dict and the backing DataFrame.
        self._progress_dict[target_id].update(new_state)
        for k, v in new_state.items():
            self.progress.at[target_id, k] = v
        # Invalidate population bin cache if tier-1 status may have changed.
        if new_state.get("tier1_done"):
            self._pop_bin_cache = None

    def get_progress(self, target_id: str) -> pd.Series:
        return self.progress.loc[target_id]

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    @property
    def tier1_completed(self) -> int:
        return int(self.progress["tier1_done"].sum())

    @property
    def tier2_completed(self) -> int:
        return int(self.progress["tier2_done"].sum())

    @property
    def tier3_completed(self) -> int:
        return int(self.progress["tier3_done"].sum())

    @property
    def population_bin_counts(self) -> dict[str, int]:
        """Number of Tier 1+ completed targets per population bin."""
        if self._pop_bin_cache is None:
            completed = self.progress[self.progress["tier1_done"]]
            merged = completed.join(self.targets.set_index("target_id")[["population_bin"]])
            self._pop_bin_cache = merged["population_bin"].value_counts().to_dict()
        return self._pop_bin_cache

    @property
    def total_targets(self) -> int:
        return len(self.targets)

    def is_done(self) -> bool:
        """Episode is over when the mission clock runs out."""
        return self.clock.mission_over

    # ------------------------------------------------------------------
    # Upcoming events
    # ------------------------------------------------------------------

    def upcoming_events(self, n: int = 50) -> pd.DataFrame:
        """Return the next *n* events from current time, only valid ones."""
        t_now = self.clock.current_time
        future = self.events[
            (self.events["window_end"] > t_now) &
            (self.events["visibility_valid"])
        ].head(n)
        return future

    def events_for_target(self, target_id: str) -> pd.DataFrame:
        return self.events[
            (self.events["target_id"] == target_id) &
            (self.events["window_end"] > self.clock.current_time)
        ]

    def next_event_for_target(self, target_id: str) -> Optional[pd.Series]:
        evs = self.events_for_target(target_id)
        if evs.empty:
            return None
        return evs.iloc[0]

    # ------------------------------------------------------------------
    # Agent-facing summary dict
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Compact dict of global mission state — used in observation building."""
        clk = self.clock.snapshot()
        return {
            **clk,
            "current_ra":         self.current_ra,
            "current_dec":        self.current_dec,
            "tier1_completed":    self.tier1_completed,
            "tier2_completed":    self.tier2_completed,
            "tier3_completed":    self.tier3_completed,
            "total_targets":      self.total_targets,
            "population_bin_counts": self.population_bin_counts,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Deep snapshot for logging / debugging (not meant for full restore)."""
        return {
            "clock":    self.clock.snapshot(),
            "progress": self.progress.to_dict(orient="index"),
            "pointing": {"ra": self.current_ra, "dec": self.current_dec},
        }

    def obs_log_df(self) -> "pd.DataFrame":
        """Return the observation log as a tidy DataFrame."""
        if not self.obs_log:
            return pd.DataFrame(columns=[
                "step", "mission_day", "target_id", "event_type",
                "window_mid", "obs_duration_days", "slew_days",
                "tier_before", "tier_after", "missed",
                "population_bin", "science_weight", "ra", "dec",
            ])
        return pd.DataFrame(self.obs_log)

    def reset(self) -> None:
        """Reset state to start of episode (clock + progress, not tables)."""
        self.clock.reset()
        self.progress = initialise_progress_table(self.targets)
        self._progress_dict = self.progress.to_dict(orient="index")
        self.obs_log = []
        self._pop_bin_cache = None
        if len(self.targets):
            self.current_ra = float(self.targets["ra"].iloc[0])
            self.current_dec = float(self.targets["dec"].iloc[0])
