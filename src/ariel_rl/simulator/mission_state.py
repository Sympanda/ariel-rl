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
# COST_FACTOR = 2.5 — the authoritative observation block multiplier:
#   block_duration_days = COST_FACTOR * T14_days
# Every observation block includes:
#   (COST_FACTOR/2 - 0.5) * T14 pre-baseline  +  T14 transit  +  (COST_FACTOR/2 - 0.5) * T14 post-baseline
# = 0.75 * T14  +  T14  +  0.75 * T14  =  2.5 * T14
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
            # If we were given a non-empty events table, wrap it in a
            # TableBackend (e.g. from from_tables or legacy test helpers).
            # Otherwise default to DynamicBackend.
            if len(self.events) > 0:
                from ariel_rl.simulator.event_backend import TableBackend
                self._backend = TableBackend(self.events)
            else:
                from ariel_rl.simulator.event_backend import DynamicBackend
                self._backend = DynamicBackend(self.targets)
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

        Timing and partial-observation model
        --------------------------------------
        The telescope slews *immediately* after the action is chosen.  The
        clock advances as:

            slew  →  idle (wait for block_start if arrived early)  →  observe

        where:
            block_duration_days = COST_FACTOR * T14_days  (= 2.5 × T14)
            block_start         = window_mid − block_duration_days / 2
            block_end           = window_mid + block_duration_days / 2

        Three cases are handled:
          Case A  t_arrive ≤ block_start   : full block, captured_fraction = 1.0
          Case B  block_start < t_arrive < block_end
                                           : partial block,
                                             captured_fraction = (block_end − t_arrive)
                                                                 / block_duration_days
          Case C  t_arrive ≥ block_end     : complete miss, only slew cost paid

        A partial capture still contributes fractional progress to obs_completed
        so the agent is never penalised for arriving slightly late.

        Parameters
        ----------
        event_id:
            Identifier returned by the active ``EventBackend``.

        Returns
        -------
        dict with keys:
            target_id, event_type, block_duration_days, slew_days, idle_days,
            obs_duration_days, total_cost_days, captured_fraction,
            tier_before, tier_after, tier_completed,
            obs_number, missed (bool), progress_before, progress_after,
            science_weight, population_bin, period, host_id
        """
        event = self._backend.get_event(event_id)
        target_id = event["target_id"]
        target = self._target_lookup[target_id]

        # ---- Slew: starts immediately from current pointing ----
        slew_days = slew_time_days(
            ra1=self.current_ra,
            dec1=self.current_dec,
            ra2=float(target["ra"]),
            dec2=float(target["dec"]),
        )

        t_arrive = self.clock.current_time + slew_days

        # ---- Observation block geometry ----
        # block_duration_days is stored on the event by DynamicBackend;
        # fall back to COST_FACTOR * duration_days for legacy TableBackend rows.
        block_duration_days = float(
            event["block_duration_days"]
            if "block_duration_days" in event.index
            else COST_FACTOR * event["duration_days"]
        )
        window_mid  = float(event["window_mid"])
        block_start = window_mid - block_duration_days / 2.0
        block_end   = window_mid + block_duration_days / 2.0  # observation block ends here

        # ---- Partial observation model ----
        # The observation block extends beyond the raw transit/eclipse window:
        #
        #   block_start ←—(0.75 × T14 baseline)—→ transit ←—(0.75 × T14 baseline)—→ block_end
        #
        # Three cases:
        #   Case A  (t_arrive ≤ block_start):  full block captured  → fraction = 1.0
        #   Case B  (block_start < t_arrive < block_end):
        #               partial block captured → fraction = (block_end − t_arrive) / block_duration
        #   Case C  (t_arrive ≥ block_end):    no useful data        → missed = True
        #
        # A partial observation still advances science progress — it contributes
        # a fraction of one equivalent observation to obs_completed.

        if t_arrive >= block_end:
            captured_fraction = 0.0
            missed = True
        elif t_arrive <= block_start:
            captured_fraction = 1.0
            missed = False
        else:
            captured_fraction = (block_end - t_arrive) / block_duration_days
            missed = False

        tier_before    = int(self._progress_dict[target_id]["current_tier"])
        progress_before = float(self._progress_dict[target_id]["progress_in_tier"])

        if missed:
            self.clock.record_miss()
            # Pay only the slew; no science, no idle wait.
            self.clock.advance(obs_duration_days=0.0, slew_days=slew_days)
            idle_days = 0.0
            obs_duration_days = 0.0
        else:
            # ---- Idle: arrived before block_start → wait ----
            idle_days = max(0.0, block_start - t_arrive)

            # ---- Observation duration = captured portion of the block ----
            # For a full capture (fraction=1.0) this equals block_duration_days.
            # For a partial capture the clock advances by only the remaining
            # portion, so telescope time is not "wasted" on the elapsed part.
            obs_duration_days = captured_fraction * block_duration_days

            self.clock.advance(
                obs_duration_days=obs_duration_days,
                slew_days=slew_days,
                idle_days=idle_days,
            )
            self.current_ra  = float(target["ra"])
            self.current_dec = float(target["dec"])
            self._increment_progress(target_id, target, captured_fraction)

        tier_after    = int(self._progress_dict[target_id]["current_tier"])
        progress_after = float(self._progress_dict[target_id]["progress_in_tier"])

        total_cost_days = slew_days + idle_days + obs_duration_days

        info = {
            "target_id":           target_id,
            "event_id":            event_id,
            "event_type":          event["event_type"],
            "block_duration_days": block_duration_days,
            "obs_duration_days":   obs_duration_days,
            "captured_fraction":   captured_fraction,
            "slew_days":           slew_days,
            "idle_days":           idle_days,
            "total_cost_days":     total_cost_days,
            "tier_before":         tier_before,
            "tier_after":          tier_after,
            "tier_completed":      tier_after > tier_before,
            "obs_number":          float(self._progress_dict[target_id]["obs_completed"]),
            "missed":              missed,
            "progress_before":     progress_before,
            "progress_after":      progress_after,
            "science_weight":      float(target.get("science_weight", 0.5)),
            "population_bin":      str(target.get("population_bin", "")),
            "period":              float(target.get("period", 0.0)),
            "host_id":             str(target.get("host_id", "")),
        }

        # Append to observation log (used for Gantt / diagnostic plots)
        self.obs_log.append({
            "step":                len(self.obs_log),
            "mission_day":         window_mid - self.clock.mission_start,
            "target_id":           target_id,
            "event_type":          str(event["event_type"]),
            "window_mid":          window_mid,
            "block_duration_days": block_duration_days,
            "obs_duration_days":   obs_duration_days,
            "captured_fraction":   captured_fraction,
            "slew_days":           slew_days,
            "idle_days":           idle_days,
            "tier_before":         tier_before,
            "tier_after":          tier_after,
            "missed":              missed,
            "population_bin":      str(target.get("population_bin", "")),
            "science_weight":      float(target.get("science_weight", 0.0)),
            "ra":                  float(target.get("ra", 0.0)),
            "dec":                 float(target.get("dec", 0.0)),
        })

        return info

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------

    def _increment_progress(
        self,
        target_id: str,
        target: pd.Series,
        captured_fraction: float = 1.0,
    ) -> None:
        """Advance obs_completed by captured_fraction (1.0 for a full observation).

        Partial observations (0 < fraction < 1) accumulate fractional progress;
        tier thresholds are crossed when obs_completed first meets an integer
        tier requirement.
        """
        old_completed = float(self._progress_dict[target_id]["obs_completed"])
        new_completed = old_completed + captured_fraction
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
                "window_mid", "block_duration_days", "obs_duration_days",
                "captured_fraction", "slew_days", "idle_days",
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
