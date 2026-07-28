"""
EventBackend: pluggable strategy for finding candidate observation events.

Two concrete implementations are provided so that ArielEnv can switch backends
at construction time without changing any other code.

TableBackend (default)
----------------------
Works from a pre-computed event DataFrame produced by ``generate_events``.
Uses a sliding-window binary search for O(log N + K) candidate retrieval
regardless of how many total events are in the table.

    env = ArielEnv(config, targets=targets, events=events)
    # or explicitly:
    env = ArielEnv(config, targets=targets,
                   backend=TableBackend(events))

DynamicBackend
--------------
Computes observation windows on-the-fly from orbital parameters via vectorised
numpy modular arithmetic.  No event table is pre-computed.  Works for any
mission horizon and is O(N_targets) per step, regardless of time.

    from ariel_rl.simulator.event_backend import DynamicBackend
    env = ArielEnv(config, targets=targets, backend=DynamicBackend(targets))

Both backends expose the same three-method interface so ArielEnv and
MissionState are completely backend-agnostic.

Candidate DataFrame schema
--------------------------
Both ``candidates()`` calls return a DataFrame with these columns:

    event_id               int64   – unique within a step (synthetic for Dynamic)
    target_id              object  – string target identifier
    event_type             object  – "transit" or "eclipse"
    window_start           float64 – BJD of observation window start
    window_mid             float64 – BJD of mid-point (transit/eclipse centre)
    window_end             float64 – BJD of observation window end
    duration               float64 – event duration in seconds
    duration_days          float64 – event duration in days
    tier_goal              int64   – maximum tier achievable for this target
    base_science_value     float64 – static science priority in [0, 1]
    visibility_valid       bool    – True if geometrically observable (MVP: always True)
    ephemeris_uncertainty  float64 – ±1σ timing uncertainty in days
    event_index            int64   – periodic index n (−1 for DynamicBackend)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# Canonical column order — both backends always return this schema.
EVENT_COLUMNS: list[str] = [
    "event_id", "target_id", "event_type",
    "window_start", "window_mid", "window_end",
    "duration", "duration_days", "block_duration_days", "tier_goal",
    "base_science_value", "visibility_valid",
    "ephemeris_uncertainty", "event_index",
]

_EMPTY_EVENTS = pd.DataFrame(columns=EVENT_COLUMNS)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EventBackend(ABC):
    """Interface that any event-candidate provider must satisfy."""

    @abstractmethod
    def candidates(self, t_now: float, k: int) -> pd.DataFrame:
        """Return up to *k* upcoming events sorted by ``window_mid``.

        Parameters
        ----------
        t_now:
            Current mission time (BJD).
        k:
            Maximum number of candidates to return.

        Returns
        -------
        pd.DataFrame with columns ``EVENT_COLUMNS``, sorted nearest-first.
        May return fewer than *k* rows if events are scarce.
        """
        ...

    @abstractmethod
    def get_event(self, event_id: int) -> pd.Series:
        """Retrieve full details of a single event by its ``event_id``.

        For ``TableBackend``: O(1) indexed lookup in the pre-computed table.
        For ``DynamicBackend``: O(1) lookup in the per-step candidate cache
        (``candidates()`` must have been called in this step first).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any per-episode mutable state (e.g. sliding-window pointer)."""
        ...


# ---------------------------------------------------------------------------
# TableBackend
# ---------------------------------------------------------------------------

class TableBackend(EventBackend):
    """Backend backed by a pre-computed events DataFrame.

    .. deprecated::
        Use :class:`DynamicBackend` instead.  ``TableBackend`` requires a
        pre-generated event table (``generate_events.py``) and does not
        populate the ``block_duration_days`` column added in Phase 1b.
        It is kept for backward-compatibility only and will be removed.

    Wraps all of the optimised access patterns previously embedded directly in
    ``MissionState`` and ``ArielEnv._candidates_topk``:

    * Binary-search sliding window avoids O(N) boolean filter.
    * Pre-indexed DataFrame for O(1) ``get_event`` lookups.
    """

    def __init__(self, events: pd.DataFrame) -> None:
        self._events = events
        self._events_idx: pd.DataFrame = (
            events.set_index("event_id")
            if len(events) and "event_id" in events.columns
            else pd.DataFrame()
        )
        self._event_wmid: np.ndarray = (
            events["window_mid"].to_numpy()
            if len(events)
            else np.array([], dtype=np.float64)
        )
        self._max_duration_days: float = (
            float(events["duration_days"].max()) if len(events) else 1.0
        )

    # ------------------------------------------------------------------
    # EventBackend interface
    # ------------------------------------------------------------------

    def candidates(self, t_now: float, k: int) -> pd.DataFrame:
        if len(self._events) == 0:
            return _EMPTY_EVENTS.copy()

        # Binary search: find the first event whose window_mid is in range.
        # Events before this pointer all have window_end ≤ t_now (expired).
        ptr = int(np.searchsorted(
            self._event_wmid,
            t_now - self._max_duration_days,
        ))

        # Grab a small window forward — events are already sorted by window_mid.
        scan = max(k * 10, 200)
        window = self._events.iloc[ptr: ptr + scan]

        # Exact filter on the small window.
        wend = window["window_end"].to_numpy()
        valid_idx = np.where(wend > t_now)[0]

        return window.iloc[valid_idx[:k]].reset_index(drop=True)

    def get_event(self, event_id: int) -> pd.Series:
        return self._events_idx.loc[event_id]

    def reset(self) -> None:
        pass  # Stateless: no per-episode mutable data.


# ---------------------------------------------------------------------------
# DynamicBackend
# ---------------------------------------------------------------------------

class DynamicBackend(EventBackend):
    """Backend that computes observation windows on-the-fly.

    Uses vectorised numpy modular arithmetic over all targets simultaneously
    so the per-step cost is O(N_targets) ≈ 0.05 ms for ~800 targets —
    independent of mission duration.

    Synthetic event IDs
    -------------------
    Because there is no pre-computed table, event IDs are assigned
    deterministically as::

        transit:  target_index * 2
        eclipse:  target_index * 2 + 1

    These IDs are valid only for the duration of one step; the per-step
    candidate cache is refreshed by each ``candidates()`` call.

    Limitations
    -----------
    * ``get_event()`` requires a prior ``candidates()`` call in the same step.
    * Ephemeris uncertainty is set to 0 (simplified; add a noise model later).
    * The ``target`` action space type (one event per target) is not yet
      supported; use ``topk`` with this backend.
    """

    def __init__(self, targets: pd.DataFrame) -> None:
        from ariel_rl.simulator.event_generator import _base_science_value
        from ariel_rl.data.schemas import METHOD_ECLIPSE, METHOD_EITHER, COST_FACTOR
        self._cost_factor: float = COST_FACTOR

        n = len(targets)
        self._target_ids: np.ndarray = targets["target_id"].to_numpy()

        # Orbital parameters (all in days).
        self._epochs  = targets["epoch"].to_numpy(dtype=float)
        self._periods = targets["period"].to_numpy(dtype=float)

        # Transit duration.
        tr_dur_s = targets["transit_duration"].to_numpy(dtype=float)  # seconds
        self._tr_dur_s    = tr_dur_s
        self._tr_dur_days = tr_dur_s / 86400.0
        self._half_tr     = self._tr_dur_days / 2.0
        self._has_transit = np.isfinite(tr_dur_s) & (tr_dur_s > 0)

        # Eclipse duration (fall back to transit duration when absent).
        ec_dur_s = targets["eclipse_duration"].to_numpy(dtype=float)
        missing_ec = ~np.isfinite(ec_dur_s) | (ec_dur_s <= 0)
        ec_dur_s = np.where(missing_ec, tr_dur_s, ec_dur_s)
        self._ec_dur_s    = ec_dur_s
        self._ec_dur_days = ec_dur_s / 86400.0
        self._half_ec     = self._ec_dur_days / 2.0

        # Which targets observe eclipses?
        preferred = (
            targets["preferred_method"]
            if "preferred_method" in targets.columns
            else pd.Series(["transit"] * n)
        )
        has_ec_method = preferred.isin([METHOD_ECLIPSE, METHOD_EITHER]).to_numpy()
        self._has_eclipse = has_ec_method & np.isfinite(ec_dur_s) & (ec_dur_s > 0)

        # Static science value per event type.
        sw = targets["science_weight"].to_numpy(dtype=float)
        rows = list(targets.itertuples(index=False))
        self._bsv_tr = np.array([
            _base_science_value(targets.iloc[i], sw[i], float(tr_dur_s[i]))
            if self._has_transit[i] else 0.0
            for i in range(n)
        ])
        self._bsv_ec = np.array([
            _base_science_value(targets.iloc[i], sw[i], float(ec_dur_s[i]))
            if self._has_eclipse[i] else 0.0
            for i in range(n)
        ])

        # Tier goals (max tier achievable per target).
        self._tier_goals = targets["max_tier"].to_numpy(dtype=int)

        # Per-step candidate cache: event_id → row dict.
        self._event_cache: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # EventBackend interface
    # ------------------------------------------------------------------

    def candidates(self, t_now: float, k: int) -> pd.DataFrame:
        periods = self._periods

        # ---- Transits ----
        tr_phase  = (t_now - self._epochs) % periods
        in_tr     = tr_phase < self._half_tr
        tr_center = np.where(in_tr, t_now - tr_phase, t_now + (periods - tr_phase))
        tr_wend   = tr_center + self._half_tr
        tr_ok     = self._has_transit & (tr_wend > t_now)

        # ---- Eclipses (centred at epoch + period/2 for circular orbits) ----
        ec_epoch  = self._epochs + periods / 2.0
        ec_phase  = (t_now - ec_epoch) % periods
        in_ec     = ec_phase < self._half_ec
        ec_center = np.where(in_ec, t_now - ec_phase, t_now + (periods - ec_phase))
        ec_wend   = ec_center + self._half_ec
        ec_ok     = self._has_eclipse & (ec_wend > t_now)

        tr_idx = np.where(tr_ok)[0]
        ec_idx = np.where(ec_ok)[0]
        n_tr, n_ec = len(tr_idx), len(ec_idx)
        n_total = n_tr + n_ec

        if n_total == 0:
            self._event_cache = {}
            return _EMPTY_EVENTS.copy()

        # Combine window_mid times and partial-sort to find K nearest.
        all_mids = np.empty(n_total)
        all_mids[:n_tr] = tr_center[tr_idx]
        all_mids[n_tr:] = ec_center[ec_idx]

        if n_total <= k:
            order = np.argsort(all_mids)
        else:
            # O(N) partial sort, then sort only the K selected.
            part = np.argpartition(all_mids, k)[:k]
            order = part[np.argsort(all_mids[part])]

        # Build result rows and populate per-step cache.
        self._event_cache = {}
        records: list[dict] = []

        for rank in order:
            if rank < n_tr:
                i    = int(tr_idx[rank])
                eid  = i * 2
                half = self._half_tr[i]
                mid  = tr_center[i]
                dur_days = self._tr_dur_days[i]
                rec  = {
                    "event_id":              eid,
                    "target_id":             self._target_ids[i],
                    "event_type":            "transit",
                    "window_start":          mid - half,
                    "window_mid":            mid,
                    "window_end":            mid + half,
                    "duration":              self._tr_dur_s[i],
                    "duration_days":         dur_days,
                    "block_duration_days":   self._cost_factor * dur_days,
                    "tier_goal":             int(self._tier_goals[i]),
                    "base_science_value":    float(self._bsv_tr[i]),
                    "visibility_valid":      True,
                    "ephemeris_uncertainty": 0.0,
                    "event_index":           -1,
                }
            else:
                j    = int(rank - n_tr)
                i    = int(ec_idx[j])
                eid  = i * 2 + 1
                half = self._half_ec[i]
                mid  = ec_center[i]
                dur_days = self._ec_dur_days[i]
                rec  = {
                    "event_id":              eid,
                    "target_id":             self._target_ids[i],
                    "event_type":            "eclipse",
                    "window_start":          mid - half,
                    "window_mid":            mid,
                    "window_end":            mid + half,
                    "duration":              self._ec_dur_s[i],
                    "duration_days":         dur_days,
                    "block_duration_days":   self._cost_factor * dur_days,
                    "tier_goal":             int(self._tier_goals[i]),
                    "base_science_value":    float(self._bsv_ec[i]),
                    "visibility_valid":      True,
                    "ephemeris_uncertainty": 0.0,
                    "event_index":           -1,
                }

            self._event_cache[eid] = rec
            records.append(rec)

        return pd.DataFrame(records)

    def get_event(self, event_id: int) -> pd.Series:
        if event_id not in self._event_cache:
            raise KeyError(
                f"event_id {event_id!r} is not in the DynamicBackend step cache. "
                "candidates() must be called before get_event() each step."
            )
        return pd.Series(self._event_cache[event_id])

    def reset(self) -> None:
        self._event_cache.clear()
