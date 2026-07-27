"""
Action mask computation for both action space types.

A valid action is one where:
  1. visibility_valid is True on the event
  2. The event's window_end hasn't already passed
  3. The total cost (slew + obs duration) fits in remaining mission time
  4. (optional) The target hasn't already reached max_tier

Returns a boolean numpy array of shape (n_candidates,).
True = agent may choose this action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ariel_rl.simulator.mission_state import MissionState
    from ariel_rl.utils.config import ActionConfig


def compute_mask(
    state: "MissionState",
    candidate_events: pd.DataFrame,
    cfg: "ActionConfig",
) -> np.ndarray:
    """Compute the action validity mask for the current candidates.

    Parameters
    ----------
    state:
        Current MissionState.
    candidate_events:
        DataFrame of K events (topk) or N next-events-per-target (target mode).
    cfg:
        ActionConfig controlling which checks to apply.

    Returns
    -------
    np.ndarray of shape (len(candidate_events),), dtype bool.
    """
    if cfg.type == "topk":
        return _mask_topk(state, candidate_events)
    elif cfg.type == "target":
        return _mask_target(state, candidate_events, cfg.target.include_completed)
    else:
        raise ValueError(f"Unknown action space type: {cfg.type!r}")


def _mask_topk(state: "MissionState", events: pd.DataFrame) -> np.ndarray:
    """Mask for the top-K upcoming event list.

    Avoids per-row Python overhead by extracting event columns as numpy arrays
    upfront and computing slew in a single loop (slew itself is a pure-Python
    scalar — a fully vectorised slew would require RA/Dec arrays on state).
    """
    from ariel_rl.simulator.slew import slew_time_days

    n = len(events)
    if n == 0:
        return np.zeros(0, dtype=bool)

    t_now = state.clock.current_time
    remaining = state.clock.remaining_time

    # Extract columns once (avoids pandas per-row overhead)
    vis = events["visibility_valid"].to_numpy(dtype=bool)
    wend = events["window_end"].to_numpy(dtype=float)
    dur = events["duration_days"].to_numpy(dtype=float)
    tids = events["target_id"].to_numpy()

    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if not vis[i]:
            continue
        if wend[i] <= t_now:
            continue
        target = state._target_lookup.get(tids[i])
        if target is None:
            continue
        # Mask out targets that have already reached their max tier.
        prog = state._progress_dict.get(tids[i])
        if prog is not None and int(prog["current_tier"]) >= int(target["max_tier"]):
            continue
        slew = slew_time_days(
            ra1=state.current_ra, dec1=state.current_dec,
            ra2=float(target["ra"]), dec2=float(target["dec"]),
        )
        if slew + dur[i] > remaining:
            continue
        # Feasibility: would the slew arrive before window_end?
        # Approximate window_start = window_end - duration (symmetric window).
        # The sim jumps to window_start first, then slews, so the effective
        # arrival time is max(t_now, window_start) + slew.
        window_start_approx = wend[i] - dur[i]
        effective_t = max(t_now, window_start_approx)
        if effective_t + slew > wend[i]:
            continue
        mask[i] = True

    return mask


def _mask_target(
    state: "MissionState",
    next_events: pd.DataFrame,
    include_completed: bool,
) -> np.ndarray:
    """Mask for the full-target action space.

    ``next_events`` has one row per target (the next available event for that
    target, or a sentinel row with ``visibility_valid=False`` if none).
    """
    n = len(next_events)
    mask = np.zeros(n, dtype=bool)
    t_now = state.clock.current_time

    for i, (_, ev) in enumerate(next_events.iterrows()):
        target_id = ev.get("target_id")
        if target_id is None:
            continue

        # Completed targets
        if not include_completed:
            prog = state.progress.loc[target_id] if target_id in state.progress.index else None
            if prog is not None:
                target_row = state._target_lookup.get(target_id)
                if target_row is not None:
                    max_tier = int(target_row["max_tier"])
                    current_tier = int(prog["current_tier"])
                    if current_tier >= max_tier:
                        continue

        # No event available (sentinel row)
        if not bool(ev.get("visibility_valid", False)):
            continue

        if float(ev.get("window_end", t_now)) <= t_now:
            continue

        from ariel_rl.simulator.slew import slew_time_days
        target = state._target_lookup.get(target_id)
        if target is None:
            continue

        slew = slew_time_days(
            ra1=state.current_ra, dec1=state.current_dec,
            ra2=float(target["ra"]), dec2=float(target["dec"]),
        )
        dur = float(ev.get("duration_days", 0.0))
        if not state.clock.can_fit(slew + dur):
            continue

        mask[i] = True

    return mask


def any_valid(mask: np.ndarray) -> bool:
    """Return True if at least one action is valid."""
    return bool(mask.any())
