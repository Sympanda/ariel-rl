"""
Action mask computation for both action space types.

A valid action is one where:
  1. visibility_valid is True on the event
  2. The observation block hasn't already fully elapsed (block_end > t_now)
  3. The telescope can arrive before the block ends (t_arrive < block_end)
     giving a non-zero captured_fraction
  4. The total cost fits in remaining mission time (topk / target modes)
  5. (optional) The target hasn't already reached max_tier

For ``full_set`` mode the check is more permissive: only conditions 3 and 5
are enforced.  The intent is to let the agent decide whether a small partial
capture is worth the slew; the environment always returns a non-negative reward
for any non-zero capture.

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
        return _mask_target(
            state, candidate_events,
            include_completed=cfg.target.include_completed,
            permissive=False,
        )
    elif cfg.type == "full_set":
        return _mask_target(
            state, candidate_events,
            include_completed=cfg.full_set.include_completed,
            permissive=True,   # relax can_fit check; let agent weigh partial captures
        )
    else:
        raise ValueError(f"Unknown action space type: {cfg.type!r}")


def _mask_topk(state: "MissionState", events: pd.DataFrame) -> np.ndarray:
    """Mask for the top-K upcoming event list.

    Timing model (consistent with MissionState.execute_observation):
        slew immediately  →  idle if arrived before block_start  →  observe block

    An action is valid when:
      1. visibility_valid is True
      2. Observation block hasn't fully elapsed (block_end > t_now)
      3. Telescope arrives before block_end (any non-zero capture possible)
      4. Actual time cost (slew + idle + *captured* duration + overhead) fits in
         remaining mission time — NOT full block duration
      5. Target has not yet reached max_tier

    The miss threshold is ``block_end`` (not ``window_end``) consistent with
    MissionState.execute_observation's partial-observation model.  The can_fit
    check uses the *captured* duration (i.e. ``capture_fraction × block_dur``)
    so that a valid partial observation is never rejected simply because the
    full 2.5×T14 block would not fit.
    """
    from ariel_rl.simulator.slew import slew_time_days
    from ariel_rl.data.schemas import COST_FACTOR

    n = len(events)
    if n == 0:
        return np.zeros(0, dtype=bool)

    t_now    = state.clock.current_time
    remaining = state.clock.remaining_time
    overhead = getattr(state, "overhead_days_per_obs", 0.0)

    # Extract columns once (avoids pandas per-row overhead)
    vis  = events["visibility_valid"].to_numpy(dtype=bool)
    wmid = events["window_mid"].to_numpy(dtype=float)
    tids = events["target_id"].to_numpy()
    # block_duration_days is populated by DynamicBackend; fall back for legacy rows
    if "block_duration_days" in events.columns:
        bldur = events["block_duration_days"].to_numpy(dtype=float)
    else:
        bldur = COST_FACTOR * events["duration_days"].to_numpy(dtype=float)

    bend = wmid + bldur / 2.0   # block_end for each event

    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if not vis[i]:
            continue
        if bend[i] <= t_now:
            continue
        target = state._target_lookup.get(tids[i])
        if target is None:
            continue
        prog = state._progress_dict.get(tids[i])
        if prog is not None and int(prog["current_tier"]) >= int(target["max_tier"]):
            continue
        slew = slew_time_days(
            ra1=state.current_ra, dec1=state.current_dec,
            ra2=float(target["ra"]), dec2=float(target["dec"]),
        )
        t_arrive = t_now + slew
        # Telescope must arrive before block_end to capture anything.
        if t_arrive >= bend[i]:
            continue
        # Idle: wait for block_start if arrived early
        block_start = wmid[i] - bldur[i] / 2.0
        idle = max(0.0, block_start - t_arrive)
        # Captured duration: mirrors execute_observation's three cases
        if t_arrive <= block_start:
            cap_frac = 1.0
        else:
            cap_frac = (bend[i] - t_arrive) / bldur[i]
        captured_dur = cap_frac * bldur[i]
        # can_fit uses actual captured duration + overhead, not full block
        total_cost = slew + idle + captured_dur + overhead
        if total_cost > remaining:
            continue
        mask[i] = True

    return mask


def _mask_target(
    state: "MissionState",
    next_events: pd.DataFrame,
    include_completed: bool,
    permissive: bool = False,
) -> np.ndarray:
    """Mask for the full-target and full_set action spaces.

    ``next_events`` has one row per target (the next available event for that
    target, or a sentinel row with ``visibility_valid=False`` if none).

    Parameters
    ----------
    include_completed:
        If False, targets already at max_tier are masked out.
    permissive:
        Used by ``full_set`` mode.  When True, the ``can_fit`` (budget) check
        is omitted — only the block_end feasibility constraint is kept.  This
        lets the agent see partial-capture opportunities even when the total
        idle+block cost would exceed the remaining budget, since the actual
        captured duration may be much shorter.
    """
    from ariel_rl.simulator.slew import slew_time_days
    from ariel_rl.data.schemas import COST_FACTOR

    n = len(next_events)
    mask = np.zeros(n, dtype=bool)
    t_now = state.clock.current_time
    overhead = getattr(state, "overhead_days_per_obs", 0.0)

    for i, (_, ev) in enumerate(next_events.iterrows()):
        target_id = ev.get("target_id")
        if target_id is None:
            continue

        # Completed targets
        if not include_completed:
            prog = state._progress_dict.get(target_id)
            if prog is not None:
                target_row = state._target_lookup.get(target_id)
                if target_row is not None:
                    if int(prog["current_tier"]) >= int(target_row["max_tier"]):
                        continue

        # No event available (sentinel row)
        if not bool(ev.get("visibility_valid", False)):
            continue

        raw_dur   = float(ev.get("duration_days", 0.0))
        block_dur = float(ev.get("block_duration_days", COST_FACTOR * raw_dur))
        wmid      = float(ev.get("window_mid", t_now))
        block_end = wmid + block_dur / 2.0

        # Block must not have fully elapsed yet
        if block_end <= t_now:
            continue

        target = state._target_lookup.get(target_id)
        if target is None:
            continue

        slew = slew_time_days(
            ra1=state.current_ra, dec1=state.current_dec,
            ra2=float(target["ra"]), dec2=float(target["dec"]),
        )
        t_arrive = t_now + slew
        # Telescope must arrive before the block ends (any capture possible)
        if t_arrive >= block_end:
            continue

        if not permissive:
            block_start = wmid - block_dur / 2.0
            idle = max(0.0, block_start - t_arrive)
            # Use actual captured duration (mirrors execute_observation)
            if t_arrive <= block_start:
                cap_frac = 1.0
            else:
                cap_frac = (block_end - t_arrive) / block_dur
            captured_dur = cap_frac * block_dur
            if not state.clock.can_fit(slew + idle + captured_dur + overhead):
                continue

        mask[i] = True

    return mask


def any_valid(mask: np.ndarray) -> bool:
    """Return True if at least one action is valid."""
    return bool(mask.any())
