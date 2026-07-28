"""
Per-planet feature builder for the full-set action space.

Computes a dense feature vector for every target in the catalogue at each
step.  This is the authoritative feature specification for Phase 3 (full
target-set observation space), where the agent sees all N planets rather
than a top-K slice.

Feature schema (per planet)
---------------------------
Static (time-invariant)
  planet_radius_norm              planet radius / 20 Re
  planet_mass_norm                planet mass / 4000 Me
  planet_temperature_norm         equilibrium temp / 3000 K
  period_norm                     orbital period / 365.25 days
  stellar_temperature_norm        stellar Teff / 10000 K
  stellar_metallicity             [Fe/H], clipped to [-3, 3]
  distance_norm                   distance / 1000 pc
  tier_goal_norm                  max_tier / 3
  science_weight                  catalogue priority weight [0, 1]
  event_type_binary               0=transit, 1=eclipse, 0.5=either
  host_multiplicity_norm          number of Ariel targets in same system / 5

Dynamic (updated each step)
  obs_completed_norm              obs_completed / tier3_required_obs
  progress_in_tier                current progress toward next tier [0, 1]
  current_tier_norm               current_tier / 3
  obs_remaining_norm              obs_remaining_next_tier / tier3_required_obs
  dt_next_event_norm              Δt to next transit/eclipse / 365.25 days
  dt_second_event_norm            Δt to second event / 365.25 days
  dt_third_event_norm             Δt to third event / 365.25 days
  block_duration_norm             block_duration_days (next event) / 3.0
  slew_norm                       slew time from current pointing / (2hr/24)
  scheduling_slack_norm           (window_end - (t_now + slew)) / block_duration
  ephemeris_uncertainty_norm      σ_timing / period (fractional uncertainty)
  remaining_opps_mission_norm     remaining opportunities in mission / available_total
  remaining_opps_season_norm      remaining opportunities in next 90 days / 10

The vector is padded to a fixed length (``n_features``) regardless of
which features are enabled.  Missing values (e.g. no upcoming event) are
filled with 0.

Public API
----------
    from ariel_rl.envs.planet_feature_builder import (
        build_planet_features,
        PLANET_FEATURE_NAMES,
        N_PLANET_FEATURES,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ariel_rl.simulator.mission_state import MissionState


# ---------------------------------------------------------------------------
# Static feature names (time-invariant per planet)
# ---------------------------------------------------------------------------

STATIC_FEATURE_NAMES: list[str] = [
    "planet_radius_norm",
    "planet_mass_norm",
    "planet_temperature_norm",
    "period_norm",
    "stellar_temperature_norm",
    "stellar_metallicity",
    "distance_norm",
    "tier_goal_norm",
    "science_weight",
    "event_type_binary",
    "host_multiplicity_norm",
]

# Dynamic feature names (computed from MissionState each step)
DYNAMIC_FEATURE_NAMES: list[str] = [
    "obs_completed_norm",
    "progress_in_tier",
    "current_tier_norm",
    "obs_remaining_norm",
    "dt_next_event_norm",
    "dt_second_event_norm",
    "dt_third_event_norm",
    "block_duration_norm",
    "slew_norm",
    "scheduling_slack_norm",
    "ephemeris_uncertainty_norm",
    "remaining_opps_mission_norm",
    "remaining_opps_season_norm",
]

PLANET_FEATURE_NAMES: list[str] = STATIC_FEATURE_NAMES + DYNAMIC_FEATURE_NAMES
N_PLANET_FEATURES: int = len(PLANET_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------

_NORM = {
    "planet_radius_norm":        20.0,
    "planet_mass_norm":          4000.0,
    "planet_temperature_norm":   3000.0,
    "period_norm":               365.25,
    "stellar_temperature_norm":  10000.0,
    "stellar_metallicity":       1.5,       # typical range ±1 dex → clip at ±3
    "distance_norm":             1000.0,
    "tier_goal_norm":            3.0,
    "dt_next_event_norm":        365.25,
    "dt_second_event_norm":      365.25,
    "dt_third_event_norm":       365.25,
    "block_duration_norm":       3.0,       # 2.5 × T14; max ≈ 2.5 d for long transits
    "slew_norm":                 2.0 / 24,  # 2-hr slew cap in days
    "host_multiplicity_norm":    5.0,
    "remaining_opps_season_norm": 10.0,
}

_SEASON_WINDOW_DAYS: float = 90.0   # "season" = next 90 days for remaining_opps_season


# ---------------------------------------------------------------------------
# Static table (pre-computed once per episode; host multiplicity included)
# ---------------------------------------------------------------------------

def build_static_features(state: "MissionState") -> np.ndarray:
    """Pre-compute static features for all N targets.

    Returns a float32 array of shape (N, len(STATIC_FEATURE_NAMES)).
    Call this once after reset(); cache the result on the env.
    """
    targets = state.targets
    n = len(targets)
    arr = np.zeros((n, len(STATIC_FEATURE_NAMES)), dtype=np.float32)

    # Pre-compute host multiplicity: number of targets sharing a host_id.
    host_counts: dict[str, int] = {}
    if "host_id" in targets.columns:
        for hid in targets["host_id"]:
            if hid and str(hid) != "nan":
                host_counts[str(hid)] = host_counts.get(str(hid), 0) + 1

    for i in range(n):
        row = targets.iloc[i]
        hid = str(row.get("host_id", ""))
        pref = str(row.get("preferred_method", "transit")).lower()

        arr[i, 0] = float(row.get("planet_radius", 0.0))           / _NORM["planet_radius_norm"]
        arr[i, 1] = float(row.get("planet_mass", 0.0))             / _NORM["planet_mass_norm"]
        arr[i, 2] = float(row.get("planet_temperature", 0.0))      / _NORM["planet_temperature_norm"]
        arr[i, 3] = float(row.get("period", 0.0))                  / _NORM["period_norm"]
        arr[i, 4] = float(row.get("stellar_temperature", 0.0))     / _NORM["stellar_temperature_norm"]
        arr[i, 5] = float(row.get("stellar_metallicity", 0.0))
        arr[i, 6] = float(row.get("distance_pc", 0.0))             / _NORM["distance_norm"]
        arr[i, 7] = float(row.get("max_tier", 1))                  / _NORM["tier_goal_norm"]
        arr[i, 8] = float(row.get("science_weight", 0.5))
        # 0=transit, 1=eclipse, 0.5=either
        arr[i, 9] = 1.0 if "eclipse" in pref else (0.5 if "either" in pref else 0.0)
        arr[i, 10] = min(host_counts.get(hid, 1), 5) / _NORM["host_multiplicity_norm"]

    # Clip stellarmetallicity to [-3, 3] range then normalise
    arr[:, 5] = np.clip(arr[:, 5], -3.0, 3.0) / _NORM["stellar_metallicity"]

    return arr


def build_planet_features(
    state: "MissionState",
    static_features: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the full per-planet feature matrix for the current step.

    Parameters
    ----------
    state:
        Current MissionState (read-only).
    static_features:
        Pre-computed static features from ``build_static_features``.
        If None, they are computed on the fly (slightly slower).

    Returns
    -------
    float32 array of shape (N, N_PLANET_FEATURES).
    """
    from ariel_rl.simulator.slew import slew_time_days
    from ariel_rl.data.schemas import COST_FACTOR, MISSION_LIFETIME_DAYS

    targets = state.targets
    n = len(targets)
    n_static = len(STATIC_FEATURE_NAMES)
    n_dynamic = len(DYNAMIC_FEATURE_NAMES)

    arr = np.zeros((n, N_PLANET_FEATURES), dtype=np.float32)

    if static_features is not None:
        arr[:, :n_static] = static_features
    else:
        arr[:, :n_static] = build_static_features(state)

    t_now = state.clock.current_time
    mission_end = state.clock.mission_end
    season_end = t_now + _SEASON_WINDOW_DAYS

    for i in range(n):
        row = targets.iloc[i]
        tid = str(row["target_id"])
        prog = state._progress_dict.get(tid, {})

        # ---- progress features ----
        t3_req = int(row.get("tier3_required_obs", 1)) or 1
        obs_done = int(prog.get("obs_completed", 0))
        t_prog = float(prog.get("progress_in_tier", 0.0))
        t_tier = int(prog.get("current_tier", 0))
        obs_rem = int(prog.get("obs_remaining_next_tier", t3_req))

        arr[i, n_static + 0] = obs_done / t3_req
        arr[i, n_static + 1] = t_prog
        arr[i, n_static + 2] = t_tier / 3.0
        arr[i, n_static + 3] = obs_rem / t3_req

        # ---- next three upcoming events (Δt, block duration, slew, slack) ----
        # Use DynamicBackend to find the next few events for this target.
        # We ask the backend for up to 3 future events by computing mid-times
        # directly from orbital parameters (avoids full candidates() call).
        period = float(row.get("period", 1.0)) or 1.0
        epoch  = float(row.get("epoch", t_now))
        tr_dur_days = float(row.get("transit_duration", 0.0)) / 86400.0
        block_dur = COST_FACTOR * tr_dur_days

        # Next three transit mid-times after t_now
        phase = (t_now - epoch) % period
        mid0 = t_now + (period - phase)          # first mid-time after t_now
        # Exact mid-times
        mids = [mid0, mid0 + period, mid0 + 2 * period]
        dts  = [m - t_now for m in mids]

        arr[i, n_static + 4] = min(dts[0], 365.25) / _NORM["dt_next_event_norm"]
        arr[i, n_static + 5] = min(dts[1], 365.25) / _NORM["dt_second_event_norm"]
        arr[i, n_static + 6] = min(dts[2], 365.25) / _NORM["dt_third_event_norm"]
        arr[i, n_static + 7] = min(block_dur, 3.0) / _NORM["block_duration_norm"]

        # Slew from current pointing
        slew = slew_time_days(
            ra1=state.current_ra, dec1=state.current_dec,
            ra2=float(row.get("ra", 0.0)), dec2=float(row.get("dec", 0.0)),
        )
        arr[i, n_static + 8] = slew / _NORM["slew_norm"]

        # Scheduling slack: how much margin before the first block starts
        #   slack = (block_start of next event) - (t_now + slew)
        #   block_start = mid0 - block_dur / 2
        block_start0 = mids[0] - block_dur / 2.0
        t_arrive0 = t_now + slew
        slack = (block_start0 - t_arrive0) / max(block_dur, 1e-6)
        arr[i, n_static + 9] = float(np.clip(slack, -1.0, 10.0)) / 10.0

        # Ephemeris uncertainty (fractional: σ_timing / period)
        unc_days = float(row.get("epoch_uncertainty", 0.0)) or 0.0
        arr[i, n_static + 10] = min(unc_days / period, 1.0)

        # Remaining opportunities in mission
        avail_total = float(row.get("available_transits", 0)) or 1.0
        remaining_mission = max(0.0, (mission_end - mid0) / period)
        arr[i, n_static + 11] = min(remaining_mission / avail_total, 1.0)

        # Remaining opportunities in next 90-day season
        remaining_season = max(0.0, (season_end - mid0) / period)
        arr[i, n_static + 12] = min(remaining_season, 10.0) / _NORM["remaining_opps_season_norm"]

    # Global clip to reasonable range
    arr = np.clip(arr, -3.0, 3.0)

    return arr
