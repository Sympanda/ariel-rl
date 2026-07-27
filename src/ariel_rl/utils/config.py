"""
Configuration system: YAML files → frozen dataclass hierarchy.

All tuneable constants live here.  Hardcoded values in the simulator
modules serve only as fallback defaults; the env always passes explicit
values derived from the loaded config.

Usage
-----
    from ariel_rl.utils.config import load_env_config, EnvConfig

    cfg = load_env_config("configs/env/simple.yaml")
    print(cfg.slew.rate_deg_per_min)   # 1.0
    print(cfg.action.topk.k)           # 50

Structure
---------
    EnvConfig
    ├── MissionConfig          mission timing and budget
    ├── SlewConfig             telescope slew model
    ├── ActionConfig           action space type + type-specific sub-config
    │   ├── TopKActionConfig   K, sort order
    │   └── TargetActionConfig N-target full-set config
    ├── ObservationConfig      which features to include, normalisation
    └── RewardConfig           weights per component (populated later)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml

from ariel_rl.data.schemas import (
    MISSION_END_BJD,
    MISSION_LIFETIME_DAYS,
    MISSION_START_BJD,
    COST_FACTOR,
)
from ariel_rl.simulator.slew import (
    MAX_SLEW_S,
    MIN_SLEW_S,
    SLEW_RATE_DEG_PER_MIN,
)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MissionConfig:
    """Mission-level constants."""
    start_bjd: float = MISSION_START_BJD
    lifetime_days: float = MISSION_LIFETIME_DAYS
    cost_factor: float = COST_FACTOR
    overhead_days_per_obs: float = 0.0  # fixed per-obs overhead beyond T14

    # Global cap on the maximum tier any target can reach, regardless of what
    # the catalogue says.  Useful for ablation studies:
    #   max_tier_cap=1 → T1-only run (baseline feasibility)
    #   max_tier_cap=2 → ignore T3 targets
    #   max_tier_cap=3 → use catalogue values (default)
    max_tier_cap: int = 3


@dataclass(frozen=True)
class SlewConfig:
    """Telescope slew model parameters.

    Note: the true Ariel slew performance is not yet published.
    ``rate_deg_per_min`` is the key uncertain parameter — keep it in config.
    """
    rate_deg_per_min: float = SLEW_RATE_DEG_PER_MIN
    min_slew_seconds: float = MIN_SLEW_S
    max_slew_seconds: float = MAX_SLEW_S


@dataclass(frozen=True)
class TopKActionConfig:
    """Config for the ``topk`` action space.

    The agent chooses from the next *k* upcoming events sorted by
    ``sort_by`` (e.g. chronological order).
    """
    k: int = 50
    sort_by: str = "window_mid"   # column in the event table to sort by


@dataclass(frozen=True)
class TargetActionConfig:
    """Config for the ``target`` action space.

    The agent picks one of the N targets directly; the env automatically
    schedules the next available event for that target.  Targets with no
    remaining events are masked out.
    """
    include_completed: bool = False   # mask out fully-completed targets?


@dataclass(frozen=True)
class ActionConfig:
    """Selects which action space to use and carries its sub-config."""
    type: str = "topk"                        # "topk" | "target"
    topk: TopKActionConfig = field(default_factory=TopKActionConfig)
    target: TargetActionConfig = field(default_factory=TargetActionConfig)


# Per-event feature names the observation builder understands.
# Listed here as a reference; the YAML can select a subset.
ALL_EVENT_FEATURES: list[str] = [
    "slew_time_days",                 # angular slew cost in days
    "window_urgency_norm",            # fraction of window already elapsed (0=fresh, 1=closing)
    "duration_days",                  # transit / eclipse duration
    "total_time_cost_days",           # slew + obs combined (excludes wait)
    "progress_in_tier",               # fraction of obs completed toward next tier
    "obs_remaining_next_tier_norm",   # obs still needed, normalised by max possible
    "base_science_value",             # catalogue SNR-derived value [0, 1]
    "science_weight",                 # catalogue priority weight [0, 1]
    "planet_radius_norm",             # planet radius / 20 Re
    "planet_temperature_norm",        # equilibrium temp / 3000 K
    "planet_mass_norm",               # planet mass / 4000 Me
    "stellar_temperature_norm",       # stellar Teff / 10000 K
    "stellar_metallicity",            # [Fe/H] (negative values allowed)
    "tier_goal_norm",                 # tier_goal / 3
    "event_type_binary",              # 0 = transit, 1 = eclipse
    "days_to_window_end_norm",        # (window_end - t_now) / 30 days
]

ALL_GLOBAL_FEATURES: list[str] = [
    "fraction_elapsed",               # mission time consumed
    "tier1_fraction",                 # T1-complete targets / total
    "tier2_fraction",                 # T2-complete targets / total
    "tier3_fraction",                 # T3-complete targets / total
    "used_science_fraction",          # science time / mission length
    "used_slew_fraction",             # slew time / mission length
    "n_observations_norm",            # cumulative obs count / 5000
    "n_completed_targets_norm",       # fraction of targets fully completed (at max tier)
]


@dataclass(frozen=True)
class ObservationConfig:
    """Controls which features appear in the agent observation."""
    event_features: list[str] = field(default_factory=lambda: list(ALL_EVENT_FEATURES))
    global_features: list[str] = field(default_factory=lambda: list(ALL_GLOBAL_FEATURES))
    include_population_bin_fractions: bool = True
    #: Only include population bins with at least this many targets.
    #: Tiny bins (< threshold) are almost never observed in a single episode
    #: and would be constant-zero features that waste model capacity.
    min_bin_targets: int = 10
    normalise: bool = True


@dataclass(frozen=True)
class RewardConfig:
    """Reward component weights.

    Tier completion bonuses (sparse, per-step)
    ------------------------------------------
    ``tier1/2/3_completion`` — base reward when a target reaches that tier.
    Scaled by ``science_weight × (1 + diversity)`` at runtime so rare,
    under-represented targets are worth proportionally more.

    Suggested starting ratio: T1=1, T2=3, T3=10.  Since tier progression is
    *cumulative* (T2 requires finishing T1 first), a single target taken all
    the way to T3 earns 1+3+10=14 × scale.

    Progress shaping (dense, per-step)
    -----------------------------------
    ``progress_weight`` — small reward proportional to the increase in
    ``progress_in_tier`` on each valid observation.  Provides a learning
    signal in the long stretches between tier completions.  Also scaled by
    ``science_weight × (1 + diversity)``.

    ``near_completion_scale`` — multiplier applied to the progress reward
    when ``progress_in_tier > near_completion_threshold``.  Encourages the
    agent to finish targets that are almost complete rather than abandoning
    them.  E.g., scale=3.0 makes the last 30% of a tier worth 3× more per
    step.

    Efficiency reward (dense, per-step)
    ------------------------------------
    ``efficiency_weight`` — reward proportional to
    ``obs_duration / (obs_duration + slew_duration)``.  Penalises wasted
    slew time without needing a separate slew penalty weight.

    Coverage milestone bonuses (sparse, one-shot)
    ----------------------------------------------
    ``t1_milestone_fractions`` — list of T1-coverage fractions at which a
    one-shot bonus is fired.  Defaults to [0.25, 0.5, 0.75, 0.90, 1.0].
    These fire at most once each per episode, directly incentivising breadth
    across the full target catalogue.

    ``t1_milestone_bonus`` — base value of each one-shot bonus.  The bonus
    at the 100 % milestone is awarded double (``2 × t1_milestone_bonus``) to
    represent completing the whole T1 survey.

    Terminal episode bonus
    ----------------------
    ``t1_terminal_weight`` — applied at episode end (or mission completion)
    as ``t1_terminal_weight × (tier1_fraction)^t1_terminal_power``.  The
    quadratic default (power=2) means the last 10 % of T1 coverage is
    disproportionately valuable, pushing the agent toward full coverage.

    Missed-event penalty
    --------------------
    ``miss_penalty`` — subtracted when the agent arrives after ``window_end``.

    Invalid-action penalty
    ----------------------
    ``invalid_action_penalty`` — applied immediately by ArielEnv when the
    agent picks a masked action (clock does not advance).
    """
    # --- sparse tier completion bonuses ---
    tier1_completion: float = 1.0
    tier2_completion: float = 3.0
    tier3_completion: float = 10.0

    # --- dense per-step shaping ---
    progress_weight:            float = 0.3
    efficiency_weight:          float = 0.5
    near_completion_threshold:  float = 0.7   # progress_in_tier above which boost applies
    near_completion_scale:      float = 3.0   # multiplier on progress reward near a tier boundary

    # --- diversity multiplier ---
    # Scales how much more attractive an under-observed population bin is vs a saturated one.
    # At max_multiplier=5.0 a fully unseen bin is 5× more rewarding than a full bin.
    # Was hardcoded at 2.0 in transformer_v1; increase to 5.0 to close the coverage gap.
    diversity_multiplier_max: float = 5.0

    # --- rarity / difficulty bonus ---
    # Added on every successful observation: rarity_weight × (period/period_ref)² / tier_worked
    # Rewards long-period targets that are costly to miss (they won't come around again soon).
    # The quadratic squashing keeps short-period targets near-zero while pushing yearly orbits
    # up to ~rarity_weight in magnitude.  Dividing by tier_worked (1/2/3) softens the bonus
    # for higher tiers which are already well-rewarded by tier_completion weights.
    rarity_weight:          float = 0.5    # scale; at period=period_ref this equals rarity_weight/tier
    rarity_period_ref_days: float = 365.0  # period [days] that maps to difficulty = 1.0

    # --- coverage milestone bonuses (one-shot per episode) ---
    t1_milestone_fractions: tuple = (0.25, 0.5, 0.75, 0.90, 1.0)
    t1_milestone_bonus:     float = 20.0   # base bonus per milestone; 100% milestone = 2×

    # --- terminal episode bonus ---
    t1_terminal_weight: float = 50.0   # scale of end-of-episode T1-coverage bonus
    t1_terminal_power:  float = 2.0    # exponent; >1 rewards near-full coverage most

    # --- random-baseline subtraction ---
    # When True, ``random_baseline_per_step`` is subtracted from the reward on
    # every valid (non-missed) observation.  This centres the reward around zero
    # for average random-agent behaviour, making the advantage signal proportional
    # to improvement *over* random rather than absolute reward magnitude.
    # Calibrate ``random_baseline_per_step`` from a quick random-agent run:
    #   total_random_reward / n_steps  (typically ~4.0 for the default reward config).
    subtract_random_baseline:  bool  = False
    random_baseline_per_step:  float = 4.0

    # --- penalties ---
    miss_penalty:           float = 0.1
    invalid_action_penalty: float = 0.5

    # ---------------------------------------------------------------------------
    # Relative reward mode
    # ---------------------------------------------------------------------------
    # Set ``reward_mode = "relative"`` to replace the per-step absolute reward with
    # a checkpoint-based signal that measures how much better the agent is doing
    # compared to a pre-recorded baseline policy trajectory.
    #
    # How it works
    # ------------
    # The underlying absolute reward is always computed internally (same formula as
    # "absolute" mode).  In relative mode the agent is NOT given the raw per-step
    # reward; instead it receives two types of bonus/penalty at fixed mission-time
    # checkpoints:
    #
    # Short-interval (weekly) comparison:
    #   Every ``comparison_interval_days`` of elapsed mission time the agent's
    #   accumulated absolute reward for that interval is compared to the stored
    #   baseline mean reward for the same interval:
    #       reward += comparison_scale × (agent_interval_reward − baseline_interval_reward)
    #
    # Compound (monthly) comparison:
    #   Every ``compound_interval_days`` the agent's TOTAL cumulative absolute
    #   reward so far is compared to the baseline total at that same mission-time
    #   checkpoint:
    #       reward += compound_scale × (agent_total_reward − baseline_total_at_checkpoint)
    #   This compounds: a consistently-better agent sees a growing bonus each month.
    #
    # Setup
    # -----
    # 1. Generate the baseline trajectory once (BEFORE training):
    #       python scripts/generate_baseline_trajectory.py \
    #           --policy smart_greedy --n-episodes 20 \
    #           --config configs/env/simple.yaml \
    #           --out data/baselines/smart_greedy_trajectory.json
    # 2. Point ``baseline_trajectory_path`` at the resulting JSON.
    # 3. Set ``reward_mode: relative`` in your reward YAML.
    #
    # The ``info["abs_reward"]`` key always carries the raw absolute reward for
    # monitoring/debugging regardless of which mode is active.
    # ---------------------------------------------------------------------------
    reward_mode: str = "absolute"          # "absolute" | "relative"

    # Interval for the weekly-style marginal comparison (mission days).
    comparison_interval_days: float = 7.0

    # Interval for the monthly-style cumulative comparison (mission days).
    # Conventionally a multiple of comparison_interval_days.
    compound_interval_days: float = 28.0

    # Scale factors applied to each checkpoint signal.
    # Keep comparison_scale ~ 1 and compound_scale < 1 so the compound bonus
    # is a supplement rather than the dominant signal.
    comparison_scale: float = 1.0
    compound_scale:   float = 0.1

    # Path to a baseline trajectory JSON produced by generate_baseline_trajectory.py.
    # Required when reward_mode = "relative".  Supports absolute and cwd-relative paths.
    baseline_trajectory_path: str = ""


# ---------------------------------------------------------------------------
# Top-level env config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvConfig:
    """Full environment configuration."""
    mission: MissionConfig = field(default_factory=MissionConfig)
    slew: SlewConfig = field(default_factory=SlewConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: int = 42


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _merge_dicts(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base*, returning a new dict."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _merge_dicts(result[key], val)
        else:
            result[key] = val
    return result


def _dict_to_dataclass(cls: type, data: dict) -> Any:
    """Recursively convert a nested dict to a (frozen) dataclass instance."""
    if data is None:
        return cls()

    kwargs: dict[str, Any] = {}
    field_map = {f.name: f for f in fields(cls)}

    for fname, fld in field_map.items():
        if fname not in data:
            continue
        val = data[fname]
        ftype = fld.type

        # Resolve string annotations
        if isinstance(ftype, str):
            import sys
            ftype = eval(ftype, sys.modules[cls.__module__].__dict__)  # noqa: S307

        # Recurse into nested dataclasses
        if hasattr(ftype, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[fname] = _dict_to_dataclass(ftype, val)
        else:
            kwargs[fname] = val

    return cls(**kwargs)


def load_env_config(path: str | Path) -> EnvConfig:
    """Load an EnvConfig from a YAML file.

    Unknown keys are silently ignored; missing keys fall back to defaults.

    Parameters
    ----------
    path:
        Path to the YAML config file.

    Returns
    -------
    EnvConfig
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    return _dict_to_dataclass(EnvConfig, raw)


def default_env_config() -> EnvConfig:
    """Return an EnvConfig with all defaults (no YAML needed)."""
    return EnvConfig()


def env_config_to_dict(cfg: EnvConfig) -> dict:
    """Serialise an EnvConfig back to a plain dict (for logging/saving)."""
    return asdict(cfg)
