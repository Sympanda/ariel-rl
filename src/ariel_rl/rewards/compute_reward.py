"""
Multi-component reward function for the Ariel mission scheduling environment.

Per-step reward components
--------------------------
1. **Tier completion bonus** (sparse)
   Fired once when a target crosses a tier boundary.  Scaled by
   ``science_weight × diversity_multiplier``.  Weights: T1=1, T2=3, T3=10.

2. **Progress shaping** (dense)
   Reward proportional to Δprogress_in_tier.  A near-completion multiplier
   (``near_completion_scale``) kicks in when ``progress_in_tier > threshold``
   so the agent is incentivised to *finish* targets rather than abandoning
   them just before a tier boundary.

3. **Efficiency reward** (dense)
   ``obs_duration / total_cost`` — penalises long slews implicitly.

4. **Rarity / difficulty bonus** (dense)
   ``rarity_weight × (period / period_ref)² / tier_worked``
   Rewards observations of long-period targets that are costly to miss —
   they won't come around again for months or years.  The quadratic term
   keeps short-period planets near-zero while pushing year-long orbits up
   to ``rarity_weight``.  Dividing by the tier being worked toward (1/2/3)
   softens the bonus for higher tiers which are already well-rewarded.

5. **Missed-event penalty**
   Subtracted when the agent arrives after ``window_end``.

One-shot per-episode components
--------------------------------
5. **Coverage milestone bonuses**
   Large one-shot bonuses when T1 completions cross 25%, 50%, 75%, 90%, 100%
   of the reachable catalogue.  Fired by ``ArielEnv`` via
   ``check_milestone_reward``.  Directly incentivise breadth.

6. **Terminal episode bonus**
   Fired at the end of each episode by ``ArielEnv`` via
   ``compute_terminal_reward``.  ``t1_terminal_weight × (t1_fraction)^power``
   — quadratic by default so near-complete T1 coverage is disproportionately
   valuable.

Public API
----------
    from ariel_rl.rewards.compute_reward import (
        compute_reward,
        check_milestone_reward,
        compute_terminal_reward,
    )
"""

from __future__ import annotations

from ariel_rl.utils.config import RewardConfig


def _diversity_multiplier(
    population_bin: str,
    bin_totals: dict[str, int],
    bin_observed: dict[str, int],
    max_multiplier: float = 5.0,
) -> float:
    """Return a coverage-diversity multiplier for *population_bin*.

    Ranges from ``max_multiplier`` (bin completely unseen) down to ``1.0``
    (bin fully saturated).  The default ``max_multiplier=5.0`` makes rare,
    under-observed bins five times more attractive than fully-covered ones,
    which is strong enough to overcome the efficiency reward and close the
    coverage gap seen in transformer_v1.

    Parameters
    ----------
    population_bin:
        Bin label for the target being observed (e.g. ``"hot_jupiter"``).
    bin_totals:
        Pre-computed static dict of ``{bin: total_targets_in_bin}``
        (from ``MissionState._bin_totals``).
    bin_observed:
        Current dict of ``{bin: tier1+_completed_count}``
        (from ``MissionState.population_bin_counts``).
    max_multiplier:
        Upper bound of the multiplier (unseen bin).  Tune via
        ``RewardConfig.diversity_multiplier_max``.
    """
    total = bin_totals.get(population_bin, 1)
    observed = bin_observed.get(population_bin, 0)
    fraction = observed / total if total > 0 else 0.0
    return 1.0 + (max_multiplier - 1.0) * max(0.0, 1.0 - fraction)


def compute_reward(
    step_result: dict,
    cfg: RewardConfig,
    bin_totals: dict[str, int],
    bin_observed: dict[str, int],
) -> float:
    """Compute the scalar reward for a completed environment step.

    Parameters
    ----------
    step_result:
        The dict returned by ``MissionState.execute_observation``.  Must
        contain the keys listed in ``MissionState.execute_observation``
        docstring, including the reward-specific extras
        (``progress_before``, ``progress_after``, ``science_weight``,
        ``population_bin``).
    cfg:
        ``RewardConfig`` holding all weight hyper-parameters.
    bin_totals:
        Static per-bin target counts (``MissionState._bin_totals``).
    bin_observed:
        Current per-bin Tier-1+ completion counts
        (``MissionState.population_bin_counts``).

    Returns
    -------
    float
        Scalar reward for this step.
    """
    missed: bool          = step_result.get("missed", False)
    science_weight: float = float(step_result.get("science_weight", 0.5))
    population_bin: str   = str(step_result.get("population_bin", ""))
    tier_before: int      = int(step_result.get("tier_before", 0))
    tier_after: int       = int(step_result.get("tier_after", 0))
    progress_before: float = float(step_result.get("progress_before", 0.0))
    progress_after: float  = float(step_result.get("progress_after", 0.0))
    obs_dur: float        = float(step_result.get("obs_duration_days", 0.0))
    slew_dur: float       = float(step_result.get("slew_days", 0.0))

    # Missed-event: pay the slew, get no science, small penalty.
    if missed:
        return -cfg.miss_penalty

    # Diversity multiplier — higher reward for under-represented bins.
    div_mult = _diversity_multiplier(
        population_bin, bin_totals, bin_observed, cfg.diversity_multiplier_max
    )
    scale = science_weight * div_mult

    reward = 0.0

    # ------------------------------------------------------------------ #
    # 1. Tier completion bonus (sparse)
    # ------------------------------------------------------------------ #
    if tier_after > tier_before:
        _tier_weights = {
            1: cfg.tier1_completion,
            2: cfg.tier2_completion,
            3: cfg.tier3_completion,
        }
        for completed_tier in range(tier_before + 1, tier_after + 1):
            reward += _tier_weights.get(completed_tier, 0.0) * scale

    # ------------------------------------------------------------------ #
    # 2. Progress shaping (dense, only when no tier boundary is crossed)
    # ------------------------------------------------------------------ #
    # When a tier completes, progress_in_tier resets — the completion bonus
    # already handles that step.
    if tier_before == tier_after:
        delta = progress_after - progress_before
        if delta > 0.0:
            # Near-completion boost: incentivise finishing targets rather than
            # abandoning them just before a tier boundary.
            near_boost = (
                cfg.near_completion_scale
                if progress_after >= cfg.near_completion_threshold
                else 1.0
            )
            reward += cfg.progress_weight * delta * scale * near_boost

    # ------------------------------------------------------------------ #
    # 3. Efficiency reward (dense)
    # ------------------------------------------------------------------ #
    total_cost = obs_dur + slew_dur
    if total_cost > 0.0:
        efficiency = obs_dur / total_cost
        reward += cfg.efficiency_weight * efficiency

    # ------------------------------------------------------------------ #
    # 4. Rarity / difficulty bonus (dense)
    # Rewards observations of long-period targets that are hard to catch:
    #   bonus = rarity_weight × difficulty² / tier_worked
    # where difficulty = min(period_days / period_ref, 1) ∈ [0, 1].
    # ------------------------------------------------------------------ #
    if cfg.rarity_weight > 0.0:
        period: float = float(step_result.get("period", 0.0))
        if period > 0.0 and cfg.rarity_period_ref_days > 0.0:
            difficulty  = min(period / cfg.rarity_period_ref_days, 1.0)
            tier_worked = max(tier_before + 1, 1)   # 1, 2, or 3
            reward += cfg.rarity_weight * (difficulty ** 2) / tier_worked

    # ------------------------------------------------------------------ #
    # 5. Random-baseline subtraction (optional)
    # Centres the per-step reward around zero for random-agent behaviour so
    # the advantage signal reflects improvement *over* random rather than
    # absolute reward.  Only subtracted from valid observations (not misses,
    # which have their own penalty path).
    # ------------------------------------------------------------------ #
    if cfg.subtract_random_baseline:
        reward -= cfg.random_baseline_per_step

    return reward


# ---------------------------------------------------------------------------
# One-shot milestone bonuses (called by ArielEnv, not per-step)
# ---------------------------------------------------------------------------

def check_milestone_reward(
    tier1_completed: int,
    total_reachable: int,
    milestones_hit: set[float],
    cfg: RewardConfig,
) -> tuple[float, set[float]]:
    """Return a one-shot bonus if a new T1-coverage milestone has been crossed.

    Called by ``ArielEnv`` after every step that changes ``tier1_completed``.

    Parameters
    ----------
    tier1_completed:
        Current number of targets with at least T1 complete.
    total_reachable:
        Total targets in the catalogue (denominator for coverage fraction).
    milestones_hit:
        Set of milestone fractions already awarded this episode.  Modified
        in-place and returned so ``ArielEnv`` can persist it in episode state.
    cfg:
        ``RewardConfig`` holding ``t1_milestone_fractions`` and
        ``t1_milestone_bonus``.

    Returns
    -------
    bonus : float
        Sum of any newly triggered milestone bonuses (0.0 if none).
    milestones_hit : set[float]
        Updated set (same object as input).
    """
    if total_reachable == 0:
        return 0.0, milestones_hit

    fraction = tier1_completed / total_reachable
    bonus = 0.0

    for milestone in cfg.t1_milestone_fractions:
        if milestone not in milestones_hit and fraction >= milestone:
            milestones_hit.add(milestone)
            # The 100 % milestone is worth double — completing the full T1 survey
            multiplier = 2.0 if milestone >= 1.0 else 1.0
            bonus += cfg.t1_milestone_bonus * multiplier

    return bonus, milestones_hit


def compute_terminal_reward(
    tier1_completed: int,
    total_reachable: int,
    cfg: RewardConfig,
) -> float:
    """Terminal bonus fired once at the end of each episode.

    Scales as ``t1_terminal_weight × (t1_fraction)^t1_terminal_power``.
    With the default power=2 this is quadratic: getting from 80%→100% T1
    coverage is worth far more than 0%→20%, pushing the agent toward
    full catalogue coverage.

    Parameters
    ----------
    tier1_completed:
        Final number of T1-complete targets at episode end.
    total_reachable:
        Total targets (denominator).
    cfg:
        ``RewardConfig``.
    """
    if total_reachable == 0:
        return 0.0
    t1_fraction = tier1_completed / total_reachable
    return cfg.t1_terminal_weight * (t1_fraction ** cfg.t1_terminal_power)
