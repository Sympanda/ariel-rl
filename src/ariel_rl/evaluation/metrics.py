"""
Episode statistics: everything you want to know after a completed episode.

``EpisodeStats`` is a frozen dataclass.  ``compute_stats(state)`` builds
it from a finished ``MissionState``.  The result can be logged, printed,
or collected across multiple runs for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ariel_rl.simulator.mission_state import MissionState

# Imported lazily inside compute_stats to avoid circular imports at module level
_coverage_gini = None


def _get_coverage_gini():
    global _coverage_gini
    if _coverage_gini is None:
        from ariel_rl.evaluation.population_coverage import coverage_gini
        _coverage_gini = coverage_gini
    return _coverage_gini


@dataclass(frozen=True)
class EpisodeStats:
    """Complete statistics for one finished episode.

    Tier completion
    ---------------
    tier1/2/3_completed : int
        Number of targets that reached each tier.
    tier1/2/3_rate : float
        Fraction of all targets that reached each tier.
    tier1/2/3_of_eligible : float
        Fraction of targets *eligible* for each tier that completed it.
        (e.g. only 129 of 814 MCS targets have max_tier=3)

    Schedule quality
    ----------------
    n_observations : int
        Total observation events executed.
    n_missed : int
        Events attempted but missed (clock arrived after window_end).
    miss_rate : float
        n_missed / (n_observations + n_missed)
    used_science_days : float
        Total wall-clock days on science (sum of T14 durations).
    used_slew_days : float
        Total wall-clock days spent slewing.
    science_efficiency : float
        used_science_days / (used_science_days + used_slew_days)
        Fraction of active time on science vs overhead.
    fraction_elapsed : float
        How much of the mission lifetime was used (should be ~1 if terminated).

    Population coverage
    -------------------
    n_bins_total : int
        Total number of distinct population bins in the target catalogue.
    n_bins_with_t1 : int
        Bins that have at least one Tier 1 completed target.
    bin_coverage : float
        n_bins_with_t1 / n_bins_total
    bin_counts : dict[str, int]
        Per-bin count of Tier 1+ completed targets.
    """

    # Tier completion (counts)
    tier1_completed: int
    tier2_completed: int
    tier3_completed: int
    total_targets: int

    # Tier completion (rates)
    tier1_rate: float
    tier2_rate: float
    tier3_rate: float

    # Tier completion among eligible targets only
    tier1_eligible: int
    tier2_eligible: int
    tier3_eligible: int
    tier1_of_eligible: float
    tier2_of_eligible: float
    tier3_of_eligible: float

    # Schedule quality
    n_observations: int
    n_missed: int
    miss_rate: float
    used_science_days: float
    used_slew_days: float
    science_efficiency: float
    fraction_elapsed: float

    # Population coverage
    n_bins_total: int
    n_bins_with_t1: int
    bin_coverage: float
    coverage_gini_t1: float   # Gini over per-bin T1 counts (0=uniform, 1=monopoly)
    coverage_gini_t2: float
    bin_counts: dict = field(compare=False)

    def summary_str(self) -> str:
        """One-paragraph human-readable summary."""
        lines = [
            f"Tier completion  : T1 {self.tier1_completed}/{self.total_targets} "
            f"({self.tier1_rate:.1%}),  T2 {self.tier2_completed} ({self.tier2_rate:.1%}),  "
            f"T3 {self.tier3_completed} ({self.tier3_rate:.1%})",
            f"Eligible rates   : T1 {self.tier1_of_eligible:.1%}  "
            f"T2 {self.tier2_of_eligible:.1%}  T3 {self.tier3_of_eligible:.1%}",
            f"Observations     : {self.n_observations} executed, {self.n_missed} missed "
            f"(miss rate {self.miss_rate:.1%})",
            f"Time             : {self.used_science_days:.1f}d science, "
            f"{self.used_slew_days:.1f}d slew  "
            f"(efficiency {self.science_efficiency:.1%})",
            f"Population bins  : {self.n_bins_with_t1}/{self.n_bins_total} covered "
            f"({self.bin_coverage:.1%})",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("bin_counts", None)   # exclude large nested dict by default
        return d


def compute_stats(state: "MissionState") -> EpisodeStats:
    """Build an EpisodeStats from a finished (or mid-episode) MissionState.

    Parameters
    ----------
    state:
        The MissionState after ``env.step()`` returned ``terminated=True``,
        or at any point during an episode for intermediate inspection.

    Returns
    -------
    EpisodeStats
    """
    clk = state.clock
    targets = state.targets
    progress = state.progress

    n_total = len(targets)

    # ---- tier counts ----
    t1_done = int(progress["tier1_done"].sum())
    t2_done = int(progress["tier2_done"].sum())
    t3_done = int(progress["tier3_done"].sum())

    # ---- eligible counts ----
    t1_elig = n_total                                        # everyone is eligible for T1
    t2_elig = int((targets["max_tier"] >= 2).sum())
    t3_elig = int((targets["max_tier"] >= 3).sum())

    def safe_rate(n: int, d: int) -> float:
        return n / d if d > 0 else 0.0

    # ---- schedule quality ----
    n_obs  = clk.n_observations
    n_miss = clk.n_missed
    total_attempts = n_obs + n_miss
    miss_rate = safe_rate(n_miss, total_attempts)

    sci_days  = clk.used_science_time
    slew_days = clk.used_slew_time
    active    = sci_days + slew_days
    sci_eff   = safe_rate(sci_days, active)

    # ---- population coverage ----
    bin_counts = state.population_bin_counts      # bins with ≥1 T1 completed
    all_bins   = sorted(targets["population_bin"].unique())
    n_bins     = len(all_bins)
    n_covered  = sum(1 for b in all_bins if bin_counts.get(b, 0) > 0)

    cov_gini = _get_coverage_gini()

    return EpisodeStats(
        tier1_completed=t1_done,
        tier2_completed=t2_done,
        tier3_completed=t3_done,
        total_targets=n_total,
        tier1_rate=safe_rate(t1_done, n_total),
        tier2_rate=safe_rate(t2_done, n_total),
        tier3_rate=safe_rate(t3_done, n_total),
        tier1_eligible=t1_elig,
        tier2_eligible=t2_elig,
        tier3_eligible=t3_elig,
        tier1_of_eligible=safe_rate(t1_done, t1_elig),
        tier2_of_eligible=safe_rate(t2_done, t2_elig),
        tier3_of_eligible=safe_rate(t3_done, t3_elig),
        n_observations=n_obs,
        n_missed=n_miss,
        miss_rate=miss_rate,
        used_science_days=sci_days,
        used_slew_days=slew_days,
        science_efficiency=sci_eff,
        fraction_elapsed=clk.fraction_elapsed,
        n_bins_total=n_bins,
        n_bins_with_t1=n_covered,
        bin_coverage=safe_rate(n_covered, n_bins),
        coverage_gini_t1=cov_gini(state, tier=1),
        coverage_gini_t2=cov_gini(state, tier=2),
        bin_counts=bin_counts,
    )
