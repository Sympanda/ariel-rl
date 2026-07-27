"""
Generate publication / poster-quality comparison figures.

One wide figure is saved containing three side-by-side subplots:
  Left   — combined tier completion score (T1 + 3×T2 + 10×T3)
  Centre — science efficiency (used_science / active_time)
  Right  — population bin coverage fraction

All metrics are normalised so RandomValid = 1.0 exactly.
Agents shown: RandomValid · GreedyValue · HillClimbing (optional) · RL model.

Style
-----
  Transparent background  · DPI 150  · large poster fonts
  Shared colour legend at the bottom (no overlapping x-axis labels)

Usage
-----
    python scripts/plot_paper_comparison.py \\
        --model-path outputs/transformer_v1/final_model.zip \\
        --model-name "Transformer (RL)" \\
        --days 365 \\
        --out-dir plots/paper/

    # With hill-climbing (≈2 min extra):
    python scripts/plot_paper_comparison.py \\
        --model-path outputs/transformer_v1/final_model.zip \\
        --hc-iter 100 \\
        --days 365

    # Average over several seeds for error bars:
    python scripts/plot_paper_comparison.py \\
        --model-path outputs/transformer_v1/final_model.zip \\
        --n-episodes 5 \\
        --out-dir plots/paper/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# ---------------------------------------------------------------------------
# Agent display names and colours
# ---------------------------------------------------------------------------

_COLORS = {
    "RandomValid":   "#9E9E9E",   # neutral grey   — random baseline
    "GreedyValue":   "#1976D2",   # material blue
    "HillClimbing":  "#43A047",   # material green — hill-climbing heuristic
    "_rl":           "#E64A19",   # deep orange    — RL model (key set at runtime)
}

_SPINE  = "#444444"
_GRID   = "#e0e0e0"
_REFCLR = "#888888"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _make_env(days: float):
    from ariel_rl.data.preprocess_targets import build_target_table
    from ariel_rl.data.schemas import MISSION_START_BJD
    from ariel_rl.envs.ariel_env import ArielEnv
    from ariel_rl.simulator.event_backend import DynamicBackend
    from ariel_rl.utils.config import (
        EnvConfig, MissionConfig, ActionConfig, TopKActionConfig,
        SlewConfig, ObservationConfig, RewardConfig,
    )
    targets = build_target_table()
    cfg = EnvConfig(
        mission=MissionConfig(start_bjd=MISSION_START_BJD, lifetime_days=days),
        slew=SlewConfig(),
        action=ActionConfig(type="topk", topk=TopKActionConfig(k=50)),
        observation=ObservationConfig(normalise=True),
        reward=RewardConfig(),
    )
    backend = DynamicBackend(targets)
    return ArielEnv(config=cfg, targets=targets, backend=backend)


def _run_agent(agent, env, n_episodes: int, seed0: int) -> list[dict]:
    from ariel_rl.evaluation.compare_runs import run_episode
    records = []
    for ep in range(n_episodes):
        stats = run_episode(env, agent, seed=seed0 + ep)
        tier_score = (
            stats.tier1_completed * 1
            + stats.tier2_completed * 3
            + stats.tier3_completed * 10
        )
        records.append({
            "tier_score":         float(tier_score),
            "science_efficiency": float(stats.science_efficiency),
            "bin_coverage":       float(stats.bin_coverage),
        })
    return records


def _mean_std(records: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in records]
    return float(np.mean(vals)), float(np.std(vals, ddof=0) if len(vals) > 1 else 0.0)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":          "sans-serif",
        "font.weight":          "bold",
        "font.size":            16,
        "axes.titlesize":       22,
        "axes.titleweight":     "bold",
        "axes.labelsize":       18,
        "axes.labelweight":     "bold",
        "xtick.labelsize":      16,
        "ytick.labelsize":      16,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "axes.grid.axis":       "y",
        "grid.color":           _GRID,
        "grid.linewidth":       0.9,
        "axes.edgecolor":       _SPINE,
        "xtick.color":          _SPINE,
        "ytick.color":          _SPINE,
        "text.color":           "#1a1a1a",
        "axes.labelcolor":      "#1a1a1a",
    })


def _draw_subplot(
    ax,
    agent_names: list[str],
    means: list[float],
    stds: list[float],
    colors: list[str],
    title: str,
    show_ylabel: bool,
    n_episodes: int,
) -> None:
    """Draw one bar chart into *ax* (no x-tick labels — use shared legend)."""
    x = np.arange(len(agent_names))
    bar_w = 0.58

    bars = ax.bar(
        x, means, width=bar_w,
        color=colors, alpha=0.88,
        linewidth=0, zorder=3,
    )

    if n_episodes > 1:
        ax.errorbar(
            x, means, yerr=stds,
            fmt="none", color=_SPINE,
            capsize=5, capthick=1.8, linewidth=1.8,
            zorder=4,
        )

    # Reference line at 1.0
    ax.axhline(1.0, color=_REFCLR, linewidth=1.6, linestyle="--", zorder=2)

    # Multiplier labels above each bar
    for bar, mean, std in zip(bars, means, stds):
        pad = (std if n_episodes > 1 else 0.0) + 0.025
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + pad,
            f"×{mean:.2f}",
            ha="center", va="bottom",
            fontsize=15, fontweight="bold",
            color="#1a1a1a",
        )

    # "baseline" label below the first bar (RandomValid = 1.0)
    ax.text(
        x[0], -0.07,
        "baseline",
        ha="center", va="top",
        fontsize=12, fontweight="bold", color="#888888",
        transform=ax.get_xaxis_transform(),
    )

    ax.set_title(title, pad=10)
    ax.set_xticks([])          # labels replaced by shared legend
    ax.set_xlim(-0.6, len(agent_names) - 0.4)

    if show_ylabel:
        ax.set_ylabel("Relative to random (×)", fontsize=16, labelpad=8)

    # Y-axis ceiling
    y_top = max(means) * 1.20 + (max(stds) if n_episodes > 1 else 0.0)
    ax.set_ylim(0, max(y_top, 1.35))

    # Light green band above 1.0
    ax.axhspan(1.0, ax.get_ylim()[1], color="#E8F5E9", alpha=0.30, zorder=1)

    ax.spines["left"].set_color(_SPINE)
    ax.spines["bottom"].set_color(_SPINE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Poster-quality normalised comparison (RandomAny = 1.0).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", required=True,
                   help="Path to a saved MaskablePPO model (.zip).")
    p.add_argument("--model-name", default="Transformer (RL)",
                   help="Display name for the RL model.")
    p.add_argument("--days",       type=float, default=365.0,
                   help="Mission duration (days).")
    p.add_argument("--n-episodes", type=int,   default=1,
                   help="Episodes per agent.  >1 adds error bars (mean ± std).")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--out-dir",    default="plots/paper")
    p.add_argument("--dpi",        type=int,   default=150)
    # Hill-climbing
    p.add_argument("--hc-iter",    type=int,   default=0,
                   help="Hill-climbing optimisation iterations (0 = skip HC).")
    p.add_argument("--hc-noise",   type=float, default=0.15,
                   help="Gaussian noise std for HC weight perturbations.")
    p.add_argument("--hc-seed",    type=int,   default=0,
                   help="Seed used for HC optimisation episodes.")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _apply_style()

    from ariel_rl.baselines.random_valid    import RandomValid
    from ariel_rl.baselines.greedy_value    import GreedyValue
    from ariel_rl.baselines.hill_climbing   import HillClimbingGreedy
    from ariel_rl.agents.rl_agent           import RLAgentWrapper

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Paper Comparison Plots ===")
    print(f"  RL model   : {args.model_path}")
    print(f"  Duration   : {args.days:.0f} days")
    print(f"  Episodes   : {args.n_episodes}")
    if args.hc_iter > 0:
        print(f"  HC iters   : {args.hc_iter}  (noise={args.hc_noise})")
    else:
        print(f"  HC         : skipped (use --hc-iter N to enable)")

    # ── Build env ──────────────────────────────────────────────────────────
    print("\nBuilding environment …")
    env = _make_env(args.days)
    obs_cfg = env.cfg.observation

    # ── Agents (ordered: RandomValid, GreedyValue, [HillClimbing,] RL) ────
    rl_agent = RLAgentWrapper.load(args.model_path, name=args.model_name)
    agent_map: dict = {
        "RandomValid": RandomValid(),
        "GreedyValue": GreedyValue(obs_cfg=obs_cfg),
    }

    if args.hc_iter > 0:
        print(f"\nOptimising HillClimbing weights ({args.hc_iter} iterations) …")
        hc_agent = HillClimbingGreedy(
            obs_cfg=obs_cfg,
            env=env,
            n_iter=args.hc_iter,
            noise_scale=args.hc_noise,
            seed=args.hc_seed,
        )
        hc_agent.fit(opt_seed=args.hc_seed, verbose=True)
        agent_map["HillClimbing"] = hc_agent
        print()

    agent_map[args.model_name] = rl_agent

    # ── Run episodes ───────────────────────────────────────────────────────
    raw: dict[str, list[dict]] = {}
    for name, agent in agent_map.items():
        print(f"  Running {name} × {args.n_episodes} episode(s) …")
        raw[name] = _run_agent(agent, env, args.n_episodes, args.seed)

    # ── Normalise to RandomValid = 1.0 ────────────────────────────────────
    ref = "RandomValid"
    denom = {
        metric: max(_mean_std(raw[ref], metric)[0], 1e-9)
        for metric in ("tier_score", "science_efficiency", "bin_coverage")
    }

    ordered_names  = list(agent_map.keys())   # preserves insertion order
    ordered_colors = [
        _COLORS.get(n, _COLORS["_rl"]) for n in ordered_names
    ]

    def _normalised(metric: str):
        ms  = [_mean_std(raw[n], metric) for n in ordered_names]
        return ([m / denom[metric] for m, _ in ms],
                [s / denom[metric] for _, s in ms])

    tier_m, tier_s = _normalised("tier_score")
    eff_m,  eff_s  = _normalised("science_efficiency")
    cov_m,  cov_s  = _normalised("bin_coverage")

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'Agent':<24} {'Tier score':>11} {'Efficiency':>11} {'Coverage':>11}")
    print("─" * 62)
    for i, name in enumerate(ordered_names):
        marker = "  ← baseline" if name == ref else ""
        print(f"{name:<24} {tier_m[i]:>10.3f}×  {eff_m[i]:>10.3f}×  {cov_m[i]:>10.3f}×{marker}")

    # ── Save raw + normalised results to CSV ───────────────────────────────
    import csv, datetime as _dt
    csv_path = out_dir / "results.csv"
    csv_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "timestamp", "agent", "days", "n_episodes", "seed",
            # raw metrics
            "tier_score_mean",        "tier_score_std",
            "science_efficiency_mean","science_efficiency_std",
            "bin_coverage_mean",      "bin_coverage_std",
            # normalised to RandomValid = 1.0
            "tier_score_norm",        "tier_score_norm_std",
            "science_efficiency_norm","science_efficiency_norm_std",
            "bin_coverage_norm",      "bin_coverage_norm_std",
        ])
        if not csv_exists:
            writer.writeheader()
        ts = _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        for i, name in enumerate(ordered_names):
            raw_ts_m, raw_ts_s = _mean_std(raw[name], "tier_score")
            raw_ef_m, raw_ef_s = _mean_std(raw[name], "science_efficiency")
            raw_cv_m, raw_cv_s = _mean_std(raw[name], "bin_coverage")
            writer.writerow({
                "timestamp":               ts,
                "agent":                   name,
                "days":                    args.days,
                "n_episodes":              args.n_episodes,
                "seed":                    args.seed,
                "tier_score_mean":         round(raw_ts_m, 4),
                "tier_score_std":          round(raw_ts_s, 4),
                "science_efficiency_mean": round(raw_ef_m, 4),
                "science_efficiency_std":  round(raw_ef_s, 4),
                "bin_coverage_mean":       round(raw_cv_m, 4),
                "bin_coverage_std":        round(raw_cv_s, 4),
                "tier_score_norm":         round(tier_m[i], 4),
                "tier_score_norm_std":     round(tier_s[i], 4),
                "science_efficiency_norm": round(eff_m[i],  4),
                "science_efficiency_norm_std": round(eff_s[i], 4),
                "bin_coverage_norm":       round(cov_m[i],  4),
                "bin_coverage_norm_std":   round(cov_s[i],  4),
            })
    print(f"Results → {csv_path}")

    # ── Combined figure: 1 row × 3 subplots ───────────────────────────────
    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5.8),
        facecolor="none",
        sharey=False,
    )
    fig.patch.set_alpha(0.0)

    panels = [
        (axes[0], tier_m, tier_s, "Tier Completion Score",     True),
        (axes[1], eff_m,  eff_s,  "Science Efficiency",        False),
        (axes[2], cov_m,  cov_s,  "Population Bin Coverage",   False),
    ]
    for ax, means, stds, title, show_y in panels:
        ax.set_facecolor("none")
        _draw_subplot(
            ax=ax,
            agent_names=ordered_names,
            means=means, stds=stds,
            colors=ordered_colors,
            title=title,
            show_ylabel=show_y,
            n_episodes=args.n_episodes,
        )

    # ── Shared legend (bottom centre) ─────────────────────────────────────
    legend_handles = [
        Patch(facecolor=_COLORS.get(n, _COLORS["_rl"]), alpha=0.88,
              label=n, linewidth=0)
        for n in ordered_names
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(ordered_names),
        fontsize=16,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        handlelength=1.8,
        handleheight=1.2,
        columnspacing=2.5,
    )

    fig.subplots_adjust(
        left=0.07, right=0.98,
        top=0.91,  bottom=0.18,
        wspace=0.30,
    )

    out_path = out_dir / "comparison.png"
    fig.savefig(
        out_path, dpi=args.dpi,
        bbox_inches="tight",
        facecolor="none", transparent=True,
    )
    plt.close(fig)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
