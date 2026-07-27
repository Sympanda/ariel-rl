"""
Run a short episode with all baseline schedulers (and optionally a trained RL
model) and produce summary plots.

Usage
-----
    python scripts/run_short_episode.py [--days 60] [--csv PATH] [--out-dir plots/short_episode]

    # Include a trained RL model for comparison:
    python scripts/run_short_episode.py --model-path outputs/my_run/model.zip --model-name MyRL

The script:
  1. Loads the target catalogue (or uses a path you provide).
  2. Builds a DynamicBackend — no event table pre-computation needed.
  3. Runs all baseline agents (+ optional RL model).
  4. Prints a summary table and saves into --out-dir:
       activity_<agent>.png  — monthly activity breakdown
       timeline_<agent>.png  — per-target Gantt chart
       schedule_<agent>.png  — classic schedule timeline
       reward_curves.png     — per-step reward for all agents
       comparison.png        — side-by-side bar chart
       coverage.png          — T1 population coverage heatmap
       plots/coverage.png          — population coverage heatmap (GreedyBalanced)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- project imports ----
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ariel_rl.data.preprocess_targets import build_target_table
from ariel_rl.data.schemas import MISSION_START_BJD
from ariel_rl.envs.ariel_env import ArielEnv
from ariel_rl.simulator.event_backend import DynamicBackend
from ariel_rl.utils.config import (
    EnvConfig, MissionConfig, ActionConfig, TopKActionConfig,
    SlewConfig, ObservationConfig, RewardConfig,
)
from ariel_rl.baselines.random_valid import RandomValid
from ariel_rl.baselines.greedy_value import GreedyValue
from ariel_rl.baselines.greedy_balanced import GreedyBalanced
from ariel_rl.baselines.earliest_deadline import EarliestDeadline
from ariel_rl.baselines.smart_greedy import SmartGreedy
from ariel_rl.evaluation.compare_runs import (
    run_episode, run_episode_with_log, compare_baselines, summary_table,
)
from ariel_rl.evaluation.metrics import EpisodeStats


def make_config(lifetime_days: float) -> EnvConfig:
    """Build an EnvConfig for a short episode."""
    return EnvConfig(
        mission=MissionConfig(
            start_bjd=MISSION_START_BJD,
            lifetime_days=lifetime_days,
        ),
        slew=SlewConfig(),
        action=ActionConfig(type="topk", topk=TopKActionConfig(k=50)),
        observation=ObservationConfig(normalise=True),
        reward=RewardConfig(),
    )


def print_summary(results: pd.DataFrame) -> None:
    cols = [
        "agent",
        "n_observations", "n_missed", "miss_rate",
        "tier1_completed", "tier2_completed", "tier3_completed",
        "science_efficiency", "bin_coverage",
    ]
    display = results[cols].copy()
    display["miss_rate"]          = display["miss_rate"].map("{:.1%}".format)
    display["science_efficiency"] = display["science_efficiency"].map("{:.1%}".format)
    display["bin_coverage"]       = display["bin_coverage"].map("{:.1%}".format)
    print("\n" + "=" * 70)
    print("  2-MONTH EPISODE SUMMARY")
    print("=" * 70)
    print(display.to_string(index=False))
    print("=" * 70 + "\n")


def save_plots(
    env: ArielEnv,
    agents: dict,
    results: pd.DataFrame,
    out_dir: Path,
    lifetime: float,
) -> None:
    """Re-run each agent (collecting obs_log + reward log) and save plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from ariel_rl.evaluation.plots import (
            plot_schedule_timeline,
            plot_agent_comparison,
            plot_coverage_heatmap,
            plot_reward_curve,
            plot_action_timeline,
            plot_activity_timeline,
        )
    except ImportError as e:
        print(f"[plots] Skipping — {e}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    reward_logs: dict[str, pd.DataFrame] = {}

    for name, agent in agents.items():
        slug = name.lower().replace(" ", "_")
        _, log_df = run_episode_with_log(env, agent, seed=0)
        reward_logs[name] = log_df

        if env.state is None or not env.state.obs_log:
            continue

        # Monthly activity breakdown
        try:
            fig, _ = plot_activity_timeline(env.state)
            fig.suptitle(f"{name} — monthly activity", fontsize=11)
            fig.savefig(out_dir / f"activity_{slug}.png", dpi=150, bbox_inches="tight")
            print(f"  Saved activity_{slug}.png")
            plt.close(fig)
        except Exception as exc:
            print(f"  [activity {name}] {exc}")

        # Per-target Gantt / timeline
        try:
            fig, _ = plot_action_timeline(env.state)
            fig.suptitle(f"{name} — {lifetime:.0f}-day action timeline", fontsize=11, y=1.01)
            fig.savefig(out_dir / f"timeline_{slug}.png", dpi=150, bbox_inches="tight")
            print(f"  Saved timeline_{slug}.png")
            plt.close(fig)
        except Exception as exc:
            print(f"  [timeline {name}] {exc}")

        # Classic Gantt (existing)
        try:
            fig, _ = plot_schedule_timeline(env.state)
            fig.suptitle(f"{name} — {lifetime:.0f}-day schedule", fontsize=11)
            fig.savefig(out_dir / f"schedule_{slug}.png", dpi=150, bbox_inches="tight")
            print(f"  Saved schedule_{slug}.png")
            plt.close(fig)
        except Exception as exc:
            print(f"  [schedule {name}] {exc}")

    # Reward curves — all agents on one plot
    try:
        fig, _ = plot_reward_curve(reward_logs, x_axis="mission_day")
        fig.savefig(out_dir / "reward_curves.png", dpi=150, bbox_inches="tight")
        print("  Saved reward_curves.png")
        plt.close(fig)
    except Exception as exc:
        print(f"  [reward curves] {exc}")

    # Comparison bar chart
    try:
        fig, _ = plot_agent_comparison(results)
        fig.savefig(out_dir / "comparison.png", dpi=150, bbox_inches="tight")
        print("  Saved comparison.png")
        plt.close(fig)
    except Exception as exc:
        print(f"  [comparison plot] {exc}")

    # Population coverage for GreedyBalanced
    best_agent_name = "GreedyBalanced"
    if best_agent_name in agents:
        run_episode(env, agents[best_agent_name], seed=0)
        try:
            fig, _ = plot_coverage_heatmap(env.state, tier=1)
            fig.suptitle(f"{best_agent_name} — T1 population coverage", fontsize=11)
            fig.savefig(out_dir / "coverage.png", dpi=150, bbox_inches="tight")
            print("  Saved coverage.png")
            plt.close(fig)
        except Exception as exc:
            print(f"  [coverage plot] {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short Ariel episode with baselines.")
    parser.add_argument("--days",    type=float, default=60.0,  help="Mission lifetime in days (default: 60)")
    parser.add_argument("--csv",     type=str,   default=None,  help="Path to MCS CSV (uses default if omitted)")
    parser.add_argument("--out-dir", type=str,   default="plots/short_episode", help="Output directory for plots (default: plots/short_episode/)")
    parser.add_argument("--no-plots", action="store_true",      help="Skip plot generation")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to a saved MaskablePPO model (.zip) to include alongside baselines.")
    parser.add_argument("--model-name", type=str, default="RLAgent",
                        help="Display name for the RL model in tables and plots.")
    parser.add_argument("--reward-config", type=str, default=None,
                        help="Optional reward-only YAML overlaid on top of the default config "
                             "(e.g. configs/reward/sparse_dominant.yaml).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    lifetime = args.days

    # ---- load targets ----
    print(f"Loading targets...")
    targets = build_target_table(args.csv)
    print(f"  {len(targets)} targets loaded.")

    # ---- build env ----
    print(f"Building env ({lifetime:.0f}-day mission, DynamicBackend)...")
    cfg = make_config(lifetime)

    if args.reward_config:
        import dataclasses, yaml as _yaml
        from ariel_rl.utils.config import RewardConfig
        from dataclasses import fields as _fields
        _rpath = Path(args.reward_config)
        if _rpath.exists():
            with open(_rpath) as _f:
                _rdata = _yaml.safe_load(_f) or {}
            if "reward" in _rdata and isinstance(_rdata["reward"], dict):
                _rdata = _rdata["reward"]
            _valid = {f.name for f in _fields(RewardConfig)}
            _kwargs = {k: v for k, v in _rdata.items() if k in _valid}
            cfg = dataclasses.replace(cfg, reward=dataclasses.replace(cfg.reward, **_kwargs))
            print(f"  Reward config: {_rpath}  ({len(_kwargs)} keys overridden)")
        else:
            print(f"  [warning] --reward-config path not found: {_rpath}")

    backend = DynamicBackend(targets)
    env = ArielEnv(config=cfg, targets=targets, backend=backend)
    print(f"  Action space: Discrete({env.n_actions})")
    print(f"  Obs shape: events={env.observation_space['events'].shape}, "
          f"global={env.observation_space['global'].shape}")

    # ---- agents ----
    obs_cfg = env.cfg.observation
    agents = {
        "RandomValid":      RandomValid(),
        "GreedyValue":      GreedyValue(obs_cfg=obs_cfg),
        "GreedyBalanced":   GreedyBalanced(obs_cfg=obs_cfg),
        "EarliestDeadline": EarliestDeadline(obs_cfg=obs_cfg),
        "SmartGreedy":      SmartGreedy(obs_cfg=obs_cfg),
    }

    # Optionally add a trained RL agent for direct comparison
    if args.model_path:
        try:
            sys.path.insert(0, str(ROOT / "src"))
            from ariel_rl.agents.rl_agent import RLAgentWrapper
            rl_agent = RLAgentWrapper.load(args.model_path, name=args.model_name)
            agents[args.model_name] = rl_agent
            print(f"  Loaded RL model: {args.model_path}")
        except Exception as e:
            print(f"  [warning] Could not load RL model: {e}")

    # ---- run ----
    print(f"\nRunning {len(agents)} agents × 1 episode each...")
    results = compare_baselines(env, agents, n_episodes=1, seed_start=42, verbose=True)

    print_summary(results)

    # ---- plots ----
    if not args.no_plots:
        print(f"Saving plots to {out_dir}/")
        save_plots(env, agents, results, out_dir, lifetime)
    else:
        print("(plots skipped — pass without --no-plots to generate them)")


if __name__ == "__main__":
    main()
