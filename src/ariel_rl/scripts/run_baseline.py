"""
CLI script: run one or more baselines against the Ariel environment and
print a comparison table.

Usage
-----
    # Run all baselines, 1 episode each
    python -m ariel_rl.scripts.run_baseline

    # Run specific baselines, 3 episodes, verbose
    python -m ariel_rl.scripts.run_baseline \\
        --baselines random greedy_value greedy_balanced \\
        --episodes 3 --verbose

    # Use a custom config
    python -m ariel_rl.scripts.run_baseline --config configs/env/full.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ariel baseline schedulers.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/env/simple.yaml"),
        help="Path to env YAML config.",
    )
    parser.add_argument(
        "--baselines", nargs="+",
        default=["random", "greedy_value", "greedy_balanced", "earliest_deadline"],
        help="Which baselines to run.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Save results DataFrame to this CSV path.",
    )
    args = parser.parse_args()

    # ---- lazy imports ----
    from ariel_rl.baselines import ALL_BASELINES
    from ariel_rl.data.preprocess_targets import build_target_table
    from ariel_rl.envs.ariel_env import ArielEnv
    from ariel_rl.evaluation.compare_runs import compare_baselines, summary_table
    from ariel_rl.simulator.event_generator import generate_events
    from ariel_rl.utils.config import load_env_config, default_env_config

    # ---- config ----
    if args.config.exists():
        cfg = load_env_config(args.config)
        print(f"Config: {args.config}")
    else:
        cfg = default_env_config()
        print("Config: defaults (no YAML found)")

    print(f"Action space: {cfg.action.type}")

    # ---- build tables ----
    print("Building target table …")
    targets = build_target_table()
    print(f"  {len(targets)} targets")

    print("Generating events …")
    events = generate_events(
        targets,
        mission_start=cfg.mission.start_bjd,
        mission_end=cfg.mission.start_bjd + cfg.mission.lifetime_days,
    )
    print(f"  {len(events)} events")

    env = ArielEnv(config=cfg, targets=targets, events=events)

    # ---- build agents ----
    agents = {}
    for name in args.baselines:
        if name not in ALL_BASELINES:
            print(f"  WARNING: unknown baseline '{name}', skipping")
            continue
        cls = ALL_BASELINES[name]
        # Pass obs_cfg to baselines that accept it
        try:
            agents[name] = cls(obs_cfg=cfg.observation, seed=args.seed)
        except TypeError:
            agents[name] = cls(seed=args.seed)

    if not agents:
        print("No valid baselines selected. Exiting.")
        return

    print(f"\nRunning {len(agents)} baseline(s), {args.episodes} episode(s) each …\n")

    # ---- run ----
    results = compare_baselines(
        env, agents,
        n_episodes=args.episodes,
        seed_start=args.seed,
        verbose=args.verbose,
    )

    # ---- display ----
    key_cols = [
        "agent",
        "tier1_completed", "tier2_completed", "tier3_completed",
        "tier1_rate", "tier2_rate",
        "n_observations", "n_missed", "miss_rate",
        "science_efficiency", "bin_coverage", "gini_t1",
    ]
    key_cols = [c for c in key_cols if c in results.columns]
    print("\n=== Per-episode results ===")
    print(results[key_cols].to_string(index=False, float_format="{:.3f}".format))

    if args.episodes > 1:
        print("\n=== Aggregated (mean) ===")
        agg = summary_table(results)
        mean_cols = [c for c in agg.columns if c.endswith("_mean") or c == "agent"]
        mean_cols = [c for c in mean_cols if any(
            k in c for k in ("agent", "tier", "miss", "efficiency", "coverage", "gini")
        )]
        print(agg[mean_cols].to_string(index=False, float_format="{:.3f}".format))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False)
        print(f"\nResults saved to {args.out}")

    # ---- print one full episode summary ----
    print("\n=== Best single-episode summary ===")
    best_row = results.sort_values("tier1_completed", ascending=False).iloc[0]
    print(f"Agent: {best_row['agent']}")
    from ariel_rl.evaluation.metrics import EpisodeStats
    # Re-run the best agent to get the state for the detailed summary
    best_agent_name = best_row["agent"]
    best_agent = agents[best_agent_name]
    from ariel_rl.evaluation.compare_runs import run_episode
    stats = run_episode(env, best_agent, seed=int(args.seed))
    print(stats.summary_str())


if __name__ == "__main__":
    main()
