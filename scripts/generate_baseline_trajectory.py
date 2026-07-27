"""
Generate a baseline policy reward trajectory for use with relative reward mode.

Runs a chosen baseline policy for N episodes, records the absolute reward
accumulated in each fixed-length mission-time interval, and saves the result
as a JSON file.  The JSON is then referenced by ``RewardConfig.baseline_trajectory_path``
when training with ``reward_mode = "relative"``.

Usage
-----
    # Basic — SmartGreedy, 20 episodes, default env config
    python scripts/generate_baseline_trajectory.py \\
        --policy smart_greedy \\
        --n-episodes 20 \\
        --out data/baselines/smart_greedy_trajectory.json

    # With a custom env / reward config
    python scripts/generate_baseline_trajectory.py \\
        --policy smart_greedy \\
        --n-episodes 20 \\
        --config configs/env/simple.yaml \\
        --reward-config configs/reward/sparse_dominant.yaml \\
        --comparison-interval 7 \\
        --compound-interval 28 \\
        --out data/baselines/smart_greedy_sparse_trajectory.json

Output JSON schema
------------------
{
    "policy":                        "smart_greedy",
    "n_episodes":                    20,
    "comparison_interval_days":      7.0,
    "compound_interval_days":        28.0,
    "mission_start_bjd":             2462867.0,
    "lifetime_days":                 1278.375,
    "interval_rewards":              [<float>, ...],   // mean agent reward per interval
    "compound_cumulative_rewards":   [<float>, ...],   // mean cumulative reward at each compound point
    "total_mean_reward":             <float>
}

The ``interval_rewards`` list has one entry per comparison interval (ceil(lifetime /
comparison_interval) entries).  Each entry is the mean reward the baseline accumulated
IN THAT SPECIFIC INTERVAL across all episodes.

The ``compound_cumulative_rewards`` list has one entry per compound checkpoint.
Each entry is the mean TOTAL cumulative reward from mission start up to that checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow running directly from repo root without install
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ariel_rl.baselines import ALL_BASELINES
from ariel_rl.data.preprocess_targets import build_target_table
from ariel_rl.envs.ariel_env import ArielEnv
from ariel_rl.simulator.event_backend import DynamicBackend
from ariel_rl.utils.config import (
    EnvConfig,
    RewardConfig,
    default_env_config,
    load_env_config,
)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: ArielEnv,
    agent,
    seed: int,
) -> tuple[list[float], list[float]]:
    """Run one full episode; return ``(mission_times_bjd, per_step_abs_rewards)``.

    The times record the mission clock AFTER each observation completes.
    ``abs_reward`` is always the underlying absolute reward (``info["abs_reward"]``),
    regardless of the env's ``reward_mode`` setting.
    """
    obs, info = env.reset(seed=seed)
    agent.reset()
    times: list[float] = []
    rewards: list[float] = []
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        t_now = float(env.state.clock.current_time)
        # Always read the raw absolute reward (available in both modes)
        abs_r = float(info.get("abs_reward", reward))
        times.append(t_now)
        rewards.append(abs_r)

    return times, rewards


# ---------------------------------------------------------------------------
# Trajectory builder
# ---------------------------------------------------------------------------

def build_trajectory(
    times_list: list[list[float]],
    rewards_list: list[list[float]],
    mission_start_bjd: float,
    lifetime_days: float,
    comparison_interval_days: float,
    compound_interval_days: float,
) -> dict:
    """Convert per-step episode data into fixed-interval trajectory summaries.

    Parameters
    ----------
    times_list:
        One entry per episode — list of BJD mission-clock times after each step.
    rewards_list:
        One entry per episode — list of absolute rewards at each step.
    mission_start_bjd, lifetime_days:
        Mission timing constants (to build the interval grid).
    comparison_interval_days:
        Width of each short comparison interval (e.g. 7 days).
    compound_interval_days:
        Width of each compound checkpoint interval (e.g. 28 days).

    Returns
    -------
    dict
        Ready to serialise to JSON for ``baseline_trajectory_path``.
    """
    n_episodes = len(times_list)
    n_comparison = int(np.ceil(lifetime_days / comparison_interval_days))
    n_compound = int(np.ceil(lifetime_days / compound_interval_days))

    # Per-episode arrays: [episode × interval]
    interval_rewards_all = np.zeros((n_episodes, n_comparison))
    compound_cumulative_all = np.zeros((n_episodes, n_compound))

    compound_checkpoints_days = [
        (i + 1) * compound_interval_days for i in range(n_compound)
    ]

    for ep_idx, (times, rewards) in enumerate(zip(times_list, rewards_list)):
        ep_days = np.array(times, dtype=float) - mission_start_bjd
        ep_rewards = np.array(rewards, dtype=float)

        # Bin each step's reward into its comparison interval
        for day, r in zip(ep_days, ep_rewards):
            bin_idx = int(day // comparison_interval_days)
            bin_idx = min(bin_idx, n_comparison - 1)
            interval_rewards_all[ep_idx, bin_idx] += r

        # Cumulative reward up to each compound checkpoint
        cumulative = np.cumsum(ep_rewards)
        for comp_idx, checkpoint_day in enumerate(compound_checkpoints_days):
            mask = ep_days <= checkpoint_day
            if mask.any():
                compound_cumulative_all[ep_idx, comp_idx] = cumulative[mask][-1]
            # else: episode ended before this checkpoint → leave as 0

    interval_mean = interval_rewards_all.mean(axis=0).tolist()
    compound_mean = compound_cumulative_all.mean(axis=0).tolist()
    total_mean = float(np.sum(interval_mean))

    return {
        "policy": None,           # filled in by caller
        "n_episodes": n_episodes,
        "comparison_interval_days": comparison_interval_days,
        "compound_interval_days": compound_interval_days,
        "mission_start_bjd": mission_start_bjd,
        "lifetime_days": lifetime_days,
        "interval_rewards": interval_mean,
        "compound_cumulative_rewards": compound_mean,
        "total_mean_reward": total_mean,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-run a baseline and save its reward trajectory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--policy", default="smart_greedy",
        choices=sorted(ALL_BASELINES.keys()),
        help="Baseline policy to run (default: smart_greedy).",
    )
    p.add_argument(
        "--n-episodes", type=int, default=20, metavar="N",
        help="Number of episodes to average over (default: 20).",
    )
    p.add_argument(
        "--config", default=None, metavar="PATH",
        help="Path to env YAML config.  Defaults to built-in defaults.",
    )
    p.add_argument(
        "--reward-config", default=None, metavar="PATH",
        help="Optional reward YAML overlay (merged onto env config reward section).",
    )
    p.add_argument(
        "--comparison-interval", type=float, default=7.0, metavar="DAYS",
        help="Short comparison interval in mission days (default: 7).",
    )
    p.add_argument(
        "--compound-interval", type=float, default=28.0, metavar="DAYS",
        help="Compound checkpoint interval in mission days (default: 28).",
    )
    p.add_argument(
        "--out", default="data/baselines/baseline_trajectory.json", metavar="PATH",
        help="Output JSON path (default: data/baselines/baseline_trajectory.json).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed (each episode uses seed+i, default: 0).",
    )
    p.add_argument(
        "--csv", default=None, metavar="PATH",
        help="Path to MCS CSV (defaults to built-in data path).",
    )
    p.add_argument(
        "--dynamic-backend", action="store_true",
        help="Use DynamicBackend instead of the default TableBackend.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-episode reward totals.",
    )
    return p.parse_args()


def _load_config(args: argparse.Namespace) -> EnvConfig:
    """Load and merge env + reward YAMLs, forced into absolute reward mode."""
    if args.config:
        cfg = load_env_config(args.config)
    else:
        cfg = default_env_config()

    if args.reward_config:
        with open(args.reward_config) as f:
            reward_raw = (yaml.safe_load(f) or {}).get("reward", {})
        from dataclasses import asdict, replace
        reward_dict = asdict(cfg.reward)
        reward_dict.update(reward_raw)
        # Force absolute mode — we always record raw absolute rewards
        reward_dict["reward_mode"] = "absolute"
        reward_dict["baseline_trajectory_path"] = ""
        from ariel_rl.utils.config import _dict_to_dataclass
        new_reward = _dict_to_dataclass(RewardConfig, reward_dict)
        # Rebuild EnvConfig with updated reward
        from dataclasses import fields
        env_kwargs = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        env_kwargs["reward"] = new_reward
        cfg = EnvConfig(**env_kwargs)
    else:
        # Still force absolute mode even without a reward override
        from dataclasses import asdict, fields
        reward_dict = asdict(cfg.reward)
        reward_dict["reward_mode"] = "absolute"
        reward_dict["baseline_trajectory_path"] = ""
        from ariel_rl.utils.config import _dict_to_dataclass
        new_reward = _dict_to_dataclass(RewardConfig, reward_dict)
        env_kwargs = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        env_kwargs["reward"] = new_reward
        cfg = EnvConfig(**env_kwargs)

    return cfg


def main() -> None:
    args = parse_args()

    print(f"[generate_baseline_trajectory] policy={args.policy}, "
          f"n_episodes={args.n_episodes}, "
          f"comparison_interval={args.comparison_interval}d, "
          f"compound_interval={args.compound_interval}d")

    # ---- build env ----
    cfg = _load_config(args)
    targets = build_target_table(args.csv)

    if args.dynamic_backend:
        backend = DynamicBackend(targets)
        env = ArielEnv(config=cfg, targets=targets, backend=backend)
    else:
        env = ArielEnv(config=cfg, targets=targets)

    agent_cls = ALL_BASELINES[args.policy]
    agent = agent_cls(seed=args.seed)

    # ---- run episodes ----
    all_times: list[list[float]] = []
    all_rewards: list[list[float]] = []

    for ep in range(args.n_episodes):
        times, rewards = run_episode(env, agent, seed=args.seed + ep)
        all_times.append(times)
        all_rewards.append(rewards)
        total = sum(rewards)
        if args.verbose:
            print(f"  episode {ep+1:3d}/{args.n_episodes}: "
                  f"steps={len(rewards):4d}, total_reward={total:.1f}")
        else:
            print(f"  episode {ep+1}/{args.n_episodes}: "
                  f"steps={len(rewards)}, total={total:.0f}")

    # ---- build trajectory summary ----
    traj = build_trajectory(
        times_list=all_times,
        rewards_list=all_rewards,
        mission_start_bjd=cfg.mission.start_bjd,
        lifetime_days=cfg.mission.lifetime_days,
        comparison_interval_days=args.comparison_interval,
        compound_interval_days=args.compound_interval,
    )
    traj["policy"] = args.policy

    mean_totals = [sum(r) for r in all_rewards]
    print(f"\nMean total reward across episodes: {np.mean(mean_totals):.1f} "
          f"± {np.std(mean_totals):.1f}")
    print(f"Intervals: {len(traj['interval_rewards'])} comparison / "
          f"{len(traj['compound_cumulative_rewards'])} compound")

    # ---- save ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(traj, f, indent=2)
    print(f"\nSaved trajectory → {out_path}")
    print("\nNext step: add to your reward YAML:")
    print(f"  reward:")
    print(f"    reward_mode: relative")
    print(f"    baseline_trajectory_path: {out_path}")
    print(f"    comparison_interval_days: {args.comparison_interval}")
    print(f"    compound_interval_days: {args.compound_interval}")


if __name__ == "__main__":
    main()
