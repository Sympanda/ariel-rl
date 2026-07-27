"""
Train an RL agent on ArielEnv with MaskablePPO.

Usage
-----
    # Phase 1 — train from scratch on absolute reward (500 k steps)
    python -m ariel_rl.scripts.train_agent \\
        --policy transformer \\
        --n-envs 4 \\
        --total-timesteps 500_000 \\
        --run-name transformer_v2

    # Phase 2 — fine-tune the phase-1 model against a greedy baseline
    #   (first generate the baseline trajectory if not already done)
    #   python scripts/generate_baseline_trajectory.py --policy smart_greedy \\
    #       --n-episodes 20 --out data/baselines/smart_greedy_trajectory.json
    python -m ariel_rl.scripts.train_agent \\
        --load-model outputs/transformer_v2/final_model.zip \\
        --reward-config configs/reward/relative_smart_greedy.yaml \\
        --total-timesteps 500_000 \\
        --run-name transformer_v2_relative

    # Curriculum: T1-only first, short episode
    python -m ariel_rl.scripts.train_agent \\
        --lifetime-days 365 \\
        --max-tier-cap 1 \\
        --run-name curriculum_stage1

    # MLP sanity-check policy
    python -m ariel_rl.scripts.train_agent \\
        --policy mlp \\
        --total-timesteps 500_000 \\
        --run-name mlp_sanity_check

Outputs
-------
    outputs/<run_name>/
        final_model.zip         — saved MaskablePPO weights
        progress.csv            — per-rollout SB3 metrics (reward, losses, KL, …)
        eval_stats.csv          — per-episode evaluation stats
        plots/                  — post-training diagnostic plots
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

# Must be set before any OpenMP-using imports (PyTorch, NumPy, SciPy) to avoid
# the duplicate libomp.dylib crash common in macOS conda environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train MaskablePPO on ArielEnv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- environment ----
    env = p.add_argument_group("Environment")
    env.add_argument("--config", type=Path, default=None,
                     help="Path to an env YAML config.  Falls back to code defaults.")
    env.add_argument("--reward-config", type=Path, default=None,
                     help="Optional reward-only YAML (just a 'reward:' block) overlaid on top "
                          "of --config.  Use to swap reward presets without duplicating the full "
                          "env YAML (e.g. configs/reward/sparse_dominant.yaml).")
    env.add_argument("--csv-path", type=Path, default=None,
                     help="Path to the MCS CSV.  Auto-detected from data/raw/ if not given.")
    env.add_argument("--lifetime-days", type=float, default=None,
                     help="Override MissionConfig.lifetime_days (curriculum lever).")
    env.add_argument("--max-tier-cap", type=int, default=None, choices=[1, 2, 3],
                     help="Override MissionConfig.max_tier_cap (curriculum lever).")
    env.add_argument("--action-type", type=str, default=None, choices=["topk", "target"],
                     help="Override action space type.")
    env.add_argument("--topk-k", type=int, default=None,
                     help="Override top-k action space size (only for --action-type topk).")

    # ---- policy ----
    pol = p.add_argument_group("Policy")
    pol.add_argument("--policy", type=str, default="transformer", choices=["transformer", "mlp"],
                     help="Which policy architecture to use.")
    pol.add_argument("--d-model", type=int, default=128,
                     help="Transformer hidden dim (ArielTransformerPolicy only).")
    pol.add_argument("--n-heads", type=int, default=4,
                     help="Attention heads — must divide d-model exactly.")
    pol.add_argument("--n-layers", type=int, default=3,
                     help="Transformer encoder depth.")
    pol.add_argument("--dropout", type=float, default=0.0,
                     help="Attention/FFN dropout (0 recommended for on-policy RL).")
    pol.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256],
                     help="MLP hidden layer widths (ignored for transformer policy).")

    # ---- PPO hyperparameters ----
    ppo = p.add_argument_group("PPO")
    ppo.add_argument("--n-envs",          type=int,   default=4)
    ppo.add_argument("--total-timesteps", type=int,   default=500_000)
    ppo.add_argument("--n-steps",         type=int,   default=2048,
                     help="Rollout steps collected per env per PPO update.")
    ppo.add_argument("--batch-size",      type=int,   default=64,
                     help="Mini-batch size for gradient updates.")
    ppo.add_argument("--n-epochs",        type=int,   default=10,
                     help="PPO gradient epochs per rollout batch.")
    ppo.add_argument("--lr",              type=float, default=3e-4, dest="learning_rate")
    ppo.add_argument("--gamma",           type=float, default=0.999,
                     help="Discount factor — high value suits long-horizon episodes.")
    ppo.add_argument("--gae-lambda",      type=float, default=0.95)
    ppo.add_argument("--clip-range",      type=float, default=0.2)
    ppo.add_argument("--ent-coef",        type=float, default=0.02,
                     help="Entropy bonus coefficient (encourages exploration).")
    ppo.add_argument("--vf-coef",         type=float, default=0.5)
    ppo.add_argument("--max-grad-norm",   type=float, default=0.5)

    # ---- output / hardware ----
    out = p.add_argument_group("Output / Hardware")
    out.add_argument("--run-name",    type=str, default="ariel_ppo",
                     help="Name used for the output checkpoint directory.")
    out.add_argument("--load-model",  type=Path, default=None,
                     help="Path to a saved model (.zip) to fine-tune from.  The policy "
                          "weights are loaded; the env / reward config comes from the "
                          "other CLI flags as usual.  Use this for Phase 2 relative-reward "
                          "fine-tuning after a Phase 1 absolute-reward run.")
    out.add_argument("--seed",        type=int, default=42)
    out.add_argument("--save-freq",   type=int, default=100_000,
                     help="Save a checkpoint every this many timesteps.")
    out.add_argument("--verbose",     type=int, default=1, choices=[0, 1, 2])
    out.add_argument("--device",      type=str, default="auto",
                     help="Compute device: 'auto' (detects MPS/CUDA/CPU), 'mps', 'cuda', 'cpu'.")

    return p


# ---------------------------------------------------------------------------
# End-of-training evaluation and plots
# ---------------------------------------------------------------------------

def post_training_plots(
    model,
    cfg,
    targets,
    events,
    out_dir: Path,
    run_name: str,
    n_eval_episodes: int = 3,
    seed: int = 0,
) -> None:
    """Generate and save all diagnostic plots after training completes.

    Reads SB3's progress.csv for training curves and runs evaluation episodes
    to produce the same diagnostic plots as run_short_episode.py.

        plots/training_curves.png   — PPO losses + episode reward over training
        plots/activity_<name>.png   — monthly mission activity breakdown
        plots/timeline_<name>.png   — per-target action Gantt
        plots/schedule_<name>.png   — classic schedule timeline
        plots/reward_curve.png      — per-step and cumulative reward
        plots/episode_summary.png   — 4-panel tier/time/bin overview
        plots/coverage.png          — population coverage heatmap
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ariel_rl.agents.rl_agent import RLAgentWrapper
    from ariel_rl.envs import ArielEnv
    from ariel_rl.evaluation.compare_runs import run_episode_with_log
    from ariel_rl.evaluation.plots import (
        plot_training_curves,
        plot_episode_summary,
        plot_reward_curve,
        plot_coverage_heatmap,
        plot_schedule_timeline,
        plot_activity_timeline,
        plot_action_timeline,
    )

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    slug = run_name.lower().replace(" ", "_")

    # ── 1. Training curves (from SB3's progress.csv) ─────────────────────────
    progress_csv = out_dir / "progress.csv"
    if progress_csv.exists():
        try:
            raw = pd.read_csv(progress_csv)
            # Rename SB3 column names to what plot_training_curves expects
            col_map = {
                "time/total_timesteps":       "step",
                "rollout/ep_rew_mean":         "episode_reward",
                "rollout/ep_len_mean":         "episode_length",
                "train/value_loss":            "value_loss",
                "train/policy_gradient_loss":  "policy_loss",
                "train/approx_kl":             "kl_divergence",
            }
            train_df = raw.rename(columns=col_map)
            # SB3 logs -entropy; flip sign so positive = more exploration
            if "train/entropy_loss" in raw.columns:
                train_df["entropy"] = -raw["train/entropy_loss"]
            if "train/explained_variance" in raw.columns:
                train_df["explained_variance"] = raw["train/explained_variance"]

            # Only plot rows where at least one reward data point exists
            has_reward = train_df.get("episode_reward", pd.Series(dtype=float)).notna()
            if has_reward.any():
                fig, _ = plot_training_curves(train_df[has_reward].reset_index(drop=True))
                path = plots_dir / "training_curves.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  training_curves  → {path}")
            else:
                print("  [info] No episode reward data in progress.csv yet — skipping training curves")
        except Exception as e:
            print(f"  [warning] training_curves: {e}")
    else:
        print("  [info] progress.csv not found — skipping training curves")

    # ── 2. Eval episodes via RLAgentWrapper ───────────────────────────────────
    print(f"\nRunning {n_eval_episodes} evaluation episode(s) …")
    agent = RLAgentWrapper(model, deterministic=True, name=run_name)
    env   = ArielEnv(cfg, targets=targets, events=events)

    all_stats, reward_logs = [], {}

    for ep in range(n_eval_episodes):
        stats, log_df = run_episode_with_log(env, agent, seed=seed + ep)
        all_stats.append(stats)
        reward_logs[f"{run_name}_ep{ep+1}"] = log_df

    # Save eval stats CSV
    stats_df = pd.DataFrame([s.to_dict() for s in all_stats])
    stats_df.to_csv(out_dir / "eval_stats.csv", index=False)
    key_cols = ["tier1_rate", "tier2_rate", "tier3_rate",
                "bin_coverage", "science_efficiency", "miss_rate"]
    print("\n  Mean eval stats:")
    print("  " + stats_df[[c for c in key_cols if c in stats_df]].mean().to_string())

    # ── 3. Per-episode plots (last eval episode's env.state) ──────────────────

    if env.state and env.state.obs_log:
        # Monthly activity breakdown
        try:
            fig, _ = plot_activity_timeline(env.state)
            fig.suptitle(f"{run_name} — monthly activity", fontsize=11)
            path = plots_dir / f"activity_{slug}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  activity         → {path}")
        except Exception as e:
            print(f"  [warning] activity: {e}")

        # Per-target Gantt
        try:
            fig, _ = plot_action_timeline(env.state)
            fig.suptitle(f"{run_name} — action timeline", fontsize=11, y=1.01)
            path = plots_dir / f"timeline_{slug}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  timeline         → {path}")
        except Exception as e:
            print(f"  [warning] timeline: {e}")

        # Classic schedule Gantt
        try:
            fig, _ = plot_schedule_timeline(env.state)
            fig.suptitle(f"{run_name} — schedule", fontsize=11)
            path = plots_dir / f"schedule_{slug}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  schedule         → {path}")
        except Exception as e:
            print(f"  [warning] schedule: {e}")

    # Reward curves — all eval episodes overlaid
    try:
        fig, _ = plot_reward_curve(reward_logs, x_axis="mission_day")
        path = plots_dir / "reward_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  reward_curve     → {path}")
    except Exception as e:
        print(f"  [warning] reward_curve: {e}")

    # Episode summary (4-panel)
    try:
        fig, _ = plot_episode_summary(env.state)
        path = plots_dir / "episode_summary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  episode_summary  → {path}")
    except Exception as e:
        print(f"  [warning] episode_summary: {e}")

    # Population coverage heatmap
    try:
        fig, _ = plot_coverage_heatmap(env.state, tier=1)
        fig.suptitle(f"{run_name} — T1 population coverage", fontsize=11)
        path = plots_dir / "coverage.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  coverage         → {path}")
    except Exception as e:
        print(f"  [warning] coverage: {e}")

    print(f"\nAll plots saved to {plots_dir}/")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _override_config(cfg, args):
    """Apply curriculum / CLI overrides to a loaded EnvConfig."""
    mission_overrides = {}
    if args.lifetime_days is not None:
        mission_overrides["lifetime_days"] = args.lifetime_days
    if args.max_tier_cap is not None:
        mission_overrides["max_tier_cap"] = args.max_tier_cap

    action_overrides = {}
    if args.action_type is not None:
        action_overrides["type"] = args.action_type
    if args.topk_k is not None:
        action_overrides["topk"] = dataclasses.replace(cfg.action.topk, k=args.topk_k)

    new_mission = dataclasses.replace(cfg.mission, **mission_overrides) if mission_overrides else cfg.mission
    new_action  = dataclasses.replace(cfg.action,  **action_overrides)  if action_overrides  else cfg.action
    return dataclasses.replace(cfg, mission=new_mission, action=new_action)


def main() -> None:
    args = build_parser().parse_args()

    # ---- lazy imports (fast --help) ----
    import torch as th
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure as configure_logger

    from ariel_rl.agents.ppo_masked import make_training_envs
    from ariel_rl.agents.policies.event_attention_policy import ArielTransformerPolicy
    from ariel_rl.agents.policies.mlp_scorer import ArielMlpPolicy
    from ariel_rl.data.preprocess_targets import build_target_table
    from ariel_rl.simulator.event_generator import generate_events
    from ariel_rl.utils.config import default_env_config, load_env_config

    # ---- device selection ----
    if args.device == "auto":
        if th.backends.mps.is_available():
            device = "mps"
        elif th.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device : {device}")

    # ---- config ----
    if args.config and args.config.exists():
        cfg = load_env_config(args.config)
        print(f"Config : {args.config}")
    else:
        cfg = default_env_config()
        print("Config : defaults (no YAML supplied or file not found)")

    # ---- optional reward overlay ----
    if args.reward_config and args.reward_config.exists():
        import yaml as _yaml
        with open(args.reward_config) as _f:
            _rdata = _yaml.safe_load(_f) or {}
        # Accept both bare  {efficiency_weight: 0.0, ...}
        # and wrapped        {reward: {efficiency_weight: 0.0, ...}}
        if "reward" in _rdata and isinstance(_rdata["reward"], dict):
            _rdata = _rdata["reward"]
        from dataclasses import fields as _fields
        _valid = {f.name for f in _fields(cfg.reward)}
        _kwargs = {k: v for k, v in _rdata.items() if k in _valid}
        cfg = dataclasses.replace(cfg, reward=dataclasses.replace(cfg.reward, **_kwargs))
        print(f"Reward : {args.reward_config}  ({len(_kwargs)} keys overridden)")
    else:
        print("Reward : defaults (no --reward-config supplied)")

    cfg = _override_config(cfg, args)
    print(f"  action.type      = {cfg.action.type}")
    print(f"  lifetime_days    = {cfg.mission.lifetime_days}")
    print(f"  max_tier_cap     = {cfg.mission.max_tier_cap}")

    # ---- pre-build shared tables ----
    print("\nBuilding shared target + event tables …")
    targets = build_target_table(args.csv_path)
    events  = generate_events(
        targets,
        mission_start=cfg.mission.start_bjd,
        mission_end=cfg.mission.start_bjd + cfg.mission.lifetime_days,
    )
    print(f"  {len(targets)} targets, {len(events):,} events")

    # ---- environments ----
    print(f"\nCreating {args.n_envs} parallel environment(s) …")
    env = make_training_envs(
        cfg, n_envs=args.n_envs, seed=args.seed,
        targets=targets, events=events,
    )

    # ---- policy ----
    if args.policy == "transformer":
        policy_cls = ArielTransformerPolicy
        policy_kwargs = {
            "d_model":  args.d_model,
            "n_heads":  args.n_heads,
            "n_layers": args.n_layers,
            "dropout":  args.dropout,
        }
        policy_desc = (
            f"ArielTransformerPolicy "
            f"(d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers})"
        )
    else:
        policy_cls = ArielMlpPolicy
        policy_kwargs = {"hidden_sizes": args.hidden_sizes}
        policy_desc = f"ArielMlpPolicy (hidden={args.hidden_sizes})"

    # ---- output dirs ----
    out_dir = Path("outputs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model ----
    print(f"\nPolicy : {policy_desc}")
    if args.load_model is not None:
        # Fine-tuning: load existing weights, swap in the new env + config.
        # The policy architecture must match (same d_model/n_heads/n_layers).
        load_path = Path(args.load_model)
        if not load_path.exists():
            raise FileNotFoundError(f"--load-model path not found: {load_path}")
        print(f"  Loading weights from {load_path}")
        model = MaskablePPO.load(
            str(load_path),
            env=env,
            device=device,
            # Pass updated PPO hyper-parameters so they take effect for this run
            custom_objects={
                "n_steps":       args.n_steps,
                "batch_size":    args.batch_size,
                "learning_rate": args.learning_rate,
                "n_epochs":      args.n_epochs,
                "gamma":         args.gamma,
                "gae_lambda":    args.gae_lambda,
                "clip_range":    args.clip_range,
                "ent_coef":      args.ent_coef,
                "vf_coef":       args.vf_coef,
                "max_grad_norm": args.max_grad_norm,
            },
        )
        print("  Fine-tuning from loaded checkpoint.")
    else:
        model = MaskablePPO(
            policy=policy_cls,
            env=env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=args.verbose,
            seed=args.seed,
            device=device,
        )

    # Use SB3's built-in CSV logger — writes outputs/<run_name>/progress.csv
    # automatically every rollout.  stdout is included so training prints still appear.
    model.set_logger(configure_logger(str(out_dir), ["stdout", "csv"]))

    if args.policy == "transformer":
        n_params = sum(p.numel() for p in model.policy.transformer_net.parameters())
    else:
        n_params = sum(p.numel() for p in model.policy.mlp_net.parameters())
    print(f"  Policy parameters : {n_params:,}")
    print(f"  Rollout buffer    : {args.n_steps * args.n_envs:,} steps per update")
    print(f"  Mini-batch size   : {args.batch_size}  ({args.n_epochs} epochs)")
    print(f"  Checkpoints       → {out_dir}/")
    print(f"\nTraining for {args.total_timesteps:,} timesteps …\n")

    # ---- callbacks ----
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(out_dir),
        name_prefix="checkpoint",
        verbose=1,
    )

    # ---- train ----
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    # ---- save final ----
    final_path = out_dir / "final_model"
    model.save(str(final_path))
    print(f"\nFinal model saved to {final_path}.zip")

    # ---- post-training evaluation + plots ----
    print("\nGenerating post-training plots …")
    post_training_plots(
        model=model,
        cfg=cfg,
        targets=targets,
        events=events,
        out_dir=out_dir,
        run_name=args.run_name,
        n_eval_episodes=3,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
