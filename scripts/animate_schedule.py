"""
Animate a trained RL (or baseline) scheduler as a sky-projection + activity
timeline GIF.

Layout
------
  Top panel  :  Full-sky Mollweide projection showing all 814 Ariel targets.
                Targets are colour-coded by their current tier completion status
                and grow slightly as tiers are reached.  A white crosshair tracks
                the telescope's pointing in real mission time — it moves smoothly
                across the sky during slew segments.

  Bottom panel:  Horizontal activity bar growing left-to-right as the episode
                 progresses.  Segment colours:
                   Sky-blue  = T1 obs      Teal   = T2 obs      Navy  = T3 obs
                   Amber     = Slew        Grey   = Idle         Red   = Missed

Usage
-----
    python scripts/animate_schedule.py \\
        --model-path outputs/transformer_v1/final_model.zip \\
        --days 365 \\
        --run-name transformer_1yr \\
        --n-frames 300 \\
        --fps 12 \\
        --out-dir plots/animations/

    # Baseline agent (no model needed):
    python scripts/animate_schedule.py \\
        --agent SmartGreedy \\
        --days 365 \\
        --run-name smartgreedy_1yr

Output
------
    plots/animations/<run_name>.gif
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Colour palette  (kept consistent with the rest of the codebase)
# ---------------------------------------------------------------------------

_BG = "#0d1117"   # near-black background
_FG = "#e6edf3"   # off-white text

_SEG_COLORS = {
    "t1_obs":      "#56b4e9",   # sky blue
    "t2_obs":      "#009e73",   # teal
    "t3_obs":      "#0072b2",   # deep blue
    "slew":        "#e69f00",   # amber
    "missed_slew": "#d62728",   # red
    "idle":        "#2d2d4e",   # muted blue-grey
}

# Tier → colour + scatter marker size (for the sky map)
_TIER_COLOR = {
    -1: "#2d2d50",   # never observed  (dark navy)
     0: "#7b7ba8",   # observed, no tier completed yet  (muted lavender)
     1: "#56b4e9",   # T1 complete
     2: "#009e73",   # T2 complete
     3: "#0072b2",   # T3 complete
}
_TIER_SIZE = {-1: 6, 0: 10, 1: 14, 2: 18, 3: 22}


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _mollweide(ra_deg: np.ndarray, dec_deg: np.ndarray):
    """(RA, Dec) in degrees → (lon, lat) in radians for Mollweide projection.

    Centred at RA=180° with East to the left (standard astronomical convention).
    """
    lon = -np.radians(np.asarray(ra_deg, dtype=float) - 180.0)
    lat = np.radians(np.asarray(dec_deg, dtype=float))
    lon = np.clip(lon, -np.pi + 1e-9, np.pi - 1e-9)
    lat = np.clip(lat, -np.pi / 2 + 1e-9, np.pi / 2 - 1e-9)
    return lon, lat


# ---------------------------------------------------------------------------
# Timeline reconstruction
# ---------------------------------------------------------------------------

def build_timeline(
    obs_df: pd.DataFrame,
    total_days: float,
    init_ra: float,
    init_dec: float,
) -> list[dict]:
    """Reconstruct the ordered activity sequence from the observation log.

    Returns a list of dicts, each describing one activity segment::

        start, end      — float, mission days
        type            — 'idle' | 'slew' | 'missed_slew' |
                          't1_obs' | 't2_obs' | 't3_obs'
        ra, dec         — telescope pointing *at* (or heading *toward*) this target
        from_ra/from_dec — start of this pointing (same as ra/dec for non-slew)
    """
    segments: list[dict] = []
    prev_end  = 0.0
    prev_ra   = float(init_ra)
    prev_dec  = float(init_dec)

    for _, row in obs_df.sort_values("mission_day").iterrows():
        half_dur     = float(row["obs_duration_days"]) / 2.0
        slew_d       = float(row["slew_days"])
        # window_start: when we start slewing toward this event
        window_start = max(prev_end, float(row["mission_day"]) - half_dur - slew_d)
        slew_end     = window_start + slew_d
        obs_end      = slew_end + (float(row["obs_duration_days"]) if not row["missed"] else 0.0)

        # ── Idle gap ──
        if window_start > prev_end + 1e-6:
            segments.append({
                "start": prev_end, "end": window_start,
                "type": "idle",
                "ra": prev_ra, "dec": prev_dec,
                "from_ra": prev_ra, "from_dec": prev_dec,
            })

        # ── Slew (or missed-slew) ──
        segments.append({
            "start":    window_start,
            "end":      slew_end,
            "type":     "missed_slew" if row["missed"] else "slew",
            "ra":       float(row["ra"]),
            "dec":      float(row["dec"]),
            "from_ra":  prev_ra,
            "from_dec": prev_dec,
        })

        if not row["missed"]:
            # tier being worked toward = tier_before + 1 (capped at 3)
            t = min(int(row["tier_before"]) + 1, 3)
            segments.append({
                "start":     slew_end,
                "end":       obs_end,
                "type":      f"t{t}_obs",
                "ra":        float(row["ra"]),
                "dec":       float(row["dec"]),
                "from_ra":   float(row["ra"]),
                "from_dec":  float(row["dec"]),
                "target_id": str(row["target_id"]),
            })

        prev_end = obs_end if not row["missed"] else slew_end
        prev_ra  = float(row["ra"])
        prev_dec = float(row["dec"])

    # Final idle to end of mission
    if prev_end < total_days - 1e-6:
        segments.append({
            "start": prev_end, "end": total_days,
            "type": "idle",
            "ra": prev_ra, "dec": prev_dec,
            "from_ra": prev_ra, "from_dec": prev_dec,
        })

    return segments


def telescope_pos_at(t: float, segments: list[dict]) -> tuple[float, float]:
    """Interpolated telescope (ra, dec) at mission_day *t*."""
    for seg in segments:
        if seg["start"] <= t < seg["end"]:
            if seg["type"] in ("slew", "missed_slew"):
                dur  = seg["end"] - seg["start"]
                frac = np.clip((t - seg["start"]) / max(dur, 1e-9), 0.0, 1.0)
                ra   = seg["from_ra"]  + frac * (seg["ra"]  - seg["from_ra"])
                dec  = seg["from_dec"] + frac * (seg["dec"] - seg["from_dec"])
                return float(ra), float(dec)
            return float(seg["ra"]), float(seg["dec"])
    if segments:
        return float(segments[-1]["ra"]), float(segments[-1]["dec"])
    return 0.0, 0.0


def target_tiers_at(t: float, obs_df: pd.DataFrame) -> dict[str, int]:
    """Return {target_id: current_tier} for all targets observed up to day *t*."""
    past = obs_df[obs_df["mission_day"] <= t]
    if past.empty:
        return {}
    return past.groupby("target_id")["tier_after"].max().to_dict()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    *,
    model_path: str | None,
    agent_name: str,
    days: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run one full episode and return (obs_df, targets_df, total_days).

    Supports either a trained RL model (``model_path`` given) or a named
    baseline (``agent_name`` from the built-in set).
    """
    from ariel_rl.data.preprocess_targets import build_target_table
    from ariel_rl.envs.ariel_env import ArielEnv
    from ariel_rl.simulator.event_backend import DynamicBackend
    from ariel_rl.utils.config import (
        EnvConfig, MissionConfig, ActionConfig, TopKActionConfig,
        SlewConfig, ObservationConfig, RewardConfig,
    )
    from ariel_rl.data.schemas import MISSION_START_BJD

    cfg = EnvConfig(
        mission=MissionConfig(start_bjd=MISSION_START_BJD, lifetime_days=days),
        slew=SlewConfig(),
        action=ActionConfig(type="topk", topk=TopKActionConfig(k=50)),
        observation=ObservationConfig(normalise=True),
        reward=RewardConfig(),
    )

    targets = build_target_table()
    backend = DynamicBackend(targets)
    env     = ArielEnv(config=cfg, targets=targets, backend=backend)

    # ── Pick agent ──────────────────────────────────────────────────────
    if model_path:
        from ariel_rl.agents.rl_agent import RLAgentWrapper
        print(f"  Loading RL model from {model_path} …")
        agent = RLAgentWrapper.load(model_path, name=agent_name)
    else:
        agent = _make_baseline(agent_name)

    # ── Run ─────────────────────────────────────────────────────────────
    print(f"  Running {agent_name} for {days:.0f} days …")
    obs, info = env.reset(seed=seed)
    agent.reset()
    terminated = truncated = False
    step = 0
    while not (terminated or truncated) and step < 200_000:
        action = agent.act(obs, info)
        obs, _, terminated, truncated, info = env.step(action)
        step += 1

    obs_df = env.state.obs_log_df()
    print(f"  {len(obs_df)} observations recorded.")
    return obs_df, targets, float(days)


def _make_baseline(name: str):
    """Instantiate a named baseline agent."""
    _registry = {
        "RandomValid":      "ariel_rl.baselines.random_valid.RandomValid",
        "GreedyValue":      "ariel_rl.baselines.greedy_value.GreedyValue",
        "GreedyBalanced":   "ariel_rl.baselines.greedy_balanced.GreedyBalanced",
        "EarliestDeadline": "ariel_rl.baselines.earliest_deadline.EarliestDeadline",
        "SmartGreedy":      "ariel_rl.baselines.smart_greedy.SmartGreedy",
    }
    if name not in _registry:
        avail = ", ".join(_registry)
        raise ValueError(f"Unknown baseline {name!r}.  Available: {avail}")
    module_path, cls_name = _registry[name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()


# ---------------------------------------------------------------------------
# GIF maker
# ---------------------------------------------------------------------------

def make_gif(
    obs_df: pd.DataFrame,
    targets: pd.DataFrame,
    total_days: float,
    out_path: Path,
    run_name: str = "",
    n_frames: int = 250,
    fps: int = 10,
    dpi: int = 100,
) -> None:
    """Render and save the animated GIF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as manim
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    # ── Build timeline from obs_log ──────────────────────────────────────
    print("Building activity timeline …")
    init_ra  = float(targets["ra"].iloc[0])
    init_dec = float(targets["dec"].iloc[0])
    segments = build_timeline(obs_df, total_days, init_ra, init_dec)

    frame_times = np.linspace(0.0, total_days, n_frames)

    # ── Pre-compute static target positions in Mollweide coordinates ─────
    all_tids = targets["target_id"].values
    all_ra   = targets["ra"].values.astype(float)
    all_dec  = targets["dec"].values.astype(float)
    all_lons, all_lats = _mollweide(all_ra, all_dec)
    tid_to_idx = {tid: i for i, tid in enumerate(all_tids)}

    init_colors = [_TIER_COLOR[-1]] * len(all_tids)
    init_sizes  = [float(_TIER_SIZE[-1])] * len(all_tids)

    # ── Max achievable tier counts (static) ──────────────────────────────
    mt = targets["max_tier"].astype(int) if "max_tier" in targets.columns else pd.Series([3] * len(targets))
    n_t1_max = int((mt >= 1).sum())
    n_t2_max = int((mt >= 2).sum())
    n_t3_max = int((mt >= 3).sum())

    # ── Figure & axes layout ─────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), facecolor=_BG)
    gs  = fig.add_gridspec(
        2, 1,
        height_ratios=[3.2, 1],
        hspace=0.10,
        left=0.03, right=0.97, top=0.93, bottom=0.07,
    )
    ax_sky = fig.add_subplot(gs[0], projection="mollweide")
    ax_bar = fig.add_subplot(gs[1])

    # ── Sky axes styling ─────────────────────────────────────────────────
    ax_sky.set_facecolor(_BG)
    ax_sky.tick_params(colors=_FG, labelsize=11)
    for spine in ax_sky.spines.values():
        spine.set_edgecolor("#2a2a55")
    ax_sky.grid(True, color="#1e1e3a", linewidth=0.5, linestyle="--", alpha=0.7)

    # ── Bar axes styling ─────────────────────────────────────────────────
    ax_bar.set_facecolor(_BG)
    ax_bar.set_yticks([])
    ax_bar.set_xlim(0, total_days)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlabel("Mission day", color=_FG, fontsize=13)
    ax_bar.tick_params(axis="x", colors=_FG, labelsize=11)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#2a2a55")

    # ── Static scatter: all targets ──────────────────────────────────────
    scat = ax_sky.scatter(
        all_lons, all_lats,
        c=init_colors, s=init_sizes,
        alpha=0.9, linewidths=0, zorder=3,
    )

    # ── Crosshair marker ─────────────────────────────────────────────────
    xh_lon, xh_lat = _mollweide(np.array([init_ra]), np.array([init_dec]))
    crosshair, = ax_sky.plot(
        xh_lon, xh_lat,
        marker="+", color="white",
        markersize=20, markeredgewidth=2.5,
        linestyle="None", zorder=11,
    )
    ring, = ax_sky.plot(
        xh_lon, xh_lat,
        marker="o", color="white",
        markersize=28, markeredgewidth=1.0,
        fillstyle="none", linestyle="None", zorder=10,
    )

    # ── Title & info panel ────────────────────────────────────────────────
    fig.suptitle(
        run_name, color=_FG, fontsize=17, fontweight="bold", y=0.97,
    )
    # Multi-line info panel: day counter + live tier completion counts.
    # Rendered in the upper-right corner of the sky axes.
    info_label = ax_sky.text(
        0.99, 0.97, "", transform=ax_sky.transAxes,
        color=_FG, fontsize=12, va="top", ha="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#111125",
                  edgecolor="#333366", alpha=0.88),
    )

    # ── Sky legend: tier colours (lower-left) ────────────────────────────
    _leg_kw = dict(
        facecolor="#111125", edgecolor="#333366",
        labelcolor=_FG, fontsize=12, framealpha=0.88,
        handlelength=1.6,
    )
    tier_legend = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=_TIER_COLOR[-1], markersize=9,  label="Unobserved"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=_TIER_COLOR[0],  markersize=11, label="In progress"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=_TIER_COLOR[1],  markersize=13, label="T1 complete"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=_TIER_COLOR[2],  markersize=15, label="T2 complete"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=_TIER_COLOR[3],  markersize=17, label="T3 complete"),
    ]
    # Create tier legend and preserve it as a figure artist so the second
    # ax.legend() call (for the telescope) doesn't overwrite it.
    leg_tier = ax_sky.legend(handles=tier_legend, loc="lower left", **_leg_kw)
    ax_sky.add_artist(leg_tier)

    # ── Sky legend: telescope marker (lower-right, separate) ─────────────
    tel_legend = [
        Line2D([0], [0], marker="+", color="white",
               markersize=14, markeredgewidth=2.5, label="Telescope pointing"),
    ]
    ax_sky.legend(handles=tel_legend, loc="lower right",
                  facecolor="#111125", edgecolor="#333366",
                  labelcolor=_FG, fontsize=12, framealpha=0.88,
                  handlelength=1.2)

    # ── Bar legend ────────────────────────────────────────────────────────
    bar_legend = [
        Rectangle((0, 0), 1, 1, color=_SEG_COLORS["t1_obs"],      label="T1 obs"),
        Rectangle((0, 0), 1, 1, color=_SEG_COLORS["t2_obs"],      label="T2 obs"),
        Rectangle((0, 0), 1, 1, color=_SEG_COLORS["t3_obs"],      label="T3 obs"),
        Rectangle((0, 0), 1, 1, color=_SEG_COLORS["slew"],        label="Slew"),
        Rectangle((0, 0), 1, 1, color=_SEG_COLORS["idle"],        label="Idle"),
        Rectangle((0, 0), 1, 1, color=_SEG_COLORS["missed_slew"], label="Missed"),
    ]
    ax_bar.legend(
        handles=bar_legend, loc="upper left",
        facecolor="#111125", edgecolor="#333366",
        labelcolor=_FG, fontsize=12, framealpha=0.88,
        ncol=6, handlelength=1.2,
    )

    # ── Cursor line (re-positioned each frame) ────────────────────────────
    cursor = ax_bar.axvline(0.0, color="white", linewidth=1.5, zorder=10, alpha=0.9)

    # ── Update function ───────────────────────────────────────────────────
    active_bars: list = []   # Rectangle patches added to ax_bar

    def update(fi: int):
        nonlocal active_bars
        t = float(frame_times[fi])

        # ── Update target colours ─────────────────────────────────────
        tier_map = target_tiers_at(t, obs_df)
        new_c = list(init_colors)
        new_s = list(init_sizes)
        for tid, tier in tier_map.items():
            idx = tid_to_idx.get(tid)
            if idx is not None:
                tc = int(np.clip(tier, -1, 3))
                new_c[idx] = _TIER_COLOR[tc]
                new_s[idx] = float(_TIER_SIZE[tc])
        scat.set_facecolors(new_c)
        scat.set_sizes(new_s)

        # ── Update crosshair ──────────────────────────────────────────
        ra, dec = telescope_pos_at(t, segments)
        lon, lat = _mollweide(np.array([ra]), np.array([dec]))
        crosshair.set_data(lon, lat)
        ring.set_data(lon, lat)

        # ── Info panel: day + live tier counts ───────────────────────
        n_t1 = sum(1 for v in tier_map.values() if v >= 1)
        n_t2 = sum(1 for v in tier_map.values() if v >= 2)
        n_t3 = sum(1 for v in tier_map.values() if v >= 3)
        p1 = n_t1 / n_t1_max * 100 if n_t1_max else 0
        p2 = n_t2 / n_t2_max * 100 if n_t2_max else 0
        p3 = n_t3 / n_t3_max * 100 if n_t3_max else 0
        info_label.set_text(
            f"Day  {t:6.1f} / {total_days:.0f}\n"
            f"T1  {n_t1:4d}/{n_t1_max}  ({p1:4.1f}%)\n"
            f"T2  {n_t2:4d}/{n_t2_max}  ({p2:4.1f}%)\n"
            f"T3  {n_t3:4d}/{n_t3_max}  ({p3:4.1f}%)"
        )

        # ── Activity bar ──────────────────────────────────────────────
        for patch in active_bars:
            patch.remove()
        active_bars.clear()

        for seg in segments:
            if seg["start"] >= t:
                break
            end   = min(seg["end"], t)
            width = end - seg["start"]
            if width <= 0:
                continue
            color = _SEG_COLORS.get(seg["type"], "#888888")
            rect  = Rectangle(
                (seg["start"], 0.08), width, 0.84,
                color=color, linewidth=0, zorder=5,
            )
            ax_bar.add_patch(rect)
            active_bars.append(rect)

        cursor.set_xdata([t, t])

        return [scat, crosshair, ring, info_label, cursor] + active_bars

    # ── Animate & save ────────────────────────────────────────────────────
    print(f"Rendering {n_frames} frames at {fps} fps …")
    anim = manim.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000 // fps,
        blit=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = manim.PillowWriter(fps=fps, metadata={"title": run_name})
    print(f"Saving → {out_path}  (this may take a minute …)")
    anim.save(str(out_path), writer=writer, dpi=dpi,
              savefig_kwargs={"facecolor": _BG})
    plt.close(fig)
    print(f"Done!  {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a telescope-schedule animation as a GIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # RL model, 1-year mission, 250 frames at 10 fps:
  python scripts/animate_schedule.py \\
      --model-path outputs/transformer_v1/final_model.zip \\
      --days 365 --run-name transformer_1yr

  # SmartGreedy baseline, 60-day mission, slower (8 fps / 200 frames):
  python scripts/animate_schedule.py \\
      --agent SmartGreedy --days 60 --n-frames 200 --fps 8 \\
      --run-name smartgreedy_60d
""",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--model-path", metavar="PATH",
        help="Path to a saved MaskablePPO model (.zip).  Activates RL mode.",
    )
    src.add_argument(
        "--agent", metavar="NAME",
        choices=["RandomValid", "GreedyValue", "GreedyBalanced",
                 "EarliestDeadline", "SmartGreedy"],
        help="Named baseline agent to animate instead of an RL model.",
    )

    p.add_argument("--run-name", default="schedule",
                   help="Stem for the output filename.")
    p.add_argument("--days", type=float, default=365.0,
                   help="Mission duration to simulate (days).")
    p.add_argument("--seed", type=int, default=42,
                   help="Episode random seed.")
    p.add_argument("--n-frames", type=int, default=250,
                   help="Total number of animation frames.")
    p.add_argument("--fps", type=int, default=10,
                   help="GIF playback speed (frames per second).")
    p.add_argument("--dpi", type=int, default=100,
                   help="Output resolution (dots per inch).")
    p.add_argument("--out-dir", default="plots/animations",
                   help="Directory for the output GIF.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Infer a sensible run-name from the model path if not set
    run_name = args.run_name
    if run_name == "schedule":
        if args.model_path:
            run_name = Path(args.model_path).parent.name or "rl_agent"
        else:
            run_name = args.agent.lower()
        run_name = f"{run_name}_{int(args.days)}d"

    print(f"\n=== Ariel Schedule Animator ===")
    print(f"  Run name : {run_name}")
    print(f"  Agent    : {args.model_path or args.agent}")
    print(f"  Duration : {args.days:.0f} days")
    print(f"  Frames   : {args.n_frames}  |  FPS: {args.fps}  |  DPI: {args.dpi}")
    duration_s = args.n_frames / args.fps
    print(f"  GIF duration ≈ {duration_s:.1f} s  "
          f"({args.days / args.n_frames:.2f} mission-days / frame)")

    agent_name = run_name if args.model_path else args.agent
    obs_df, targets, total_days = run_episode(
        model_path=args.model_path,
        agent_name=agent_name,
        days=args.days,
        seed=args.seed,
    )

    out_path = Path(args.out_dir) / f"{run_name}.gif"
    make_gif(
        obs_df=obs_df,
        targets=targets,
        total_days=total_days,
        out_path=out_path,
        run_name=run_name,
        n_frames=args.n_frames,
        fps=args.fps,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
