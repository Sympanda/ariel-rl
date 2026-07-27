"""
Regenerate the poster comparison figure directly from results.csv.

Usage
-----
    python scripts/plot_from_csv.py                          # uses plots/paper/results.csv
    python scripts/plot_from_csv.py --csv plots/paper/results.csv --out-dir plots/paper/
    python scripts/plot_from_csv.py --days 365               # filter to a specific run length
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

_COLORS = {
    "RandomValid":      "#9E9E9E",
    "GreedyValue":      "#1976D2",
    "HillClimbing":     "#43A047",
    "Transformer (RL)": "#E64A19",
    "_default":         "#E64A19",
}
_SPINE = "#444444"
_GRID  = "#e0e0e0"


def _apply_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.weight":       "bold",
        "font.size":         16,
        "axes.titlesize":    22,
        "axes.titleweight":  "bold",
        "axes.labelsize":    18,
        "axes.labelweight":  "bold",
        "xtick.labelsize":   16,
        "ytick.labelsize":   16,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "grid.color":        _GRID,
        "grid.linewidth":    0.9,
        "axes.edgecolor":    _SPINE,
        "xtick.color":       _SPINE,
        "ytick.color":       _SPINE,
        "text.color":        "#1a1a1a",
        "axes.labelcolor":   "#1a1a1a",
    })


def _draw(ax, agents, means, stds, colors, title, show_ylabel, multi_ep):
    x = np.arange(len(agents))
    bars = ax.bar(x, means, width=0.58, color=colors, alpha=0.88, linewidth=0, zorder=3)

    if multi_ep:
        ax.errorbar(x, means, yerr=stds,
                    fmt="none", color=_SPINE, capsize=5, capthick=1.8,
                    linewidth=1.8, zorder=4)

    ax.axhline(1.0, color="#888888", linewidth=1.6, linestyle="--", zorder=2)

    for bar, m, s in zip(bars, means, stds):
        pad = (s if multi_ep else 0.0) + 0.025
        ax.text(bar.get_x() + bar.get_width() / 2, m + pad,
                f"×{m:.2f}", ha="center", va="bottom",
                fontsize=15, fontweight="bold", color="#1a1a1a")

    ax.text(x[0], -0.07, "baseline", ha="center", va="top",
            fontsize=12, fontweight="bold", color="#888888",
            transform=ax.get_xaxis_transform())

    ax.set_title(title, pad=10)
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(agents) - 0.4)
    y_top = max(means) * 1.20 + (max(stds) if multi_ep else 0.0)
    ax.set_ylim(0, max(y_top, 1.35))
    ax.axhspan(1.0, ax.get_ylim()[1], color="#E8F5E9", alpha=0.30, zorder=1)
    if show_ylabel:
        ax.set_ylabel("Relative to random (×)", fontsize=16, labelpad=8)
    ax.spines["left"].set_color(_SPINE)
    ax.spines["bottom"].set_color(_SPINE)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot comparison figure from results.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",     default="plots/paper/results.csv",
                   help="Path to the results CSV file.")
    p.add_argument("--out-dir", default="plots/paper",
                   help="Directory to save comparison.png.")
    p.add_argument("--days",    type=float, default=None,
                   help="Filter to rows with this mission duration. "
                        "If omitted the most recent timestamp is used.")
    p.add_argument("--dpi",     type=int, default=150)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    csv_path = ROOT / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load CSV ─────────────────────────────────────────────────────────
    rows: list[dict] = []
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print("ERROR: CSV is empty.", file=sys.stderr)
        sys.exit(1)

    # Filter by days if requested, else pick latest timestamp
    if args.days is not None:
        rows = [r for r in rows if float(r["days"]) == args.days]
        if not rows:
            print(f"ERROR: no rows with days={args.days}", file=sys.stderr)
            sys.exit(1)
    else:
        latest = max(r["timestamp"] for r in rows)
        rows = [r for r in rows if r["timestamp"] == latest]

    # Preserve order as found in the CSV (already insertion-ordered)
    agents = [r["agent"] for r in rows]
    n_ep   = int(rows[0]["n_episodes"])
    days   = float(rows[0]["days"])

    tier_m = [float(r["tier_score_norm"])         for r in rows]
    tier_s = [float(r["tier_score_norm_std"])      for r in rows]
    eff_m  = [float(r["science_efficiency_norm"])  for r in rows]
    eff_s  = [float(r["science_efficiency_norm_std"]) for r in rows]
    cov_m  = [float(r["bin_coverage_norm"])        for r in rows]
    cov_s  = [float(r["bin_coverage_norm_std"])    for r in rows]

    colors = [_COLORS.get(a, _COLORS["_default"]) for a in agents]
    multi  = n_ep > 1

    print(f"Plotting {len(agents)} agents  |  days={days}  |  n_episodes={n_ep}")
    for a, t, e, c in zip(agents, tier_m, eff_m, cov_m):
        print(f"  {a:<24}  tier={t:.3f}×  eff={e:.3f}×  cov={c:.3f}×")

    # ── Figure ────────────────────────────────────────────────────────────
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), facecolor="none")
    fig.patch.set_alpha(0.0)

    for ax, means, stds, title, show_y in [
        (axes[0], tier_m, tier_s, "Tier Completion Score",  True),
        (axes[1], eff_m,  eff_s,  "Science Efficiency",     False),
        (axes[2], cov_m,  cov_s,  "Population Bin Coverage",False),
    ]:
        ax.set_facecolor("none")
        _draw(ax, agents, means, stds, colors, title, show_y, multi)

    fig.legend(
        handles=[Patch(facecolor=_COLORS.get(a, _COLORS["_default"]),
                       alpha=0.88, label=a, linewidth=0) for a in agents],
        loc="lower center", ncol=len(agents), fontsize=16, frameon=False,
        bbox_to_anchor=(0.5, -0.04), handlelength=1.8, handleheight=1.2,
        columnspacing=2.5,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.18, wspace=0.30)

    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
