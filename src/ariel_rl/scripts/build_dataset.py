"""
CLI script: build processed targets + events and cache to Parquet.

Usage
-----
    python -m ariel_rl.scripts.build_dataset
    python -m ariel_rl.scripts.build_dataset --csv path/to/custom.csv --force
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Ariel target and event datasets.")
    parser.add_argument("--csv", type=Path, default=None, help="Path to raw MCS CSV.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/processed"),
        help="Output directory for Parquet files.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if Parquet exists.")
    parser.add_argument(
        "--mission-start-bjd", type=float, default=None,
        help="BJD of mission start (default: ~2029-01-01).",
    )
    args = parser.parse_args()

    # Lazy import so the script fails fast if deps are missing
    from ariel_rl.data.preprocess_targets import build_target_table
    from ariel_rl.data.schemas import MISSION_END_BJD, MISSION_START_BJD
    from ariel_rl.simulator.event_generator import generate_events, save_events

    args.out_dir.mkdir(parents=True, exist_ok=True)

    target_parquet = args.out_dir / "targets.parquet"
    event_parquet  = args.out_dir / "events.parquet"

    # ---- targets ----
    if target_parquet.exists() and not args.force:
        print(f"Loading cached targets from {target_parquet}")
        import pandas as pd
        targets = pd.read_parquet(target_parquet)
    else:
        print("Building target table …")
        t0 = time.time()
        targets = build_target_table(csv_path=args.csv)
        targets.to_parquet(target_parquet, index=False)
        print(f"  {len(targets)} targets  ({time.time()-t0:.1f}s)  → {target_parquet}")

    # ---- events ----
    if event_parquet.exists() and not args.force:
        print(f"Loading cached events from {event_parquet}")
        import pandas as pd
        events = pd.read_parquet(event_parquet)
    else:
        m_start = args.mission_start_bjd or MISSION_START_BJD
        m_end   = m_start + (MISSION_END_BJD - MISSION_START_BJD)
        print(f"Generating events (BJD {m_start:.1f} → {m_end:.1f}) …")
        t0 = time.time()
        events = generate_events(targets, mission_start=m_start, mission_end=m_end)
        save_events(events, event_parquet)
        print(f"  {len(events)} events  ({time.time()-t0:.1f}s)  → {event_parquet}")

    # ---- summary ----
    print("\n--- Summary ---")
    print(f"Targets : {len(targets)}")
    print(f"Events  : {len(events)}")
    print(f"Events per target (mean): {len(events)/len(targets):.1f}")
    print(f"Event types: {events['event_type'].value_counts().to_dict()}")
    if "population_bin" in targets.columns:
        from ariel_rl.data.population_bins import bin_summary
        print("\nTop 10 population bins:")
        print(bin_summary(targets).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
