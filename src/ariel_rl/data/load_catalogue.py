"""
Load the Ariel MCS catalogue CSV and return a clean, renamed DataFrame.

The output contains only the columns defined in schemas.RAW_COL_MAP,
renamed to the canonical internal names.  No derived columns are added
here — that happens in preprocess_targets.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ariel_rl.data.schemas import RAW_COL_MAP, TARGET_DTYPES

_REPO_ROOT = Path(__file__).parents[3]   # src/ariel_rl/data/ → repo root
_DATA_RAW  = _REPO_ROOT / "data" / "raw"

# Try the canonical name first, then fall back to any MCS CSV in data/raw/
# so the script works regardless of what the file was named locally.
_CANDIDATE_NAMES = [
    "Ariel_MCS_Known_2025-08-18.csv",
    "MCS.csv",
]

def _find_default_csv() -> Path:
    for name in _CANDIDATE_NAMES:
        p = _DATA_RAW / name
        if p.exists():
            return p
    # Last resort: any *.csv directly in data/raw/
    csvs = sorted(_DATA_RAW.glob("*.csv"))
    if csvs:
        return csvs[0]
    # Return the canonical name so the FileNotFoundError message is helpful
    return _DATA_RAW / _CANDIDATE_NAMES[0]


def load_mcs(path: str | Path | None = None) -> pd.DataFrame:
    """Load the MCS CSV and return a tidy DataFrame with canonical column names.

    Parameters
    ----------
    path:
        Path to the raw CSV.  Defaults to ``data/raw/Ariel_MCS_Known_2025-08-18.csv``
        relative to the repository root.

    Returns
    -------
    pd.DataFrame
        One row per target, columns from ``schemas.RAW_COL_MAP`` values.
        Rows with missing ``period`` or ``epoch`` are dropped (shouldn't
        happen per the data, but guards against future catalogue updates).
    """
    csv_path = Path(path) if path is not None else _find_default_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MCS catalogue not found at {csv_path}.  "
            f"Place a CSV in {_DATA_RAW}/ or pass path= explicitly."
        )

    raw = pd.read_csv(csv_path, low_memory=False)

    # Keep only mapped columns
    available = [c for c in RAW_COL_MAP if c in raw.columns]
    missing = [c for c in RAW_COL_MAP if c not in raw.columns]
    if missing:
        import warnings
        warnings.warn(f"Columns not found in CSV (will be NaN): {missing}", stacklevel=2)

    df = raw[available].rename(columns=RAW_COL_MAP).copy()

    # Add any columns that were in the map but missing from the CSV as NaN
    for new_name in RAW_COL_MAP.values():
        if new_name not in df.columns:
            df[new_name] = np.nan

    # ---------- type coercion ----------
    for col, dtype in TARGET_DTYPES.items():
        if col not in df.columns:
            continue
        try:
            if dtype in ("string",):
                df[col] = df[col].astype("string")
            elif dtype in ("Int64",):
                df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
            elif dtype in ("float64",):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif dtype in ("bool",):
                df[col] = df[col].astype("boolean")
        except Exception:
            pass  # leave as-is; caller will see NaN / object

    # ---------- essential rows ----------
    before = len(df)
    df = df.dropna(subset=["period", "epoch"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        import warnings
        warnings.warn(f"Dropped {dropped} rows with missing period/epoch.", stacklevel=2)

    # ---------- convenience: integer target index ----------
    df.insert(0, "target_idx", np.arange(len(df), dtype=np.int32))

    return df


def load_mcs_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Return the full raw CSV with no column selection or renaming."""
    csv_path = Path(path) if path is not None else _find_default_csv()
    return pd.read_csv(csv_path, low_memory=False)
