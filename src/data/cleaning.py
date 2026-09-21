from __future__ import annotations

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def clean_and_align(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ffill_limit = cfg["preprocessing"].get("ffill_limit", 5)
    min_ratio = cfg["preprocessing"].get("min_non_missing_ratio", 0.80)

    report_rows = []
    original_rows = len(df)
    report_rows.append({"step": "raw_rows", "value": original_rows})
    report_rows.append({"step": "raw_columns", "value": len(df.columns)})

    df = df.copy().sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Remove impossible gold prices.
    invalid_gold = int((df["gold_close"] <= 0).sum()) if "gold_close" in df.columns else 0
    df = df[df["gold_close"] > 0]
    report_rows.append({"step": "removed_non_positive_gold", "value": invalid_gold})

    missing_before = df.isna().sum().sum()
    report_rows.append({"step": "missing_cells_before_ffill", "value": int(missing_before)})

    # FRED monthly variables and some markets have different calendars; use limited ffill to avoid very long fake values.
    df = df.ffill(limit=ffill_limit)

    # CPI and Fed are monthly by nature. They are allowed to forward-fill for daily alignment after initial release.
    for col in ["fed_rate", "cpi"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    row_non_missing_ratio = df.notna().mean(axis=1)
    low_quality_rows = int((row_non_missing_ratio < min_ratio).sum())
    df = df[row_non_missing_ratio >= min_ratio]
    report_rows.append({"step": "removed_low_non_missing_ratio_rows", "value": low_quality_rows})

    # Core target column must exist.
    df = df.dropna(subset=["gold_close"])
    missing_after = df.isna().sum().sum()
    report_rows.append({"step": "missing_cells_after_cleaning", "value": int(missing_after)})
    report_rows.append({"step": "final_rows_after_cleaning", "value": len(df)})

    # Numeric conversion and inf handling.
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    report = pd.DataFrame(report_rows)
    return df, report
