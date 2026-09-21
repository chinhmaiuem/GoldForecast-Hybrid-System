from __future__ import annotations

from typing import Dict, Any, Tuple
import pandas as pd


def _to_ts(value, fallback=None):
    if value is None:
        return fallback
    return pd.to_datetime(value)


def _ensure_business_day_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a time series to a regular business-day DatetimeIndex.

    statsmodels SARIMAX requires a supported index with an explicit or inferable
    frequency; a raw DatetimeIndex that has gaps or no frequency metadata can
    trigger ``ValueError: No supported index is available`` during forecasting.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    idx = pd.DatetimeIndex(df.index)
    if idx.freq is not None:
        return df

    try:
        inferred = pd.infer_freq(idx)
    except Exception:
        inferred = None

    if inferred is not None:
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index, freq=inferred)
        return df

    first = idx.min()
    last = idx.max()
    target = pd.date_range(start=first, end=last, freq="B")
    if len(target) == 0:
        return df

    df = df.reindex(target)
    df = df.ffill().bfill()
    df.index = pd.DatetimeIndex(df.index, freq="B")
    return df


def split_train_validation_test(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split used to avoid trading-parameter leakage.

    Train       : used to fit forecasting models for validation.
    Validation  : used only to choose trading thresholds / strategy settings.
    Test        : final out-of-sample period. No parameter selection is performed here.
    """
    data_cfg = cfg["data"]
    train_start = _to_ts(data_cfg.get("train_start"))
    train_end = _to_ts(data_cfg.get("train_end"))
    val_start = _to_ts(data_cfg.get("validation_start"))
    val_end = _to_ts(data_cfg.get("validation_end"))
    test_start = _to_ts(data_cfg.get("test_start"))
    test_end = _to_ts(data_cfg.get("test_end"), fallback=df.index.max())

    train = df.loc[(df.index >= train_start) & (df.index <= train_end)].copy()
    validation = df.loc[(df.index >= val_start) & (df.index <= val_end)].copy()
    test = df.loc[(df.index >= test_start) & (df.index <= test_end)].copy()

    train = _ensure_business_day_index(train)
    validation = _ensure_business_day_index(validation)
    test = _ensure_business_day_index(test)

    if train.empty:
        raise RuntimeError("Train split is empty. Check config dates.")
    if validation.empty:
        raise RuntimeError("Validation split is empty. Check config dates.")
    if test.empty:
        raise RuntimeError("Test split is empty. Check config dates.")
    if not (train.index.max() < validation.index.min() < validation.index.max() < test.index.min()):
        raise RuntimeError("Train/validation/test overlap or wrong chronological order detected.")

    return train, validation, test


def split_time_series(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible train/test split.

    This function is kept for older scripts. The methodology-fixed pipeline uses
    split_train_validation_test instead.
    """
    data_cfg = cfg["data"]
    train_start = _to_ts(data_cfg.get("train_start"))
    train_end = _to_ts(data_cfg.get("validation_end") or data_cfg.get("train_end"))
    test_start = _to_ts(data_cfg.get("test_start"))
    test_end = _to_ts(data_cfg.get("test_end"), fallback=df.index.max())

    train = df.loc[(df.index >= train_start) & (df.index <= train_end)].copy()
    test = df.loc[(df.index >= test_start) & (df.index <= test_end)].copy()

    train = _ensure_business_day_index(train)
    test = _ensure_business_day_index(test)

    if train.empty:
        raise RuntimeError("Train split is empty. Check config dates.")
    if test.empty:
        raise RuntimeError("Test split is empty. Check config dates.")
    if train.index.max() >= test.index.min():
        raise RuntimeError("Train/test overlap detected. Check date ranges.")
    return train, test
