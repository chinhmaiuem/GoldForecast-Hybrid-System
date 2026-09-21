from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(clean_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, list[str], list[str]]:
    df = clean_df.copy().sort_index()

    # Target: short-term gold log-return. Forecasted returns are later converted back to close price.
    df["actual_close"] = df["gold_close"]
    df["close_lag_1"] = df["gold_close"].shift(1)
    df["target_log_return"] = np.log(df["gold_close"] / df["gold_close"].shift(1))

    price_lags = cfg["features"].get("price_lags", [1, 2, 3, 5, 10, 20])
    macro_lags = cfg["features"].get("macro_lags", [1, 2, 5])
    windows = cfg["features"].get("rolling_windows", [5, 10, 20])

    # Price dynamics known up to t-1.
    for lag in price_lags:
        df[f"gold_return_lag_{lag}"] = df["target_log_return"].shift(lag)

    for w in windows:
        past_close = df["gold_close"].shift(1)
        df[f"gold_vol_{w}"] = df["target_log_return"].shift(1).rolling(w).std()
        df[f"gold_ma_gap_{w}"] = past_close / past_close.rolling(w).mean() - 1
        df[f"gold_momentum_{w}"] = np.log(past_close / past_close.shift(w))

    rsi_window = cfg["features"].get("rsi_window", 14)
    df[f"gold_rsi_{rsi_window}"] = calc_rsi(df["gold_close"], rsi_window).shift(1)

    for w in cfg["features"].get("ema_windows", [10, 20]):
        ema = df["gold_close"].ewm(span=w, adjust=False).mean()
        df[f"gold_ema_gap_{w}"] = (df["gold_close"] / ema - 1).shift(1)

    # Macroeconomic variables inspired by the SARIMA-LSTM-RF gold paper.
    # They are shifted/lagged to avoid using current-day information as if known before the forecast.
    macro_base = []
    if "usd_index" in df.columns:
        df["usd_index_return"] = np.log(df["usd_index"] / df["usd_index"].shift(1))
        macro_base += ["usd_index_return"]
    if "oil_price" in df.columns:
        df["oil_return"] = np.log(df["oil_price"] / df["oil_price"].shift(1))
        macro_base += ["oil_return"]
    if "sp500" in df.columns:
        df["sp500_return"] = np.log(df["sp500"] / df["sp500"].shift(1))
        macro_base += ["sp500_return"]
    if "bond_yield" in df.columns:
        df["bond_yield_diff"] = df["bond_yield"].diff()
        macro_base += ["bond_yield_diff"]
    if "fed_rate" in df.columns:
        df["fed_rate_diff"] = df["fed_rate"].diff()
        df["fed_rate_level"] = df["fed_rate"]
        macro_base += ["fed_rate_diff", "fed_rate_level"]
    if "cpi" in df.columns:
        df["cpi_mom"] = np.log(df["cpi"] / df["cpi"].shift(21))  # approx monthly daily-aligned change
        df["cpi_yoy"] = np.log(df["cpi"] / df["cpi"].shift(252))
        df["cpi_level"] = df["cpi"]
        macro_base += ["cpi_mom", "cpi_yoy", "cpi_level"]

    macro_feature_cols = []
    for col in macro_base:
        for lag in macro_lags:
            name = f"{col}_lag_{lag}"
            df[name] = df[col].shift(lag)
            macro_feature_cols.append(name)

    price_feature_cols = [
        c for c in df.columns
        if c.startswith("gold_return_lag_")
        or c.startswith("gold_vol_")
        or c.startswith("gold_ma_gap_")
        or c.startswith("gold_momentum_")
        or c.startswith("gold_rsi_")
        or c.startswith("gold_ema_gap_")
    ]

    keep_cols = ["actual_close", "close_lag_1", "target_log_return"] + price_feature_cols + macro_feature_cols
    dataset = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()
    dataset.index.name = "date"

    return dataset, price_feature_cols, macro_feature_cols
