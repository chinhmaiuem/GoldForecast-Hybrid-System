from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Any
import warnings

import pandas as pd


def _today_str() -> str:
    return date.today().isoformat()


def _download_yfinance_series(ticker: str, name: str, start: str, end: str | None) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned empty data for ticker {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    col = "Close" if "Close" in df.columns else df.columns[0]
    s = df[col].copy()
    s.name = name
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def _download_fred_series(code: str, name: str, start: str, end: str | None) -> pd.Series:
    from pandas_datareader import data as web

    end = end or _today_str()
    s = web.DataReader(code, "fred", start, end)[code]
    s.name = name
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def download_market_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    start = cfg["data"]["start"]
    end = cfg["data"].get("end") or _today_str()

    series = []
    errors = []

    for name, ticker in cfg["data"]["yfinance"].items():
        try:
            print(f"[download] yfinance {name}: {ticker}")
            series.append(_download_yfinance_series(ticker, name, start, end))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name} ({ticker}): {exc}")

    for name, code in cfg["data"].get("fred", {}).items():
        try:
            print(f"[download] FRED {name}: {code}")
            series.append(_download_fred_series(code, name, start, end))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Could not download FRED {name}={code}. Error: {exc}")
            errors.append(f"{name} ({code}): {exc}")

    if not series:
        raise RuntimeError("No data downloaded. Check internet connection and tickers.")

    df = pd.concat(series, axis=1).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "date"

    if "gold_close" not in df.columns:
        raise RuntimeError("gold_close is required but was not downloaded.")

    if errors:
        print("[download] Some series failed. The pipeline will continue if core columns exist:")
        for err in errors:
            print("  -", err)

    return df


def save_raw_data(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8-sig")
    print(f"[download] saved raw data -> {path}")
