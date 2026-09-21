from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecasts(pred_df: pd.DataFrame, close_pred_cols: Iterable[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 6))
    plt.plot(pred_df.index, pred_df["actual_close"], label="Actual Close", linewidth=2)
    for col in close_pred_cols:
        if col in pred_df.columns:
            plt.plot(pred_df.index, pred_df[col], label=col.replace("_close_pred", ""), alpha=0.85)
    plt.title("Gold Price Forecast: Actual vs Forecast Models")
    plt.xlabel("Date")
    plt.ylabel("Gold Price")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_equity(equity_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 6))
    for col in equity_df.columns:
        plt.plot(equity_df.index, equity_df[col], label=col)
    plt.title("Trading Backtest: Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity Curve")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(fi_df: pd.DataFrame, output_path: str | Path, top_n: int = 20) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fi_df.empty:
        return
    d = fi_df.sort_values("importance", ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(d["feature"], d["importance"])
    plt.title("Selected Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pipeline(output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    boxes = [
        "1. Download data\nyfinance + FRED",
        "2. Clean & align\nmissing/outlier checks",
        "3. Feature engineering\nreturns, lags, RSI, EMA, macro",
        "4. Data filtering\nwinsorize + correlation filter",
        "5. Feature selection\nRF importance top-k",
        "6. SARIMAX\nlinear + exogenous X",
        "7. LSTM residual\nnonlinear residual pattern",
        "8. Forecast + trading\nevaluation + reports",
    ]
    plt.figure(figsize=(15, 4.8))
    ax = plt.gca()
    ax.axis("off")
    x_positions = [0.03, 0.16, 0.29, 0.42, 0.55, 0.68, 0.81, 0.94]
    for i, (x, txt) in enumerate(zip(x_positions, boxes)):
        ax.text(x, 0.55, txt, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", lw=1))
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x_positions[i + 1] - 0.055, 0.55), xytext=(x + 0.055, 0.55),
                        arrowprops=dict(arrowstyle="->", lw=1.2))
    plt.title("GoldForecast-Hybrid System: SARIMAX-LSTM Forecasting Pipeline", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
