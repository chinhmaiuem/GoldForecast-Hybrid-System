"""
Bổ sung các chỉ số đánh giá giao dịch nâng cao.

Backtest hiện tại (backtest.py) có Sharpe, drawdown, win_rate.
Module này bổ sung:
- Sortino ratio (chỉ phạt downside, phù hợp hơn Sharpe cho long-only)
- Calmar ratio (annual_return / max_drawdown, dùng nhiều trong fund)
- Monthly returns heatmap (thấy được tính nhất quán theo tháng)
- Regime-conditioned metrics (bull vs bear market)
- Recovery time (thời gian phục hồi sau drawdown)

Cách dùng:
    from src.trading.metrics_extended import extended_backtest_metrics
    ext = extended_backtest_metrics(equity_df, pred_df, strategy_col="SARIMAX_LSTM_LongHold")
    print(ext["summary"])
    ext["monthly_returns"].to_csv("monthly_returns.csv")
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


# ─── Core ratios ────────────────────────────────────────────────────────────

def sortino_ratio(returns: pd.Series, target_return: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Sortino ratio: (mean_return - target) / downside_deviation * sqrt(periods_per_year).
    Ưu điểm so với Sharpe: không phạt upside volatility, chỉ phạt rủi ro giảm.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    excess = returns - target_return
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    return float((excess.mean() / downside.std()) * np.sqrt(periods_per_year))


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calmar ratio: annual_return / abs(max_drawdown).
    Thường dùng để đánh giá fund theo annual basis.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    equity = np.exp(returns.fillna(0).cumsum())
    peak = equity.cummax()
    dd = (equity / peak - 1).min()
    if dd == 0:
        return float("inf")
    annual_ret = np.exp(returns.mean() * periods_per_year) - 1
    return float(annual_ret / abs(dd))


def omega_ratio(returns: pd.Series, target_return: float = 0.0) -> float:
    """
    Omega ratio: sum(positive excess returns) / sum(negative excess returns).
    Omega > 1 → chiến lược tốt hơn target.
    """
    excess = returns.dropna() - target_return
    pos_sum = excess[excess > 0].sum()
    neg_sum = abs(excess[excess < 0].sum())
    if neg_sum == 0:
        return float("inf") if pos_sum > 0 else 1.0
    return float(pos_sum / neg_sum)


def max_drawdown_duration(equity: pd.Series) -> int:
    """Số ngày dài nhất trong trạng thái drawdown (chưa phục hồi về peak cũ)."""
    peak = equity.cummax()
    in_drawdown = equity < peak
    if not in_drawdown.any():
        return 0
    # Tính run lengths của in_drawdown
    runs = in_drawdown.astype(int).diff().ne(0).cumsum()
    dd_runs = runs[in_drawdown]
    return int(dd_runs.value_counts().max()) if not dd_runs.empty else 0


# ─── Monthly returns ─────────────────────────────────────────────────────────

def monthly_returns_table(strategy_returns: pd.Series) -> pd.DataFrame:
    """
    Tính monthly return cho từng tháng/năm.
    Trả về DataFrame với index=year, columns=month (1-12).
    Dùng cho heatmap trong báo cáo.
    """
    r = strategy_returns.copy()
    r.index = pd.to_datetime(r.index)
    monthly = r.resample("ME").apply(lambda x: float(np.exp(x.fillna(0).sum()) - 1) * 100)
    monthly.index.name = "date"
    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return_pct": monthly.values,
    })
    table = pivot.pivot(index="year", columns="month", values="return_pct")
    table.columns = [f"M{c:02d}" for c in table.columns]
    # Thêm cột annual return
    annual = r.resample("YE").apply(lambda x: float(np.exp(x.fillna(0).sum()) - 1) * 100)
    annual.index = annual.index.year
    table["Annual"] = annual
    return table.round(2)


def annual_stats(strategy_returns: pd.Series) -> pd.DataFrame:
    """Thống kê theo từng năm: return, Sharpe, max drawdown."""
    r = strategy_returns.copy()
    r.index = pd.to_datetime(r.index)
    rows = []
    for year, grp in r.groupby(r.index.year):
        equity_yr = np.exp(grp.fillna(0).cumsum())
        peak = equity_yr.cummax()
        dd = float((equity_yr / peak - 1).min() * 100)
        total = float((equity_yr.iloc[-1] - 1) * 100)
        sr = float((grp.mean() / grp.std()) * np.sqrt(252)) if grp.std() > 0 else 0.0
        rows.append({"year": year, "total_return_pct": round(total, 2), "sharpe": round(sr, 3), "max_dd_pct": round(dd, 2)})
    return pd.DataFrame(rows)


# ─── Regime analysis ─────────────────────────────────────────────────────────

def _classify_regime(gold_returns: pd.Series, window: int = 60) -> pd.Series:
    """
    Phân loại Bull/Bear/Sideways dựa trên rolling return trung bình.
    Bull: 60-day rolling return > +1%
    Bear: < -1%
    Sideways: giữa hai ngưỡng
    """
    rolling = gold_returns.rolling(window).sum()  # log return cumulative
    regime = pd.Series("Sideways", index=gold_returns.index)
    regime[rolling > 0.01] = "Bull"
    regime[rolling < -0.01] = "Bear"
    return regime


def regime_metrics(
    strategy_returns: pd.Series,
    actual_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """
    Metrics của chiến lược chia theo regime thị trường vàng.
    Giúp hiểu: chiến lược hoạt động tốt nhất trong điều kiện nào?
    """
    regime = _classify_regime(actual_returns, window)
    rows = []
    for r_name in ["Bull", "Bear", "Sideways"]:
        mask = regime == r_name
        seg = strategy_returns[mask].dropna()
        if len(seg) < 5:
            continue
        equity = np.exp(seg.fillna(0).cumsum())
        total = float((equity.iloc[-1] - 1) * 100)
        sr = float((seg.mean() / seg.std()) * np.sqrt(252)) if seg.std() > 0 else 0.0
        srt = sortino_ratio(seg)
        win = float((seg > 0).mean() * 100)
        rows.append({
            "regime": r_name,
            "n_days": int(mask.sum()),
            "total_return_pct": round(total, 2),
            "sharpe": round(sr, 3),
            "sortino": round(srt, 3),
            "win_rate_pct": round(win, 1),
        })
    return pd.DataFrame(rows)


# ─── Main extended metrics ───────────────────────────────────────────────────

def extended_backtest_metrics(
    equity_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    strategy_cols: Optional[list] = None,
    risk_free_daily: float = 0.0,
) -> Dict[str, Any]:
    """
    Tính toàn bộ extended metrics cho một hoặc nhiều chiến lược.

    Args:
        equity_df: equity curves từ run_backtest().
        pred_df: predictions DataFrame (có actual_return).
        strategy_cols: danh sách cột strategy cần tính. None = tất cả.
        risk_free_daily: lãi suất phi rủi ro hàng ngày (default 4% annual).

    Returns:
        dict với keys:
            "summary": pd.DataFrame các chỉ số tổng hợp
            "monthly_returns": dict {strategy: monthly_table}
            "annual_stats": dict {strategy: annual_df}
            "regime_metrics": dict {strategy: regime_df}
    """
    if strategy_cols is None:
        # Include Buy_Hold so annual/regime/extended tables remain comparable
        # with final_test_trading_metrics.csv.
        strategy_cols = list(equity_df.columns)

    summary_rows = []
    monthly_tables = {}
    annual_tables = {}
    regime_tables = {}

    for col in strategy_cols:
        if col not in equity_df.columns:
            continue

        equity = equity_df[col].dropna()
        if len(equity) < 10:
            continue

        # Tính returns từ equity curve
        ret = equity.pct_change().dropna()
        log_ret = np.log(equity / equity.shift(1)).dropna()
        excess_ret = log_ret - risk_free_daily

        max_dd = float((equity / equity.cummax() - 1).min() * 100)
        total_ret = float((equity.iloc[-1] - 1) * 100)
        annual_factor = 252 / len(log_ret)
        ann_ret = float((equity.iloc[-1] ** annual_factor - 1) * 100)

        summary_rows.append({
            "strategy": col,
            "total_return_pct": round(total_ret, 2),
            "annualized_return_pct": round(ann_ret, 2),
            "sharpe": round(float(excess_ret.mean() / excess_ret.std() * np.sqrt(252)) if excess_ret.std() > 0 else 0.0, 3),
            "sortino": round(sortino_ratio(log_ret), 3),
            "calmar": round(calmar_ratio(log_ret), 3),
            "omega": round(omega_ratio(log_ret), 3),
            "max_drawdown_pct": round(max_dd, 2),
            "max_dd_duration_days": max_drawdown_duration(equity),
            "win_rate_pct": round(float((log_ret > 0).mean() * 100), 1),
        })

        monthly_tables[col] = monthly_returns_table(log_ret)
        annual_tables[col] = annual_stats(log_ret)

        if "actual_return" in pred_df.columns:
            regime_tables[col] = regime_metrics(log_ret, pred_df["actual_return"])

    summary_df = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    return {
        "summary": summary_df,
        "monthly_returns": monthly_tables,
        "annual_stats": annual_tables,
        "regime_metrics": regime_tables,
    }
