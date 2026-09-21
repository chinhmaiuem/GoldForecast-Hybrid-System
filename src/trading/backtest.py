from __future__ import annotations

from typing import Tuple, Dict, List
import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min() * 100)


def _sharpe(returns: pd.Series, risk_free_daily: float = 0.0) -> float:
    """Annualized Sharpe ratio using daily returns.

    risk_free_daily defaults to 0.0. This is equivalent to reporting Sharpe
    under a zero risk-free/excess-return assumption, which is common in short
    empirical backtests but should be stated in the report.
    """
    returns = returns.dropna()
    excess = returns - float(risk_free_daily)
    std = excess.std()
    if std == 0 or pd.isna(std) or len(excess) < 2:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(252))


def _safe_equity(strategy_ret: pd.Series, name: str) -> pd.Series:
    """Convert log-return strategy returns to an equity curve and strip attrs."""
    equity = np.exp(strategy_ret.fillna(0).cumsum())
    equity = pd.Series(equity.to_numpy(), index=strategy_ret.index, name=name)
    equity.attrs = {}
    return equity


def _metrics_from_signal(
    strategy_name: str,
    model: str,
    signal: pd.Series,
    actual_ret: pd.Series,
    transaction_cost: float,
    slippage: float = 0.0,
    threshold: float | None = None,
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
    allow_short: bool = False,
    min_holding_days: int | None = None,
    max_holding_days: int | None = None,
    risk_free_daily: float = 0.0,
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    signal = signal.astype(float).reindex(actual_ret.index).fillna(0.0)
    turnover = signal.diff().abs().fillna(signal.abs())
    effective_cost = float(transaction_cost) + float(slippage)
    strategy_ret = signal * actual_ret - turnover * effective_cost
    name = f"{model}_{strategy_name}" if model else strategy_name
    equity = _safe_equity(strategy_ret, name=name)

    active = signal != 0
    rows = {
        "strategy": name,
        "model": model if model else strategy_name,
        "strategy_type": strategy_name,
        "threshold": np.nan if threshold is None else threshold,
        "buy_threshold": np.nan if buy_threshold is None else buy_threshold,
        "sell_threshold": np.nan if sell_threshold is None else sell_threshold,
        "transaction_cost": float(transaction_cost),
        "slippage": float(slippage),
        "effective_transaction_cost": float(effective_cost),
        "allow_short": allow_short,
        "min_holding_days": np.nan if min_holding_days is None else int(min_holding_days),
        "max_holding_days": np.nan if max_holding_days is None else int(max_holding_days),
        "risk_free_daily": float(risk_free_daily),
        "total_return_percent": float((equity.iloc[-1] - 1) * 100),
        "sharpe": _sharpe(strategy_ret, risk_free_daily=risk_free_daily),
        "max_drawdown_percent": _max_drawdown(equity),
        "win_rate_percent": float((strategy_ret[active] > 0).mean() * 100) if active.any() else 0.0,
        "number_of_trades": int((turnover > 0).sum()),
        "exposure_percent": float(active.mean() * 100),
    }
    return equity, strategy_ret, rows


def _daily_signal(pred_ret: pd.Series, threshold: float, allow_short: bool) -> pd.Series:
    """Old/simple strategy: re-decide position every day from predicted return."""
    signal = pd.Series(0.0, index=pred_ret.index)
    signal[pred_ret > threshold] = 1.0
    if allow_short:
        signal[pred_ret < -threshold] = -1.0
    return signal


def _long_hold_until_reversal_signal(
    pred_ret: pd.Series,
    buy_threshold: float,
    sell_threshold: float,
    min_holding_days: int = 1,
    max_holding_days: int | None = None,
) -> pd.Series:
    """Stateful long-only strategy: buy, hold, exit on reversal or max holding days."""
    pred_ret = pred_ret.fillna(0.0)
    signal_values: list[float] = []
    position = 0.0
    holding_days = 0

    for value in pred_ret.to_numpy():
        if position == 0.0:
            if value > buy_threshold:
                position = 1.0
                holding_days = 1
            else:
                holding_days = 0
        else:
            holding_days += 1
            can_exit = holding_days >= max(1, int(min_holding_days))
            force_exit = max_holding_days is not None and holding_days >= int(max_holding_days)
            if can_exit and (value < sell_threshold or force_exit):
                position = 0.0
                holding_days = 0
        signal_values.append(position)
    return pd.Series(signal_values, index=pred_ret.index, dtype=float)


def _long_hold_with_trailing_stop_signal(
    pred_df: pd.DataFrame,
    pred_ret: pd.Series,
    buy_threshold: float,
    sell_threshold: float,
    trailing_stop_percent: float = 0.08,
    min_holding_days: int = 1,
    max_holding_days: int | None = None,
) -> pd.Series:
    """Long-only hold-until-reversal with trailing stop on actual close."""
    close = pred_df["actual_close"]
    pred_ret = pred_ret.fillna(0.0)
    signal_values: list[float] = []
    position = 0.0
    holding_days = 0
    peak_price = np.nan

    for dt, value in zip(pred_ret.index, pred_ret.to_numpy()):
        current_close = float(close.loc[dt])
        if position == 0.0:
            if value > buy_threshold:
                position = 1.0
                holding_days = 1
                peak_price = current_close
            else:
                holding_days = 0
                peak_price = np.nan
        else:
            holding_days += 1
            peak_price = max(float(peak_price), current_close)
            drawdown_from_peak = current_close / peak_price - 1.0 if peak_price > 0 else 0.0
            can_exit = holding_days >= max(1, int(min_holding_days))
            force_exit = max_holding_days is not None and holding_days >= int(max_holding_days)
            stop_exit = drawdown_from_peak <= -abs(float(trailing_stop_percent))
            if can_exit and (value < sell_threshold or stop_exit or force_exit):
                position = 0.0
                holding_days = 0
                peak_price = np.nan
        signal_values.append(position)
    return pd.Series(signal_values, index=pred_ret.index, dtype=float)


def _strategy_from_prediction(
    pred_df: pd.DataFrame,
    model: str,
    threshold: float,
    transaction_cost: float,
    slippage: float,
    allow_short: bool,
    strategy_mode: str = "long_hold",
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
    min_holding_days: int = 1,
    max_holding_days: int | None = None,
    trailing_stop_percent: float = 0.08,
    risk_free_daily: float = 0.0,
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    pred_ret = pred_df[f"{model}_return_pred"]
    actual_ret = pred_df["actual_return"]
    buy_threshold = threshold if buy_threshold is None else float(buy_threshold)
    sell_threshold = -threshold if sell_threshold is None else float(sell_threshold)

    mode = str(strategy_mode).lower()
    if mode in {"daily", "daily_signal", "old"}:
        signal = _daily_signal(pred_ret, threshold=threshold, allow_short=allow_short)
        strategy_name = "DailySignal"
        used_allow_short = allow_short
    elif mode in {"long_hold", "hold", "hold_until_reversal", "long_only"}:
        signal = _long_hold_until_reversal_signal(
            pred_ret,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            min_holding_days=min_holding_days,
            max_holding_days=max_holding_days,
        )
        strategy_name = "LongHold"
        used_allow_short = False
    elif mode in {"long_hold_trailing", "trailing_stop"}:
        signal = _long_hold_with_trailing_stop_signal(
            pred_df,
            pred_ret,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            trailing_stop_percent=trailing_stop_percent,
            min_holding_days=min_holding_days,
            max_holding_days=max_holding_days,
        )
        strategy_name = "LongHoldTrailing"
        used_allow_short = False
    else:
        raise ValueError(f"Unknown strategy_mode={strategy_mode!r}")

    return _metrics_from_signal(
        strategy_name=strategy_name,
        model=model,
        signal=signal,
        actual_ret=actual_ret,
        transaction_cost=transaction_cost,
        slippage=slippage,
        threshold=threshold if strategy_name == "DailySignal" else None,
        buy_threshold=buy_threshold if strategy_name != "DailySignal" else None,
        sell_threshold=sell_threshold if strategy_name != "DailySignal" else None,
        allow_short=used_allow_short,
        min_holding_days=min_holding_days if strategy_name != "DailySignal" else None,
        max_holding_days=max_holding_days if strategy_name != "DailySignal" else None,
        risk_free_daily=risk_free_daily,
    )


def _buy_hold_metrics(pred_df: pd.DataFrame, risk_free_daily: float = 0.0) -> Tuple[pd.Series, Dict[str, float]]:
    bh_ret = pred_df["actual_return"].copy()
    bh_equity = _safe_equity(bh_ret, "Buy_Hold")
    row = {
        "strategy": "Buy_Hold",
        "model": "Buy_Hold",
        "strategy_type": "Buy_Hold",
        "threshold": np.nan,
        "buy_threshold": np.nan,
        "sell_threshold": np.nan,
        "transaction_cost": 0.0,
        "slippage": 0.0,
        "effective_transaction_cost": 0.0,
        "allow_short": False,
        "min_holding_days": np.nan,
        "max_holding_days": np.nan,
        "risk_free_daily": float(risk_free_daily),
        "total_return_percent": float((bh_equity.iloc[-1] - 1) * 100),
        "sharpe": _sharpe(bh_ret, risk_free_daily=risk_free_daily),
        "max_drawdown_percent": _max_drawdown(bh_equity),
        "win_rate_percent": float((bh_ret > 0).mean() * 100),
        "number_of_trades": 1,
        "exposure_percent": 100.0,
    }
    return bh_equity, row


def run_backtest(
    pred_df: pd.DataFrame,
    threshold: float,
    transaction_cost: float,
    slippage: float,
    allow_short: bool,
    strategy_mode: str = "long_hold",
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
    min_holding_days: int = 3,
    max_holding_days: int | None = None,
    trailing_stop_percent: float = 0.08,
    include_daily_signal_metrics: bool = True,
    risk_free_daily: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_df = pred_df.copy()
    pred_df.attrs = {}
    equity_curves: List[pd.Series] = []
    rows = []

    bh_equity, bh_row = _buy_hold_metrics(pred_df, risk_free_daily=risk_free_daily)
    equity_curves.append(bh_equity)
    rows.append(bh_row)

    models = sorted([c.replace("_return_pred", "") for c in pred_df.columns if c.endswith("_return_pred")])
    for model in models:
        equity, _, metrics = _strategy_from_prediction(
            pred_df,
            model,
            threshold=threshold,
            transaction_cost=transaction_cost,
            slippage=slippage,
            allow_short=allow_short,
            strategy_mode=strategy_mode,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            min_holding_days=min_holding_days,
            max_holding_days=max_holding_days,
            trailing_stop_percent=trailing_stop_percent,
            risk_free_daily=risk_free_daily,
        )
        equity_curves.append(equity)
        rows.append(metrics)

        if include_daily_signal_metrics and str(strategy_mode).lower() not in {"daily", "daily_signal", "old"}:
            _, _, old_metrics = _strategy_from_prediction(
                pred_df,
                model,
                threshold=threshold,
                transaction_cost=transaction_cost,
                slippage=slippage,
                allow_short=allow_short,
                strategy_mode="daily_signal",
                risk_free_daily=risk_free_daily,
            )
            rows.append(old_metrics)

    clean_curves = [pd.Series(s.to_numpy(), index=s.index, name=s.name) for s in equity_curves]
    equity_df = pd.concat(clean_curves, axis=1)
    metrics_df = pd.DataFrame(rows).sort_values(["total_return_percent", "sharpe"], ascending=False).reset_index(drop=True)
    return equity_df, metrics_df


def run_threshold_sweep(
    pred_df: pd.DataFrame,
    thresholds: list[float],
    allow_short_values: list[bool],
    transaction_cost: float,
    slippage: float = 0.0,
    strategy_modes: list[str] | None = None,
    buy_thresholds: list[float] | None = None,
    sell_thresholds: list[float] | None = None,
    min_holding_days_values: list[int] | None = None,
    max_holding_days: int | None = None,
    trailing_stop_percent: float = 0.08,
    risk_free_daily: float = 0.0,
) -> pd.DataFrame:
    rows = []
    pred_df = pred_df.copy()
    pred_df.attrs = {}
    models = sorted([c.replace("_return_pred", "") for c in pred_df.columns if c.endswith("_return_pred")])
    strategy_modes = strategy_modes or ["long_hold", "daily_signal"]
    min_holding_days_values = min_holding_days_values or [1, 3, 5]

    for mode in strategy_modes:
        mode_l = str(mode).lower()
        if mode_l in {"daily", "daily_signal", "old"}:
            for allow_short in allow_short_values:
                for th in thresholds:
                    for model in models:
                        _, _, metrics = _strategy_from_prediction(
                            pred_df,
                            model,
                            threshold=th,
                            transaction_cost=transaction_cost,
                            slippage=slippage,
                            allow_short=allow_short,
                            strategy_mode="daily_signal",
                            risk_free_daily=risk_free_daily,
                        )
                        rows.append(metrics)
        else:
            buy_grid = buy_thresholds if buy_thresholds is not None else thresholds
            sell_grid = sell_thresholds if sell_thresholds is not None else [-x for x in thresholds]
            for buy_th in buy_grid:
                for sell_th in sell_grid:
                    if sell_th >= buy_th:
                        continue
                    for min_days in min_holding_days_values:
                        for model in models:
                            _, _, metrics = _strategy_from_prediction(
                                pred_df,
                                model,
                                threshold=float(buy_th),
                                transaction_cost=transaction_cost,
                                slippage=slippage,
                                allow_short=False,
                                strategy_mode=mode,
                                buy_threshold=float(buy_th),
                                sell_threshold=float(sell_th),
                                min_holding_days=int(min_days),
                                max_holding_days=max_holding_days,
                                trailing_stop_percent=trailing_stop_percent,
                                risk_free_daily=risk_free_daily,
                            )
                            rows.append(metrics)

    return pd.DataFrame(rows).sort_values(["sharpe", "total_return_percent"], ascending=False).reset_index(drop=True)


def select_best_params_by_model(
    validation_sweep: pd.DataFrame,
    primary_metric: str = "sharpe",
    secondary_metric: str = "total_return_percent",
) -> pd.DataFrame:
    """Select one strategy configuration per model using validation results only."""
    if validation_sweep.empty:
        return validation_sweep.copy()
    rows = []
    for model, group in validation_sweep.groupby("model", sort=False):
        group = group.copy()
        group = group.sort_values([primary_metric, secondary_metric], ascending=False)
        best = group.iloc[0].to_dict()
        best["selection_primary_metric"] = primary_metric
        best["selection_secondary_metric"] = secondary_metric
        best["selected_on"] = "validation"
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def run_backtest_with_selected_params(
    pred_df: pd.DataFrame,
    selected_params: pd.DataFrame,
    transaction_cost: float,
    trailing_stop_percent: float,
    slippage: float = 0.0,
    risk_free_daily: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Final test backtest using model-specific params selected on validation.

    This avoids choosing the best test-set result from a sweep. The test set is
    used only once after parameters are frozen.
    """
    pred_df = pred_df.copy()
    pred_df.attrs = {}
    equity_curves: List[pd.Series] = []
    rows = []

    bh_equity, bh_row = _buy_hold_metrics(pred_df, risk_free_daily=risk_free_daily)
    equity_curves.append(bh_equity)
    rows.append(bh_row)

    selected_by_model = {str(row["model"]): row for _, row in selected_params.iterrows()}
    models = sorted([c.replace("_return_pred", "") for c in pred_df.columns if c.endswith("_return_pred")])

    for model in models:
        row = selected_by_model.get(model)
        if row is None:
            continue
        strategy_type = str(row.get("strategy_type", "LongHoldTrailing"))
        strategy_mode = {
            "DailySignal": "daily_signal",
            "LongHold": "long_hold",
            "LongHoldTrailing": "long_hold_trailing",
        }.get(strategy_type, str(strategy_type).lower())

        threshold = 0.003 if pd.isna(row.get("threshold", np.nan)) else float(row.get("threshold"))
        buy_threshold = None if pd.isna(row.get("buy_threshold", np.nan)) else float(row.get("buy_threshold"))
        sell_threshold = None if pd.isna(row.get("sell_threshold", np.nan)) else float(row.get("sell_threshold"))
        min_days = 1 if pd.isna(row.get("min_holding_days", np.nan)) else int(row.get("min_holding_days"))
        max_days = None if pd.isna(row.get("max_holding_days", np.nan)) else int(row.get("max_holding_days"))
        allow_short = bool(row.get("allow_short", False))
        tc = transaction_cost if pd.isna(row.get("transaction_cost", np.nan)) else float(row.get("transaction_cost"))
        sp = slippage if pd.isna(row.get("slippage", np.nan)) else float(row.get("slippage", slippage))

        equity, _, metrics = _strategy_from_prediction(
            pred_df,
            model,
            threshold=threshold,
            transaction_cost=tc,
            slippage=sp,
            allow_short=allow_short,
            strategy_mode=strategy_mode,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            min_holding_days=min_days,
            max_holding_days=max_days,
            trailing_stop_percent=trailing_stop_percent,
            risk_free_daily=risk_free_daily,
        )
        metrics["selected_on"] = "validation"
        for k in ["total_return_percent", "sharpe", "max_drawdown_percent", "number_of_trades", "exposure_percent"]:
            metrics[f"validation_{k}"] = row.get(k, np.nan)
        equity_curves.append(equity)
        rows.append(metrics)

    clean_curves = [pd.Series(s.to_numpy(), index=s.index, name=s.name) for s in equity_curves]
    equity_df = pd.concat(clean_curves, axis=1)
    metrics_df = pd.DataFrame(rows).sort_values(["total_return_percent", "sharpe"], ascending=False).reset_index(drop=True)
    return equity_df, metrics_df
