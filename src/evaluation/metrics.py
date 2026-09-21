from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = y_true.replace(0, np.nan).abs()
    return float((np.abs((y_true - y_pred) / denom)).dropna().mean() * 100)


def make_prediction_frame(test_df: pd.DataFrame, return_predictions: Dict[str, pd.Series]) -> pd.DataFrame:
    out = pd.DataFrame(index=test_df.index)
    out["actual_close"] = test_df["actual_close"]
    out["close_lag_1"] = test_df["close_lag_1"]
    out["actual_return"] = test_df["target_log_return"]
    for model, pred_ret in return_predictions.items():
        pred_ret = pred_ret.reindex(out.index).astype(float)
        out[f"{model}_return_pred"] = pred_ret
        out[f"{model}_close_pred"] = out["close_lag_1"] * np.exp(pred_ret)
    return out.dropna()


def forecast_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Close-price metrics.

    price_directional_accuracy compares whether predicted and actual close move
    in the same direction relative to close_lag_1.
    """
    rows = []
    model_names = sorted([c.replace("_close_pred", "") for c in pred_df.columns if c.endswith("_close_pred")])
    actual = pred_df["actual_close"]
    actual_price_dir = np.sign(pred_df["actual_close"] - pred_df["close_lag_1"])
    for model in model_names:
        pred = pred_df[f"{model}_close_pred"]
        pred_price_dir = np.sign(pred - pred_df["close_lag_1"])
        rows.append({
            "model": model,
            "MAE": mean_absolute_error(actual, pred),
            "RMSE": mean_squared_error(actual, pred) ** 0.5,
            "MAPE_percent": _mape(actual, pred),
            "price_directional_accuracy_percent": float((actual_price_dir == pred_price_dir).mean() * 100),
            "R2": r2_score(actual, pred),
            "n_obs": len(pred_df),
        })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def return_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Target-return metrics.

    return_directional_accuracy compares the signs of actual and predicted
    target_log_return directly. This is reported separately from price DA to
    avoid ambiguity.
    """
    rows = []
    model_names = sorted([c.replace("_return_pred", "") for c in pred_df.columns if c.endswith("_return_pred")])
    actual = pred_df["actual_return"]
    actual_return_dir = np.sign(actual)
    for model in model_names:
        pred = pred_df[f"{model}_return_pred"]
        pred_return_dir = np.sign(pred)
        rows.append({
            "model": model,
            "target_MAE": mean_absolute_error(actual, pred),
            "target_RMSE": mean_squared_error(actual, pred) ** 0.5,
            "return_directional_accuracy_percent": float((actual_return_dir == pred_return_dir).mean() * 100),
            "target_R2": r2_score(actual, pred),
            "n_obs": len(pred_df),
        })
    return pd.DataFrame(rows).sort_values("target_MAE").reset_index(drop=True)
