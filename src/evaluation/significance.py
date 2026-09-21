"""
Kiểm định thống kê độ chính xác dự báo.

Bao gồm:
1. Diebold-Mariano (DM) test: so sánh hai mô hình có khác nhau về độ chính xác không.
2. Harvey-Leybourne-Newbold (HLN) correction: điều chỉnh cho sample nhỏ.
3. Bảng so sánh pairwise tất cả mô hình.

Tại sao cần DM test:
    Nếu SARIMAX-LSTM đạt MAE thấp hơn Ridge 5%, câu hỏi là:
    "Sự khác biệt đó có ý nghĩa thống kê hay chỉ là may mắn trong giai đoạn test?"
    DM test trả lời câu hỏi này bằng kiểm định giả thuyết chuẩn.

Cách dùng:
    from src.evaluation.significance import dm_test_table
    table = dm_test_table(pred_df, baseline_model="Naive")
    print(table.to_string())
"""

from __future__ import annotations

from typing import Tuple, List, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats


def _dm_loss(y_true: np.ndarray, y_pred: np.ndarray, loss: str = "mse") -> np.ndarray:
    """Tính loss sequence cho từng timestep."""
    err = y_true - y_pred
    if loss == "mse":
        return err ** 2
    elif loss == "mae":
        return np.abs(err)
    else:
        raise ValueError(f"loss phải là 'mse' hoặc 'mae', nhận được '{loss}'")


def dm_test(
    y_true: pd.Series,
    y_pred_1: pd.Series,
    y_pred_2: pd.Series,
    h: int = 1,
    loss: str = "mse",
    hln_correction: bool = True,
) -> Tuple[float, float, str]:
    """
    Kiểm định Diebold-Mariano so sánh y_pred_1 vs y_pred_2.

    H0: Hai mô hình có cùng độ chính xác dự báo (E[d_t] = 0).
    H1: Mô hình 2 chính xác hơn mô hình 1 (one-sided: E[d_t] > 0).

    Args:
        y_true: giá trị thực tế.
        y_pred_1: dự báo của mô hình baseline (model 1).
        y_pred_2: dự báo của mô hình challenger (model 2, kỳ vọng tốt hơn).
        h: horizon dự báo (1 = one-step-ahead).
        loss: hàm loss, "mse" hoặc "mae".
        hln_correction: áp dụng Harvey-Leybourne-Newbold correction (khuyến nghị).

    Returns:
        (dm_statistic, p_value, interpretation)
        p_value < 0.05 → model 2 tốt hơn model 1 có ý nghĩa thống kê.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_1 = np.asarray(y_pred_1, dtype=float)
    y_pred_2 = np.asarray(y_pred_2, dtype=float)

    # Loại bỏ NaN ở bất kỳ vị trí nào
    mask = ~(np.isnan(y_true) | np.isnan(y_pred_1) | np.isnan(y_pred_2))
    y_true, y_pred_1, y_pred_2 = y_true[mask], y_pred_1[mask], y_pred_2[mask]
    T = len(y_true)

    if T < 10:
        warnings.warn("DM test cần ít nhất 10 observations. Trả về NaN.")
        return float("nan"), float("nan"), "không đủ dữ liệu"

    loss_1 = _dm_loss(y_true, y_pred_1, loss)
    loss_2 = _dm_loss(y_true, y_pred_2, loss)
    d = loss_1 - loss_2   # d > 0 → model 2 tốt hơn

    d_bar = d.mean()

    # Ước lượng phương sai của d_bar với autocorrelation
    # Dùng Newey-West truncated lag = h-1 để xử lý autocorrelation ở h-step
    autocovariances = [np.cov(d[j:], d[:-j])[0, 1] if j > 0 else np.var(d, ddof=1) for j in range(h)]
    var_d = autocovariances[0]
    for j in range(1, h):
        var_d += 2 * autocovariances[j]
    var_d = max(var_d, 1e-12)  # tránh chia cho 0

    dm_stat = d_bar / np.sqrt(var_d / T)

    if hln_correction and T >= 3:
        # Harvey, Leybourne, Newbold (1997) t-correction
        correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        dm_stat = dm_stat * correction
        # Dùng t-distribution thay vì normal
        p_val = 1 - stats.t.cdf(dm_stat, df=T - 1)
    else:
        p_val = 1 - stats.norm.cdf(dm_stat)

    # Diễn giải
    if np.isnan(p_val):
        interpretation = "N/A"
    elif p_val < 0.01:
        interpretation = "***" if d_bar > 0 else "(model 2 kém hơn ***)"
    elif p_val < 0.05:
        interpretation = "**" if d_bar > 0 else "(model 2 kém hơn **)"
    elif p_val < 0.10:
        interpretation = "*" if d_bar > 0 else "(model 2 kém hơn *)"
    else:
        interpretation = "không có ý nghĩa thống kê"

    return float(dm_stat), float(p_val), interpretation


def dm_test_table(
    pred_df: pd.DataFrame,
    baseline_model: str = "Naive",
    target: str = "close",
    loss: str = "mse",
    h: int = 1,
) -> pd.DataFrame:
    """
    Tạo bảng DM test so sánh tất cả mô hình với baseline.

    Args:
        pred_df: DataFrame từ make_prediction_frame().
        baseline_model: tên model làm baseline (mẫu số so sánh).
        target: "close" để so sánh forecast price, "return" để so sánh return.
        loss: "mse" hoặc "mae".
        h: forecast horizon.

    Returns:
        DataFrame với columns: model, dm_stat, p_value, significance, mean_loss_reduction_pct.
    """
    if target == "close":
        actual = pred_df["actual_close"]
        suffix = "_close_pred"
    else:
        actual = pred_df["actual_return"]
        suffix = "_return_pred"

    baseline_col = f"{baseline_model}{suffix}"
    if baseline_col not in pred_df.columns:
        raise KeyError(f"Baseline column '{baseline_col}' không có trong pred_df.")

    models = sorted([c.replace(suffix, "") for c in pred_df.columns if c.endswith(suffix)])
    rows = []
    for model in models:
        if model == baseline_model:
            continue
        dm_stat, p_val, sig = dm_test(
            actual,
            pred_df[baseline_col],
            pred_df[f"{model}{suffix}"],
            h=h,
            loss=loss,
        )
        baseline_loss = _dm_loss(actual.values, pred_df[baseline_col].values, loss).mean()
        model_loss = _dm_loss(actual.values, pred_df[f"{model}{suffix}"].values, loss).mean()
        reduction_pct = (baseline_loss - model_loss) / baseline_loss * 100 if baseline_loss > 0 else float("nan")
        rows.append({
            "model": model,
            "baseline": baseline_model,
            "loss_type": loss,
            "dm_statistic": round(dm_stat, 4) if not np.isnan(dm_stat) else float("nan"),
            "p_value": round(p_val, 4) if not np.isnan(p_val) else float("nan"),
            "significance": sig,
            "loss_reduction_vs_baseline_pct": round(reduction_pct, 2),
        })

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def pairwise_dm_table(
    pred_df: pd.DataFrame,
    target: str = "close",
    loss: str = "mse",
) -> pd.DataFrame:
    """
    Bảng DM test pairwise giữa tất cả cặp mô hình.
    Trả về matrix p-value: hàng = model i, cột = model j.
    p_value[i,j] < 0.05 → model j tốt hơn model i có ý nghĩa thống kê.
    """
    if target == "close":
        actual = pred_df["actual_close"]
        suffix = "_close_pred"
    else:
        actual = pred_df["actual_return"]
        suffix = "_return_pred"

    models = sorted([c.replace(suffix, "") for c in pred_df.columns if c.endswith(suffix)])
    pvals = pd.DataFrame(np.nan, index=models, columns=models)

    for m1 in models:
        for m2 in models:
            if m1 == m2:
                pvals.loc[m1, m2] = 1.0
                continue
            _, p, _ = dm_test(actual, pred_df[f"{m1}{suffix}"], pred_df[f"{m2}{suffix}"], loss=loss)
            pvals.loc[m1, m2] = round(p, 4)

    return pvals
