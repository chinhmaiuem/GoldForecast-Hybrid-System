from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, List
import itertools
import warnings
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


def _fit_aic(
    y: pd.Series,
    exog: Optional[pd.DataFrame],
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    maxiter: int,
) -> float:
    """Return AIC for one SARIMAX order, or inf when the fit fails."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False, maxiter=maxiter)
        return float(res.aic)
    except Exception:
        return float("inf")


def _auto_select_order(
    y_train: pd.Series,
    exog_train: Optional[pd.DataFrame],
    cfg: Dict[str, Any],
    seasonal_order: Tuple[int, int, int, int],
) -> tuple[Tuple[int, int, int], pd.DataFrame]:
    """Small AIC/BIC grid search for SARIMAX order.

    It is disabled by default because it can slow the pipeline. When enabled,
    the search is performed on the most recent subset of train data so the main
    project remains practical on a laptop.
    """
    mcfg = cfg["models"]["sarimax"]
    p_range: List[int] = list(mcfg.get("auto_p_range", [0, 1, 2]))
    d_range: List[int] = list(mcfg.get("auto_d_range", [0, 1]))
    q_range: List[int] = list(mcfg.get("auto_q_range", [0, 1, 2]))
    max_candidates = int(mcfg.get("auto_max_candidates", 30))
    maxiter = int(mcfg.get("auto_maxiter", min(80, int(mcfg.get("maxiter", 200)))))
    subset_rows = int(mcfg.get("auto_subset_rows", 600))

    y_sub = y_train.dropna().astype(float)
    if len(y_sub) > subset_rows:
        y_sub = y_sub.iloc[-subset_rows:]
    exog_sub = exog_train.reindex(y_sub.index) if exog_train is not None else None

    candidates = list(itertools.product(p_range, d_range, q_range))
    if len(candidates) > max_candidates:
        priority = [(1, 0, 1), (1, 1, 1), (0, 1, 1), (1, 0, 0), (2, 0, 1), (1, 1, 0)]
        candidates = priority + [c for c in candidates if c not in priority]
        candidates = candidates[:max_candidates]

    rows = []
    best_order = tuple(mcfg.get("order", [1, 0, 1]))
    best_aic = float("inf")
    for order in candidates:
        aic = _fit_aic(y_sub, exog_sub, tuple(order), seasonal_order, maxiter=maxiter)
        rows.append({"order": str(tuple(order)), "AIC": aic})
        if aic < best_aic:
            best_aic = aic
            best_order = tuple(order)

    search_df = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    return best_order, search_df


def fit_sarimax(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.Series, pd.Series, object, pd.DataFrame]:
    mcfg = cfg["models"]["sarimax"]
    order = tuple(mcfg.get("order", [1, 0, 1]))
    seasonal_order = tuple(mcfg.get("seasonal_order", [0, 0, 0, 0]))
    maxiter = int(mcfg.get("maxiter", 200))
    use_exog = bool(mcfg.get("use_exog", True))
    auto_order = bool(mcfg.get("auto_order", False))

    exog_train = X_train if use_exog and X_train.shape[1] > 0 else None
    exog_test = X_test if use_exog and X_test.shape[1] > 0 else None

    diagnostics = []
    auto_search_df = pd.DataFrame()
    if auto_order:
        try:
            order, auto_search_df = _auto_select_order(y_train, exog_train, cfg, seasonal_order)
            diagnostics.append({"test": "SARIMAX_auto_order_enabled", "statistic": True, "p_value": None})
            diagnostics.append({"test": "SARIMAX_auto_order_selected", "statistic": str(order), "p_value": None})
            if not auto_search_df.empty:
                diagnostics.append({"test": "SARIMAX_auto_order_best_AIC", "statistic": float(auto_search_df.iloc[0]["AIC"]), "p_value": None})
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Auto order selection failed; using configured order {order}. Error: {exc}")
            diagnostics.append({"test": "SARIMAX_auto_order_warning", "statistic": str(exc), "p_value": None})
    else:
        diagnostics.append({"test": "SARIMAX_auto_order_enabled", "statistic": False, "p_value": None})

    used_exog = exog_train is not None
    sarimax_warning = ""

    try:
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=maxiter)
    except Exception as exc:  # noqa: BLE001
        sarimax_warning = (
            "SARIMAX with exogenous variables failed. "
            "The model was refitted without exogenous variables. "
            f"Original error: {exc}"
        )
        warnings.warn(sarimax_warning)
        model = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=maxiter)
        exog_test = None
        used_exog = False

    train_pred = pd.Series(res.fittedvalues, index=y_train.index, name="SARIMAX_train")
    forecast = res.get_forecast(steps=len(X_test), exog=exog_test).predicted_mean
    test_pred = pd.Series(forecast.values, index=X_test.index, name="SARIMAX")

    diagnostics.extend([
        {"test": "SARIMAX_order", "statistic": str(order), "p_value": None},
        {"test": "SARIMAX_seasonal_order", "statistic": str(seasonal_order), "p_value": None},
        {"test": "SARIMAX_exog_used", "statistic": bool(used_exog), "p_value": None},
    ])
    if sarimax_warning:
        diagnostics.append({"test": "SARIMAX_warning", "statistic": sarimax_warning, "p_value": None})
    try:
        diagnostics.append({"test": "SARIMAX_AIC", "statistic": float(res.aic), "p_value": None})
        diagnostics.append({"test": "SARIMAX_BIC", "statistic": float(res.bic), "p_value": None})
    except Exception:
        pass
    try:
        adf_stat, adf_p, *_ = adfuller(y_train.dropna())
        diagnostics.append({"test": "ADF_target_train", "statistic": adf_stat, "p_value": adf_p})
    except Exception as exc:  # noqa: BLE001
        diagnostics.append({"test": "ADF_target_train", "statistic": None, "p_value": None, "error": str(exc)})
    try:
        lb = acorr_ljungbox(res.resid.dropna(), lags=[10, 20], return_df=True)
        for lag, row in lb.iterrows():
            diagnostics.append({"test": f"LjungBox_residual_lag_{lag}", "statistic": row["lb_stat"], "p_value": row["lb_pvalue"]})
    except Exception as exc:  # noqa: BLE001
        diagnostics.append({"test": "LjungBox_residual", "statistic": None, "p_value": None, "error": str(exc)})

    diagnostics_df = pd.DataFrame(diagnostics)
    return train_pred, test_pred, res, diagnostics_df
