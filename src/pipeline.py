from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
import pandas as pd

from src.utils.config import resolve_path, ensure_parent
from src.utils.io import save_csv, read_csv_date_index
from src.utils.model_io import ensure_dir, save_pickle, save_joblib, save_keras_model_if_possible
from src.utils.plotting import plot_forecasts, plot_equity, plot_feature_importance, plot_pipeline
from src.utils.experiment_log import ExperimentLogger
from src.data.downloader import download_market_data, save_raw_data
from src.data.cleaning import clean_and_align
from src.data.features import build_features
from src.data.split import split_train_validation_test, _ensure_business_day_index
from src.data.filtering import FeatureFilter, build_quality_report
from src.models.baselines import naive_return_predictions, train_ridge_predict, train_rf_predict
from src.models.sarimax_model import fit_sarimax
from src.models.lstm_model import DirectLSTMModel
from src.models.hybrid import build_sarimax_lstm_predictions
from src.evaluation.metrics import make_prediction_frame, forecast_metrics, return_metrics
from src.evaluation.significance import dm_test_table
from src.evaluation.walk_forward import WalkForwardValidator
from src.trading.metrics_extended import extended_backtest_metrics
from src.trading.backtest import (
    run_threshold_sweep,
    select_best_params_by_model,
    run_backtest_with_selected_params,
)

META_COLS = ["actual_close", "close_lag_1", "target_log_return"]


def step_download(cfg: Dict[str, Any]) -> pd.DataFrame:
    print("\n========== STEP 1/7: DOWNLOAD RAW DATA ==========")
    df = download_market_data(cfg)
    raw_path = resolve_path(cfg, cfg["outputs"]["raw_data"])
    save_raw_data(df, raw_path)
    return df


def step_prepare(cfg: Dict[str, Any], raw_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n========== STEP 2/7: CLEAN + FEATURE ENGINEERING ==========")
    if raw_df is None:
        raw_path = resolve_path(cfg, cfg["outputs"]["raw_data"])
        raw_df = read_csv_date_index(raw_path)
    clean_df, raw_report = clean_and_align(raw_df, cfg)
    dataset, price_features, macro_features = build_features(clean_df, cfg)
    processed_path = resolve_path(cfg, cfg["outputs"]["processed_data"])
    save_csv(dataset, processed_path, index=True)
    print(f"[prepare] processed dataset -> {processed_path}")
    print(f"[prepare] rows={len(dataset)}, price_features={len(price_features)}, macro_features={len(macro_features)}")
    return dataset, raw_report


def _fit_predict_stage(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    cfg: Dict[str, Any],
    stage_name: str,
    save_models: bool = False,
) -> tuple[pd.DataFrame, FeatureFilter, dict[str, Any]]:
    """Fit all forecasting models on train and predict eval_df.

    Used for validation forecasting and final test forecasting. RandomForest is
    a benchmark/feature-importance model only, not part of the main hybrid.
    """
    print(f"\n========== FIT/PREDICT STAGE: {stage_name.upper()} ==========")
    feature_cols = [c for c in train.columns if c not in META_COLS]
    X_train_raw = train[feature_cols]
    y_train = train["target_log_return"]
    X_eval_raw = eval_df[feature_cols]
    y_eval = eval_df["target_log_return"]

    ff = FeatureFilter(cfg)
    X_train = ff.fit_transform(X_train_raw, y_train)
    X_eval = ff.transform(X_eval_raw)

    print(f"[{stage_name}] train rows={len(train)}, eval rows={len(eval_df)}")
    print(f"[{stage_name}] selected features={len(ff.selected_features_)}")

    return_preds: dict[str, pd.Series] = {}
    models_dir = ensure_dir(resolve_path(cfg, cfg.get("outputs", {}).get("models_dir", "artifacts/models")))

    return_preds["Naive"] = naive_return_predictions(eval_df.index)

    ridge_pred, ridge_model = train_ridge_predict(X_train, y_train, X_eval, cfg)
    return_preds["Ridge"] = ridge_pred

    rf_pred, rf_model = train_rf_predict(X_train, y_train, X_eval, cfg)
    return_preds["RandomForest"] = rf_pred

    sarimax_train, sarimax_eval, sarimax_result, diagnostics = fit_sarimax(y_train, X_train, X_eval, cfg)
    return_preds["SARIMAX"] = sarimax_eval

    direct_lstm = DirectLSTMModel(cfg)
    return_preds["LSTM"] = direct_lstm.fit_predict(X_train, y_train, X_eval)

    sarimax_lstm, residual_lstm = build_sarimax_lstm_predictions(
        y_train=y_train,
        y_test=y_eval,
        sarimax_train_pred=sarimax_train,
        sarimax_test_pred=sarimax_eval,
        cfg=cfg,
    )
    return_preds["SARIMAX_LSTM"] = sarimax_lstm

    pred_df = make_prediction_frame(eval_df, return_preds)

    info: dict[str, Any] = {
        "stage_name": stage_name,
        "selected_features": list(ff.selected_features_),
        "diagnostics": diagnostics,
        "direct_lstm_backend": direct_lstm.backend_,
        "residual_lstm_backend": residual_lstm.backend_,
    }

    if save_models:
        save_csv(ff.feature_importance_, resolve_path(cfg, cfg["outputs"]["feature_importance"]), index=False)
        selected_path = resolve_path(cfg, cfg["outputs"]["selected_features"])
        ensure_parent(selected_path)
        selected_path.write_text("\n".join(ff.selected_features_), encoding="utf-8")
        save_csv(diagnostics, resolve_path(cfg, cfg["outputs"]["diagnostics"]), index=False)

        save_joblib(ridge_model, models_dir / "ridge_benchmark.pkl")
        save_joblib(rf_model, models_dir / "random_forest_benchmark.pkl")
        save_pickle(sarimax_result, models_dir / "sarimax_result.pkl")
        if not save_keras_model_if_possible(direct_lstm.model_, models_dir / "direct_lstm_model.keras"):
            save_pickle(direct_lstm.model_, models_dir / "direct_lstm_or_fallback_model.pkl")
        if not save_keras_model_if_possible(residual_lstm.model_, models_dir / "residual_lstm_model.keras"):
            save_pickle(residual_lstm.model_, models_dir / "residual_lstm_or_fallback_model.pkl")

        backend_text = (
            f"Direct LSTM backend: {direct_lstm.backend_}\n"
            f"Residual LSTM backend: {residual_lstm.backend_}\n"
            f"Hybrid alpha: {cfg.get('hybrid', {}).get('lstm_residual_weight', 1.0)}\n"
            "If backend starts with 'fallback', install TensorFlow in a compatible Python environment to use the Keras LSTM backend.\n"
        )
        backend_path = resolve_path(cfg, cfg["outputs"]["lstm_backend"])
        ensure_parent(backend_path)
        backend_path.write_text(backend_text, encoding="utf-8")

        model_manifest = {
            "project": cfg.get("project", {}).get("name", "GoldForecast-Hybrid System"),
            "main_model": "SARIMAX-LSTM",
            "target": "target_log_return",
            "hybrid_formula": "final_return = SARIMAX_return + alpha * LSTM_residual",
            "random_forest_role": "benchmark_and_feature_importance_only_not_part_of_hybrid",
            "selected_features": list(ff.selected_features_),
            "sarimax_order": cfg.get("models", {}).get("sarimax", {}).get("order"),
            "sarimax_auto_order": cfg.get("models", {}).get("sarimax", {}).get("auto_order", False),
            "sarimax_seasonal_order": cfg.get("models", {}).get("sarimax", {}).get("seasonal_order"),
            "direct_lstm_backend": direct_lstm.backend_,
            "residual_lstm_backend": residual_lstm.backend_,
            "methodology": "thresholds_selected_on_validation_then_evaluated_once_on_test",
            "saved_files": sorted([x.name for x in models_dir.glob("*")]),
        }
        (models_dir / "model_manifest.json").write_text(json.dumps(model_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (models_dir / "README_models.txt").write_text(
            "This folder stores trained model artifacts generated by the final test stage.\n"
            "Main research model: SARIMAX-LSTM.\n"
            "Ridge and RandomForest are benchmarks only and are not components of the hybrid model.\n"
            "Trading thresholds are selected on validation, then frozen for final test evaluation.\n",
            encoding="utf-8",
        )
    return pred_df, ff, info


def _optimize_hybrid_alpha_on_validation(val_pred_df: pd.DataFrame, cfg: Dict[str, Any]) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Select alpha for SARIMAX + alpha*LSTM residual on validation only."""
    hcfg = cfg.get("hybrid", {})
    base_alpha = float(hcfg.get("lstm_residual_weight", 1.0))
    if not bool(hcfg.get("optimize_alpha", False)):
        return base_alpha, val_pred_df, pd.DataFrame([{"alpha": base_alpha, "selected": True, "note": "optimize_alpha_disabled"}])

    if "SARIMAX_return_pred" not in val_pred_df or "SARIMAX_LSTM_return_pred" not in val_pred_df:
        return base_alpha, val_pred_df, pd.DataFrame([{"alpha": base_alpha, "selected": True, "note": "missing_columns"}])

    candidates = [float(a) for a in hcfg.get("alpha_candidates", [0.0, 0.25, 0.5, 0.75, 1.0])]
    metric = str(hcfg.get("alpha_metric", "mae")).lower()
    sarimax = val_pred_df["SARIMAX_return_pred"].astype(float)
    # val_pred was built with base_alpha; recover the residual contribution.
    denom = base_alpha if abs(base_alpha) > 1e-12 else 1.0
    residual_component = (val_pred_df["SARIMAX_LSTM_return_pred"].astype(float) - sarimax) / denom
    actual = val_pred_df["actual_return"].astype(float)

    rows = []
    for alpha in candidates:
        pred = sarimax + alpha * residual_component
        err = actual - pred
        mae = float(np.abs(err).mean())
        mse = float((err ** 2).mean())
        rows.append({"alpha": alpha, "validation_mae": mae, "validation_mse": mse})
    alpha_df = pd.DataFrame(rows)
    score_col = "validation_mse" if metric == "mse" else "validation_mae"
    best_alpha = float(alpha_df.sort_values(score_col).iloc[0]["alpha"])
    alpha_df["selected"] = alpha_df["alpha"] == best_alpha

    # Patch validation prediction frame for threshold tuning using the selected alpha.
    patched = val_pred_df.copy()
    patched["SARIMAX_LSTM_return_pred"] = sarimax + best_alpha * residual_component
    patched["SARIMAX_LSTM_close_pred"] = patched["close_lag_1"] * np.exp(patched["SARIMAX_LSTM_return_pred"])
    print(f"[hybrid] selected alpha from VALIDATION: {best_alpha} ({score_col})")
    return best_alpha, patched, alpha_df


def _write_methodology_notes(cfg: Dict[str, Any]) -> None:
    text = """
Methodology notes
=================

1. Train / validation / test split
- Train is used to fit forecasting models during validation.
- Validation is used to select trading thresholds, strategy parameters, and optionally the SARIMAX-LSTM residual weight alpha.
- Test is used only once after parameters are frozen.
This reduces in-sample selection bias / backtest overfitting caused by sweeping thresholds on the test set.

2. Macro data alignment
CPI and Fed Rate are low-frequency macro variables aligned to daily data by limited forward-fill.
This is only an approximation and may create temporal inconsistency because one macro value can be repeated across many daily rows.
For production-quality forecasting, macro variables should be aligned by official FRED release dates so that only information available at the forecast date is used.

3. SARIMAX order and seasonality
The default SARIMAX order is chosen empirically from the config. Optional AIC grid search can be enabled with models.sarimax.auto_order=true.
The default seasonal_order uses a weekly trading-cycle approximation, but stronger seasonal experiments such as [1,0,1,252] should be treated as optional because they can be much slower and may be unstable on limited data.

4. Directional accuracy
forecast_metrics reports price_directional_accuracy_percent.
return_metrics reports return_directional_accuracy_percent.
They are separated to avoid confusion.

5. Sharpe ratio
Sharpe is annualized with sqrt(252) and uses risk_free_daily from config, default 0.0.
Thus the default is a zero-risk-free/excess-return assumption.

6. Walk-forward validation
The main result still uses train/validation/final-test to avoid test-set parameter selection.
A walk-forward robustness check can be enabled in config. It is not used for threshold selection; it only checks whether results are stable across multiple expanding-window folds.

7. Trading cost realism
The backtest applies transaction_cost plus slippage as the effective per-turnover cost. The default values are still assumptions and should be tested under cost sensitivity scenarios.

8. Statistical testing and extended trading metrics
The system exports Diebold-Mariano tests versus the Naive baseline and extended trading metrics such as Sortino, Calmar, Omega, annual/monthly returns, and regime-conditioned metrics.
""".strip()
    out = resolve_path(cfg, cfg["outputs"].get("methodology_notes", "artifacts/metrics/methodology_notes.txt"))
    ensure_parent(out)
    out.write_text(text, encoding="utf-8")


def _save_extended_metrics(cfg: Dict[str, Any], equity_df: pd.DataFrame, pred_df: pd.DataFrame, risk_free_daily: float) -> None:
    try:
        ext = extended_backtest_metrics(equity_df, pred_df, risk_free_daily=risk_free_daily)
        save_csv(ext["summary"], resolve_path(cfg, cfg["outputs"].get("extended_trading_metrics", "artifacts/metrics/extended_trading_metrics.csv")), index=False)

        # Flatten monthly / annual / regime tables for easy CSV inspection.
        monthly_frames = []
        for strategy, table in ext.get("monthly_returns", {}).items():
            monthly_frames.append(table.reset_index().assign(strategy=strategy))
        if monthly_frames:
            save_csv(pd.concat(monthly_frames, ignore_index=True), resolve_path(cfg, cfg["outputs"].get("monthly_returns", "artifacts/metrics/monthly_returns.csv")), index=False)

        annual_frames = []
        for strategy, table in ext.get("annual_stats", {}).items():
            annual_frames.append(table.assign(strategy=strategy))
        if annual_frames:
            save_csv(pd.concat(annual_frames, ignore_index=True), resolve_path(cfg, cfg["outputs"].get("annual_stats", "artifacts/metrics/annual_stats.csv")), index=False)

        regime_frames = []
        for strategy, table in ext.get("regime_metrics", {}).items():
            regime_frames.append(table.assign(strategy=strategy))
        if regime_frames:
            save_csv(pd.concat(regime_frames, ignore_index=True), resolve_path(cfg, cfg["outputs"].get("regime_metrics", "artifacts/metrics/regime_metrics.csv")), index=False)
    except Exception as exc:  # noqa: BLE001
        warn_path = resolve_path(cfg, "artifacts/metrics/extended_metrics_warning.txt")
        ensure_parent(warn_path)
        warn_path.write_text(str(exc), encoding="utf-8")


def _save_dm_tests(cfg: Dict[str, Any], pred_df: pd.DataFrame) -> None:
    try:
        dm_close = dm_test_table(pred_df, baseline_model="Naive", target="close", loss="mse")
        dm_return = dm_test_table(pred_df, baseline_model="Naive", target="return", loss="mse")
        dm = pd.concat([dm_close.assign(target="close"), dm_return.assign(target="return")], ignore_index=True)
        save_csv(dm, resolve_path(cfg, cfg["outputs"].get("dm_test_results", "artifacts/metrics/dm_test_results.csv")), index=False)
    except Exception as exc:  # noqa: BLE001
        warn_path = resolve_path(cfg, "artifacts/metrics/dm_test_warning.txt")
        ensure_parent(warn_path)
        warn_path.write_text(str(exc), encoding="utf-8")


def _run_walk_forward_robustness_check(cfg: Dict[str, Any], dataset: pd.DataFrame) -> None:
    """Optional expanding-window robustness check.

    This is not used for parameter selection. It runs with the configured/default
    trading parameters to show whether model performance is stable across folds.
    Keep n_folds small (3--5) because each fold refits SARIMAX/LSTM models.
    """
    wf = WalkForwardValidator(cfg)
    if not wf.enabled:
        return

    trading_cfg = cfg.get("trading", {})
    risk_free_daily = float(trading_cfg.get("risk_free_daily", 0.0))

    def _fold_eval(fold_cfg: Dict[str, Any], train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        pred_df, _, _ = _fit_predict_stage(train_df, test_df, fold_cfg, stage_name="walk_forward_fold", save_models=False)
        fm = forecast_metrics(pred_df)
        rm = return_metrics(pred_df)
        from src.trading.backtest import run_backtest
        equity_df, tm = run_backtest(
            pred_df,
            threshold=float(trading_cfg.get("threshold", 0.003)),
            transaction_cost=float(trading_cfg.get("transaction_cost", 0.0005)),
            slippage=float(trading_cfg.get("slippage", 0.0)),
            allow_short=bool(trading_cfg.get("allow_short", False)),
            strategy_mode=str(trading_cfg.get("strategy_mode", "long_hold_trailing")),
            buy_threshold=float(trading_cfg.get("buy_threshold", trading_cfg.get("threshold", 0.003))),
            sell_threshold=float(trading_cfg.get("sell_threshold", -trading_cfg.get("threshold", 0.003))),
            min_holding_days=int(trading_cfg.get("min_holding_days", 5)),
            max_holding_days=trading_cfg.get("max_holding_days", None),
            trailing_stop_percent=float(trading_cfg.get("trailing_stop_percent", 0.08)),
            include_daily_signal_metrics=False,
            risk_free_daily=risk_free_daily,
        )
        return {"forecast_metrics": fm, "return_metrics": rm, "trading_metrics": tm, "pred_df": pred_df, "equity_df": equity_df}

    out_dir = resolve_path(cfg, cfg.get("outputs", {}).get("walk_forward_dir", "artifacts/metrics/walk_forward"))
    try:
        summary = wf.run(dataset, _fold_eval)
        if summary.folds:
            wf.save_summary(summary, out_dir)
        else:
            ensure_parent(out_dir / "walk_forward_warning.txt")
            (out_dir / "walk_forward_warning.txt").write_text("No successful walk-forward folds.", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        ensure_parent(out_dir / "walk_forward_warning.txt")
        (out_dir / "walk_forward_warning.txt").write_text(str(exc), encoding="utf-8")


def step_train_evaluate(cfg: Dict[str, Any], dataset: pd.DataFrame | None = None, raw_report: pd.DataFrame | None = None) -> pd.DataFrame:
    logger = ExperimentLogger(resolve_path(cfg, cfg["outputs"].get("experiment_log", "artifacts/metrics/experiment_log.csv")))
    logger.start()

    print("\n========== STEP 3/7: TRAIN / VALIDATION / TEST SPLIT ==========")
    if dataset is None:
        dataset = read_csv_date_index(resolve_path(cfg, cfg["outputs"]["processed_data"]))

    train, validation, test = split_train_validation_test(dataset, cfg)
    final_train = pd.concat([train, validation], axis=0).sort_index()
    final_train = _ensure_business_day_index(final_train)
    print(f"[split] train      : {train.index.min().date()} -> {train.index.max().date()} ({len(train)} rows)")
    print(f"[split] validation : {validation.index.min().date()} -> {validation.index.max().date()} ({len(validation)} rows)")
    print(f"[split] test       : {test.index.min().date()} -> {test.index.max().date()} ({len(test)} rows)")

    print("\n========== STEP 4/7: VALIDATION FORECASTS + PARAMETER SELECTION ==========")
    val_pred_df, _, _ = _fit_predict_stage(train, validation, cfg, stage_name="validation", save_models=False)
    best_alpha, val_pred_df, alpha_df = _optimize_hybrid_alpha_on_validation(val_pred_df, cfg)
    save_csv(alpha_df, resolve_path(cfg, cfg["outputs"].get("selected_alpha", "artifacts/metrics/selected_alpha.csv")), index=False)
    save_csv(val_pred_df, resolve_path(cfg, cfg["outputs"].get("validation_predictions", "data/predictions/validation_predictions.csv")), index=True)

    # Freeze the validation-selected alpha for the final test stage.
    final_cfg = dict(cfg)
    final_cfg["hybrid"] = {**cfg.get("hybrid", {}), "lstm_residual_weight": float(best_alpha)}

    trading_cfg = cfg["trading"]
    risk_free_daily = float(trading_cfg.get("risk_free_daily", 0.0))
    validation_sweep = run_threshold_sweep(
        val_pred_df,
        thresholds=[float(x) for x in trading_cfg.get("threshold_sweep", [0.003])],
        allow_short_values=[bool(x) for x in trading_cfg.get("allow_short_sweep", [True, False])],
        transaction_cost=float(trading_cfg.get("transaction_cost", 0.0005)),
        slippage=float(trading_cfg.get("slippage", 0.0)),
        strategy_modes=[str(x) for x in trading_cfg.get("strategy_modes_sweep", ["long_hold", "daily_signal"])],
        buy_thresholds=[float(x) for x in trading_cfg.get("buy_threshold_sweep", trading_cfg.get("threshold_sweep", [0.003]))],
        sell_thresholds=[float(x) for x in trading_cfg.get("sell_threshold_sweep", [-0.001, -0.002, -0.003])],
        min_holding_days_values=[int(x) for x in trading_cfg.get("min_holding_days_sweep", [1, 3, 5, 10])],
        max_holding_days=trading_cfg.get("max_holding_days", None),
        trailing_stop_percent=float(trading_cfg.get("trailing_stop_percent", 0.08)),
        risk_free_daily=risk_free_daily,
    )
    save_csv(validation_sweep, resolve_path(cfg, cfg["outputs"].get("validation_sweep_metrics", "artifacts/metrics/validation_sweep_metrics.csv")), index=False)
    save_csv(validation_sweep, resolve_path(cfg, cfg["outputs"].get("trading_sweep_metrics", "artifacts/metrics/trading_sweep_metrics.csv")), index=False)

    selected_params = select_best_params_by_model(
        validation_sweep,
        primary_metric=str(trading_cfg.get("selection_metric", "sharpe")),
        secondary_metric=str(trading_cfg.get("selection_secondary_metric", "total_return_percent")),
    )
    save_csv(selected_params, resolve_path(cfg, cfg["outputs"].get("selected_trading_params", "artifacts/metrics/selected_trading_params.csv")), index=False)
    print("\nSelected trading params from VALIDATION")
    cols_to_print = [c for c in ["model", "strategy_type", "buy_threshold", "sell_threshold", "min_holding_days", "total_return_percent", "sharpe"] if c in selected_params.columns]
    print(selected_params[cols_to_print].to_string(index=False))

    print("\n========== STEP 5/7: FINAL TEST FORECASTS ==========")
    test_pred_df, final_filter, _ = _fit_predict_stage(final_train, test, final_cfg, stage_name="final_test", save_models=True)
    pred_path = resolve_path(cfg, cfg["outputs"]["predictions"])
    save_csv(test_pred_df, pred_path, index=True)
    save_csv(test_pred_df, resolve_path(cfg, cfg["outputs"].get("test_predictions", "data/predictions/test_predictions.csv")), index=True)

    fm = forecast_metrics(test_pred_df)
    rm = return_metrics(test_pred_df)
    save_csv(fm, resolve_path(cfg, cfg["outputs"]["forecast_metrics"]), index=False)
    save_csv(rm, resolve_path(cfg, cfg["outputs"]["return_metrics"]), index=False)
    _save_dm_tests(cfg, test_pred_df)

    print("\nFinal TEST close-price forecast metrics")
    print(fm.to_string(index=False))
    print("\nFinal TEST target return metrics")
    print(rm.to_string(index=False))

    print("\n========== STEP 6/7: FINAL TEST BACKTEST WITH FROZEN VALIDATION PARAMS ==========")
    equity_df, tm = run_backtest_with_selected_params(
        test_pred_df,
        selected_params=selected_params,
        transaction_cost=float(trading_cfg.get("transaction_cost", 0.0005)),
        trailing_stop_percent=float(trading_cfg.get("trailing_stop_percent", 0.08)),
        slippage=float(trading_cfg.get("slippage", 0.0)),
        risk_free_daily=risk_free_daily,
    )
    save_csv(tm, resolve_path(cfg, cfg["outputs"].get("final_test_trading_metrics", "artifacts/metrics/final_test_trading_metrics.csv")), index=False)
    save_csv(tm, resolve_path(cfg, cfg["outputs"]["trading_metrics"]), index=False)
    _save_extended_metrics(cfg, equity_df, test_pred_df, risk_free_daily=risk_free_daily)

    print("\nFinal TEST trading metrics")
    print(tm.to_string(index=False))

    print("\n========== STEP 7/7: REPORTS + PLOTS ==========")
    quality = build_quality_report(raw_report if raw_report is not None else pd.DataFrame(), dataset, final_train, test, final_filter, validation=validation)
    save_csv(quality, resolve_path(cfg, cfg["outputs"]["data_quality_report"]), index=False)
    _write_methodology_notes(cfg)

    plot_forecasts(
        test_pred_df,
        [c for c in test_pred_df.columns if c.endswith("_close_pred")],
        resolve_path(cfg, cfg["outputs"]["forecast_plot"]),
    )
    plot_equity(equity_df, resolve_path(cfg, cfg["outputs"]["equity_plot"]))
    plot_feature_importance(final_filter.feature_importance_, resolve_path(cfg, cfg["outputs"]["feature_importance_plot"]))
    plot_pipeline(resolve_path(cfg, cfg["outputs"]["pipeline_plot"]))
    _run_walk_forward_robustness_check(cfg, dataset)

    try:
        logger.log(
            final_cfg,
            forecast_metrics=fm,
            trading_metrics=tm,
            extra={"selected_alpha": best_alpha, "risk_free_daily": risk_free_daily},
        )
    except Exception as exc:  # noqa: BLE001
        warn_path = resolve_path(cfg, "artifacts/metrics/experiment_log_warning.txt")
        ensure_parent(warn_path)
        warn_path.write_text(str(exc), encoding="utf-8")

    models_dir = resolve_path(cfg, cfg.get("outputs", {}).get("models_dir", "artifacts/models"))
    print(f"\n[done] validation sweep -> {resolve_path(cfg, cfg['outputs'].get('validation_sweep_metrics', 'artifacts/metrics/validation_sweep_metrics.csv'))}")
    print(f"[done] selected params -> {resolve_path(cfg, cfg['outputs'].get('selected_trading_params', 'artifacts/metrics/selected_trading_params.csv'))}")
    print(f"[done] final test trading metrics -> {resolve_path(cfg, cfg['outputs'].get('final_test_trading_metrics', 'artifacts/metrics/final_test_trading_metrics.csv'))}")
    print(f"[done] trained models -> {models_dir}")
    return test_pred_df


def run_all(cfg: Dict[str, Any]) -> pd.DataFrame:
    raw = step_download(cfg)
    dataset, raw_report = step_prepare(cfg, raw)
    return step_train_evaluate(cfg, dataset, raw_report)
