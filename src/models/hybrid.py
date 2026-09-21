from __future__ import annotations

from typing import Dict, Any
import pandas as pd

from .lstm_model import ResidualSequenceModel


def build_sarimax_lstm_predictions(
    y_train: pd.Series,
    y_test: pd.Series,
    sarimax_train_pred: pd.Series,
    sarimax_test_pred: pd.Series,
    cfg: Dict[str, Any],
) -> tuple[pd.Series, ResidualSequenceModel]:
    """Build the main Hybrid SARIMAX-LSTM model.

    Research model used in the report:
        1) SARIMAX forecasts target_log_return using exogenous X.
        2) Residual = actual_return - SARIMAX_return.
        3) LSTM learns the residual sequence.
        4) Final return = SARIMAX_return + alpha * LSTM_residual.

    Random Forest is intentionally NOT part of this hybrid forecast in this clean version.
    RF remains available only as a benchmark and feature-importance/feature-selection tool.
    """
    residual_train = (y_train - sarimax_train_pred).fillna(0.0)

    residual_model = ResidualSequenceModel(cfg).fit(residual_train)
    lstm_resid_test = residual_model.predict_one_step_rolling(
        residual_train,
        y_test,
        sarimax_test_pred,
    )

    weight_lstm = float(cfg.get("hybrid", {}).get("lstm_residual_weight", 1.0))
    sarimax_lstm = (sarimax_test_pred + weight_lstm * lstm_resid_test).rename("SARIMAX_LSTM")
    return sarimax_lstm, residual_model
