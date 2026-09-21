from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


def naive_return_predictions(index: pd.Index) -> pd.Series:
    return pd.Series(np.zeros(len(index)), index=index, name="Naive")


def train_ridge_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, cfg: Dict[str, Any]) -> tuple[pd.Series, Ridge]:
    model = Ridge(alpha=float(cfg["models"]["ridge"].get("alpha", 1.0)))
    model.fit(X_train, y_train)
    pred = pd.Series(model.predict(X_test), index=X_test.index, name="Ridge")
    return pred, model


def train_rf_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, cfg: Dict[str, Any]) -> tuple[pd.Series, RandomForestRegressor]:
    rf_cfg = cfg["models"].get("random_forest", {})
    model = RandomForestRegressor(
        n_estimators=int(rf_cfg.get("n_estimators", 400)),
        max_depth=rf_cfg.get("max_depth", 8),
        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 3)),
        random_state=int(cfg["project"].get("random_state", 42)),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = pd.Series(model.predict(X_test), index=X_test.index, name="RandomForest")
    return pred, model
