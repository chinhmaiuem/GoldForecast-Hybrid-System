from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureFilter:
    cfg: Dict[str, Any]
    feature_cols_: List[str] = field(default_factory=list)
    selected_features_: List[str] = field(default_factory=list)
    medians_: pd.Series | None = None
    lower_: pd.Series | None = None
    upper_: pd.Series | None = None
    dropped_correlated_: List[str] = field(default_factory=list)
    scaler_: StandardScaler | None = None
    feature_importance_: pd.DataFrame = field(default_factory=pd.DataFrame)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "FeatureFilter":
        pp = self.cfg["preprocessing"]
        fs_cfg = pp.get("feature_selection", {})
        self.feature_cols_ = list(X_train.columns)

        X = X_train.copy().replace([np.inf, -np.inf], np.nan)
        self.medians_ = X.median(numeric_only=True)
        X = X.fillna(self.medians_)

        winsor_cfg = pp.get("winsorize", {})
        if winsor_cfg.get("enabled", True):
            self.lower_ = X.quantile(float(winsor_cfg.get("lower_q", 0.01)))
            self.upper_ = X.quantile(float(winsor_cfg.get("upper_q", 0.99)))
            X = X.clip(self.lower_, self.upper_, axis=1)
        else:
            self.lower_ = pd.Series(index=X.columns, dtype=float)
            self.upper_ = pd.Series(index=X.columns, dtype=float)

        # Correlation filter fitted on train only.
        corr_threshold = float(fs_cfg.get("correlation_threshold", 0.95))
        corr = X.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.dropped_correlated_ = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        X = X.drop(columns=self.dropped_correlated_, errors="ignore")

        # RF feature selection fitted on train only.
        if fs_cfg.get("enabled", True) and X.shape[1] > 0:
            rf = RandomForestRegressor(
                n_estimators=int(fs_cfg.get("rf_estimators", 300)),
                random_state=int(self.cfg["project"].get("random_state", 42)),
                n_jobs=-1,
                min_samples_leaf=3,
            )
            rf.fit(X, y_train)
            fi = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
            fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
            top_k = min(int(fs_cfg.get("top_k", 18)), len(fi))
            self.selected_features_ = fi.head(top_k)["feature"].tolist()
            self.feature_importance_ = fi
        else:
            self.selected_features_ = list(X.columns)
            self.feature_importance_ = pd.DataFrame({"feature": X.columns, "importance": 1.0})

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.selected_features_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None or self.scaler_ is None:
            raise RuntimeError("FeatureFilter must be fitted before transform.")
        X2 = X.copy().reindex(columns=self.feature_cols_)
        X2 = X2.replace([np.inf, -np.inf], np.nan).fillna(self.medians_)
        if self.lower_ is not None and len(self.lower_) > 0:
            X2 = X2.clip(self.lower_, self.upper_, axis=1)
        X2 = X2.drop(columns=self.dropped_correlated_, errors="ignore")
        X2 = X2.reindex(columns=self.selected_features_)
        arr = self.scaler_.transform(X2)
        return pd.DataFrame(arr, index=X.index, columns=self.selected_features_)

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        self.fit(X_train, y_train)
        return self.transform(X_train)


def build_quality_report(
    raw_report: Optional[pd.DataFrame],
    dataset: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_filter: FeatureFilter,
    validation: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    if raw_report is not None and not raw_report.empty:
        rows.extend(raw_report.to_dict("records"))
    rows += [
        {"step": "dataset_rows_after_feature_engineering", "value": len(dataset)},
        {"step": "train_rows_final_model", "value": len(train)},
        {"step": "validation_rows_for_threshold_selection", "value": 0 if validation is None else len(validation)},
        {"step": "test_rows_final_evaluation", "value": len(test)},
        {"step": "initial_feature_count", "value": len(feature_filter.feature_cols_)},
        {"step": "dropped_correlated_feature_count", "value": len(feature_filter.dropped_correlated_)},
        {"step": "selected_feature_count", "value": len(feature_filter.selected_features_)},
        {"step": "train_start_final_model", "value": str(train.index.min().date())},
        {"step": "train_end_final_model", "value": str(train.index.max().date())},
        {"step": "validation_start", "value": "" if validation is None or validation.empty else str(validation.index.min().date())},
        {"step": "validation_end", "value": "" if validation is None or validation.empty else str(validation.index.max().date())},
        {"step": "test_start", "value": str(test.index.min().date())},
        {"step": "test_end", "value": str(test.index.max().date())},
    ]
    return pd.DataFrame(rows)
