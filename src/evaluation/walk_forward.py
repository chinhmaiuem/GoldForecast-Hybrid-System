"""
Walk-forward (expanding window) cross-validation for time-series models.

Thay thế cho single train/test split cố định trong pipeline.py.
Walk-forward giúp đánh giá mô hình trên nhiều giai đoạn thị trường khác nhau,
tránh việc kết quả phụ thuộc quá nhiều vào một khoảng test duy nhất.

Cách dùng trong pipeline.py:
    from src.evaluation.walk_forward import WalkForwardValidator
    wf = WalkForwardValidator(cfg)
    results = wf.run(dataset, step_train_evaluate_fn)
    wf.save_summary(results, out_path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Callable, List, Tuple
import warnings

import numpy as np
import pandas as pd


@dataclass
class WalkForwardResult:
    """Kết quả của một fold trong walk-forward."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    forecast_metrics: pd.DataFrame   # MAE, RMSE, MAPE, DA, R2
    return_metrics: pd.DataFrame     # target MAE, DA, R2
    trading_metrics: pd.DataFrame    # Sharpe, drawdown, return
    pred_df: pd.DataFrame            # predictions cho fold này


@dataclass
class WalkForwardSummary:
    """Tổng hợp kết quả tất cả fold."""
    folds: List[WalkForwardResult] = field(default_factory=list)

    def forecast_summary(self) -> pd.DataFrame:
        """Trung bình và std của forecast metrics qua các fold."""
        frames = [f.forecast_metrics.assign(fold=f.fold) for f in self.folds]
        df = pd.concat(frames, ignore_index=True)
        return (
            df.groupby("model")[["MAE", "RMSE", "MAPE_percent", "price_directional_accuracy_percent", "R2"]]
            .agg(["mean", "std"])
            .round(4)
        )

    def trading_summary(self) -> pd.DataFrame:
        """Trung bình và std của trading metrics qua các fold."""
        frames = [f.trading_metrics.assign(fold=f.fold) for f in self.folds]
        df = pd.concat(frames, ignore_index=True)
        return (
            df.groupby(["model", "strategy_type"])[["total_return_percent", "sharpe", "max_drawdown_percent"]]
            .agg(["mean", "std"])
            .round(4)
        )

    def combined_pred_df(self) -> pd.DataFrame:
        """Ghép predictions của tất cả fold theo thời gian (không overlap)."""
        return pd.concat([f.pred_df for f in self.folds]).sort_index()


class WalkForwardValidator:
    """
    Walk-forward cross-validation với expanding window.

    Config cần có (trong config.yaml):
        walk_forward:
            enabled: true
            n_folds: 5
            min_train_years: 3
            test_months: 6
            mode: "expanding"   # hoặc "rolling" (fixed window)

    Với n_folds=5 và test_months=6, sẽ có 5 fold mỗi fold test 6 tháng,
    tổng test period = 30 tháng = 2.5 năm.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        wf_cfg = cfg.get("walk_forward", {})
        self.enabled = bool(wf_cfg.get("enabled", False))
        self.n_folds = int(wf_cfg.get("n_folds", 5))
        self.test_months = int(wf_cfg.get("test_months", 6))
        self.mode = str(wf_cfg.get("mode", "expanding"))
        # Min train size: tính theo trading days (252/year)
        self.min_train_days = int(wf_cfg.get("min_train_years", 3)) * 252

    def _make_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Tạo danh sách (train_df, test_df) cho từng fold.
        Expanding window: train ngày càng dài thêm theo từng fold.
        Rolling window: train cố định kích thước, chỉ dịch chuyển.
        """
        dates = df.index.sort_values()
        n = len(dates)
        test_size = int(self.test_months * 21)  # ~21 trading days/month

        if n < self.min_train_days + test_size:
            raise ValueError(
                f"Không đủ dữ liệu cho walk-forward: cần ít nhất "
                f"{self.min_train_days + test_size} rows, có {n} rows."
            )

        # Tính vị trí test cuối cùng
        # Fold cuối cùng: test kết thúc ở ngày cuối của dataset
        # Fold đầu tiên: test bắt đầu cách cuối một khoảng n_folds * test_size
        splits = []
        for fold_idx in range(self.n_folds):
            # Fold 0 = sớm nhất, fold (n_folds-1) = gần đây nhất
            test_end_pos = n - (self.n_folds - 1 - fold_idx) * test_size
            test_start_pos = test_end_pos - test_size
            if self.mode == "rolling":
                rolling_days = int(self.cfg.get("walk_forward", {}).get("rolling_train_days", self.min_train_days))
                train_start_pos = max(0, test_start_pos - rolling_days)
            else:
                train_start_pos = 0
            train_end_pos = test_start_pos

            if train_end_pos - train_start_pos < self.min_train_days:
                warnings.warn(
                    f"Fold {fold_idx}: train chỉ có {train_end_pos - train_start_pos} rows "
                    f"(cần {self.min_train_days}). Bỏ qua fold này."
                )
                continue

            train = df.iloc[train_start_pos:train_end_pos].copy()
            test = df.iloc[test_start_pos:test_end_pos].copy()
            splits.append((fold_idx, train, test))

        return splits

    def run(
        self,
        dataset: pd.DataFrame,
        train_eval_fn: Callable[[Dict[str, Any], pd.DataFrame, pd.DataFrame], Dict[str, Any]],
    ) -> WalkForwardSummary:
        """
        Chạy walk-forward cross-validation.

        Args:
            dataset: full dataset (sau feature engineering).
            train_eval_fn: hàm nhận (cfg, train_df, test_df) và trả về dict chứa:
                - "forecast_metrics": pd.DataFrame
                - "return_metrics": pd.DataFrame
                - "trading_metrics": pd.DataFrame
                - "pred_df": pd.DataFrame

        Returns:
            WalkForwardSummary chứa kết quả tất cả fold.
        """
        splits = self._make_splits(dataset)
        summary = WalkForwardSummary()

        for fold_idx, train, test in splits:
            print(
                f"\n[walk_forward] Fold {fold_idx + 1}/{len(splits)}: "
                f"train {train.index.min().date()} → {train.index.max().date()} | "
                f"test {test.index.min().date()} → {test.index.max().date()}"
            )
            # Tạm thời override config dates để pipeline dùng đúng split này
            fold_cfg = dict(self.cfg)
            fold_cfg = _patch_config_dates(fold_cfg, train, test)

            try:
                results = train_eval_fn(fold_cfg, train, test)
                result = WalkForwardResult(
                    fold=fold_idx,
                    train_start=str(train.index.min().date()),
                    train_end=str(train.index.max().date()),
                    test_start=str(test.index.min().date()),
                    test_end=str(test.index.max().date()),
                    forecast_metrics=results["forecast_metrics"],
                    return_metrics=results["return_metrics"],
                    trading_metrics=results["trading_metrics"],
                    pred_df=results["pred_df"],
                )
                summary.folds.append(result)
            except Exception as exc:
                warnings.warn(f"[walk_forward] Fold {fold_idx} thất bại: {exc}")

        return summary

    def save_summary(self, summary: WalkForwardSummary, out_dir: str | Path) -> None:
        """Lưu kết quả tổng hợp ra thư mục artifacts."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fc_summary = summary.forecast_summary()
        td_summary = summary.trading_summary()
        combined = summary.combined_pred_df()

        fc_summary.to_csv(out_dir / "wf_forecast_summary.csv")
        td_summary.to_csv(out_dir / "wf_trading_summary.csv")
        combined.to_csv(out_dir / "wf_combined_predictions.csv")

        # Fold-level detail
        fold_rows = []
        for f in summary.folds:
            base = {
                "fold": f.fold,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
            }
            # Lấy metrics của SARIMAX_LSTM làm đại diện
            for _, row in f.forecast_metrics.iterrows():
                fold_rows.append({**base, "model": row["model"], **{k: row[k] for k in ["MAE", "RMSE", "price_directional_accuracy_percent"]}})
        pd.DataFrame(fold_rows).to_csv(out_dir / "wf_fold_detail.csv", index=False)
        print(f"[walk_forward] Kết quả lưu tại {out_dir}")


def _patch_config_dates(cfg: Dict[str, Any], train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Any]:
    """Cập nhật config dates để phù hợp với fold hiện tại."""
    import copy
    cfg = copy.deepcopy(cfg)
    cfg["data"]["train_start"] = str(train.index.min().date())
    cfg["data"]["train_end"] = str(train.index.max().date())
    cfg["data"]["test_start"] = str(test.index.min().date())
    cfg["data"]["test_end"] = str(test.index.max().date())
    return cfg
