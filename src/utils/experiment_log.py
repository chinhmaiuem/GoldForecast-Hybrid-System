"""
Experiment tracking đơn giản, không cần MLflow hay Weights&Biases.
Lưu mỗi lần chạy pipeline vào CSV để theo dõi tiến trình và reproduce.

Thông tin được log:
- Timestamp và config hash (để biết config nào cho kết quả nào)
- Tất cả metrics chính (MAE, Sharpe, etc.)
- Git commit hash nếu có
- Thời gian chạy
- Backend LSTM (tensorflow hay fallback)

Cách dùng:
    from src.utils.experiment_log import ExperimentLogger
    logger = ExperimentLogger("artifacts/metrics/experiment_log.csv")
    logger.log(cfg, forecast_metrics_df, trading_metrics_df, extra={"lstm_backend": "tensorflow"})
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd


def _config_hash(cfg: Dict[str, Any]) -> str:
    """SHA8 của config serialized. Dùng để nhận ra khi config thay đổi."""
    try:
        cfg_str = json.dumps(cfg, sort_keys=True, default=str)
        return hashlib.sha256(cfg_str.encode()).hexdigest()[:8]
    except Exception:
        return "unknown"


def _git_commit() -> str:
    """Lấy git commit hash ngắn. Trả về 'no-git' nếu không có."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "no-git"
    except Exception:
        return "no-git"


class ExperimentLogger:
    """
    Log kết quả mỗi lần chạy pipeline vào một file CSV duy nhất.
    Append-only: không xóa kết quả cũ.
    """

    def __init__(self, log_path: str | Path = "artifacts/metrics/experiment_log.csv"):
        self.log_path = Path(log_path)
        self._start_time = time.time()

    def start(self) -> None:
        """Đặt mốc thời gian bắt đầu. Gọi đầu pipeline."""
        self._start_time = time.time()

    def log(
        self,
        cfg: Dict[str, Any],
        forecast_metrics: pd.DataFrame,
        trading_metrics: pd.DataFrame,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Ghi một dòng vào log file.

        Args:
            cfg: config dict (để tính hash).
            forecast_metrics: DataFrame từ forecast_metrics().
            trading_metrics: DataFrame từ run_backtest().
            extra: thông tin thêm tùy ý (lstm_backend, walk_forward_fold, etc.)
        """
        elapsed = round(time.time() - self._start_time, 1)
        row: Dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config_hash": _config_hash(cfg),
            "git_commit": _git_commit(),
            "runtime_seconds": elapsed,
            "train_start": cfg.get("data", {}).get("train_start", ""),
            "train_end": cfg.get("data", {}).get("train_end", ""),
            "validation_start": cfg.get("data", {}).get("validation_start", ""),
            "validation_end": cfg.get("data", {}).get("validation_end", ""),
            "test_start": cfg.get("data", {}).get("test_start", ""),
            "n_features": cfg.get("preprocessing", {}).get("feature_selection", {}).get("top_k", ""),
            "sarimax_order": str(cfg.get("models", {}).get("sarimax", {}).get("order", "")),
            "lstm_units": cfg.get("models", {}).get("lstm", {}).get("units", ""),
            "lstm_epochs": cfg.get("models", {}).get("lstm", {}).get("epochs", ""),
            "hybrid_alpha": cfg.get("hybrid", {}).get("lstm_residual_weight", ""),
        }

        # Trích metrics của SARIMAX_LSTM từ forecast_metrics
        if not forecast_metrics.empty:
            sl_fm = forecast_metrics[forecast_metrics["model"] == "SARIMAX_LSTM"]
            if not sl_fm.empty:
                r = sl_fm.iloc[0]
                row.update({
                    "sl_MAE": round(r["MAE"], 4),
                    "sl_RMSE": round(r["RMSE"], 4),
                    "sl_price_DA_pct": round(r.get("price_directional_accuracy_percent", float("nan")), 2),
                    "sl_R2": round(r["R2"], 4),
                })

        # Trích trading metrics của SARIMAX_LSTM LongHold
        if not trading_metrics.empty:
            sl_tm = trading_metrics[
                (trading_metrics["model"] == "SARIMAX_LSTM") &
                (trading_metrics["strategy_type"].isin(["LongHold", "LongHoldTrailing"]))
            ]
            if not sl_tm.empty:
                r = sl_tm.sort_values("sharpe", ascending=False).iloc[0]
                row.update({
                    "sl_total_return_pct": round(r["total_return_percent"], 2),
                    "sl_sharpe": round(r["sharpe"], 3),
                    "sl_max_dd_pct": round(r["max_drawdown_percent"], 2),
                    "sl_win_rate_pct": round(r["win_rate_percent"], 1),
                })

        if extra:
            for k, v in extra.items():
                row[f"extra_{k}"] = v

        # Append vào file
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        new_row_df = pd.DataFrame([row])
        if self.log_path.exists():
            existing = pd.read_csv(self.log_path)
            updated = pd.concat([existing, new_row_df], ignore_index=True)
        else:
            updated = new_row_df
        updated.to_csv(self.log_path, index=False, encoding="utf-8-sig")
        print(f"[experiment_log] Ghi kết quả vào {self.log_path} ({elapsed:.1f}s)")

    def load_history(self) -> pd.DataFrame:
        """Đọc toàn bộ lịch sử experiment."""
        if not self.log_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.log_path)

    def best_run(self, metric: str = "sl_sharpe") -> pd.Series:
        """Tìm run có metric tốt nhất."""
        history = self.load_history()
        if history.empty or metric not in history.columns:
            return pd.Series(dtype=object)
        return history.sort_values(metric, ascending=False).iloc[0]
