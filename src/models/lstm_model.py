from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


def _make_supervised_2d(series: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_len, len(series)):
        X.append(series[i - seq_len:i])
        y.append(series[i])
    if not X:
        return np.empty((0, seq_len)), np.empty((0,))
    return np.asarray(X), np.asarray(y)


def _make_feature_sequences(X: pd.DataFrame, y: pd.Series, seq_len: int) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    vals = X.values
    target = y.values
    xs, ys, idx = [], [], []
    for i in range(seq_len - 1, len(X)):
        xs.append(vals[i - seq_len + 1:i + 1])
        ys.append(target[i])
        idx.append(X.index[i])
    if not xs:
        return np.empty((0, seq_len, X.shape[1])), np.empty((0,)), pd.Index([])
    return np.asarray(xs), np.asarray(ys), pd.Index(idx)


def _try_import_tf():
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except Exception:  # noqa: BLE001
        return None


@dataclass
class DirectLSTMModel:
    cfg: Dict[str, Any]
    backend_: str = ""
    model_: object | None = field(default=None)

    def fit_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> pd.Series:
        lcfg = self.cfg["models"].get("lstm", {})
        seq_len = int(lcfg.get("seq_len", 20))
        tf = _try_import_tf()

        if tf is not None and len(X_train) > seq_len + 10:
            self.backend_ = "tensorflow.keras.LSTM"
            tf.random.set_seed(int(self.cfg["project"].get("random_state", 42)))
            X_seq, y_seq, _ = _make_feature_sequences(X_train, y_train, seq_len)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(seq_len, X_train.shape[1])),
                tf.keras.layers.LSTM(int(lcfg.get("units", 32))),
                tf.keras.layers.Dropout(float(lcfg.get("dropout", 0.15))),
                tf.keras.layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(
                X_seq, y_seq,
                epochs=int(lcfg.get("epochs", 80)),
                batch_size=int(lcfg.get("batch_size", 32)),
                validation_split=float(lcfg.get("validation_split", 0.10)),
                verbose=0,
            )
            combined_X = pd.concat([X_train.tail(seq_len - 1), X_test])
            dummy_y = pd.Series(np.zeros(len(combined_X)), index=combined_X.index)
            X_test_seq, _, idx = _make_feature_sequences(combined_X, dummy_y, seq_len)
            pred = model.predict(X_test_seq, verbose=0).reshape(-1)
            self.model_ = model
            return pd.Series(pred, index=idx, name="LSTM")

        # Fallback so the pipeline does not break on Python versions without TensorFlow wheels.
        self.backend_ = "fallback_MLPRegressor_not_real_LSTM"
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=800,
            random_state=int(self.cfg["project"].get("random_state", 42)),
            early_stopping=True,
        )
        model.fit(X_train, y_train)
        self.model_ = model
        return pd.Series(model.predict(X_test), index=X_test.index, name="LSTM")


@dataclass
class ResidualSequenceModel:
    cfg: Dict[str, Any]
    backend_: str = ""
    model_: object | None = field(default=None)
    seq_len_: int = 20

    def fit(self, residual_train: pd.Series) -> "ResidualSequenceModel":
        lcfg = self.cfg["models"].get("lstm", {})
        self.seq_len_ = int(lcfg.get("seq_len", 20))
        series = residual_train.fillna(0.0).values.astype(float)
        X, y = _make_supervised_2d(series, self.seq_len_)
        tf = _try_import_tf()

        if tf is not None and len(X) > 10:
            self.backend_ = "tensorflow.keras.LSTM_residual"
            tf.random.set_seed(int(self.cfg["project"].get("random_state", 42)))
            X3 = X.reshape((X.shape[0], X.shape[1], 1))
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.seq_len_, 1)),
                tf.keras.layers.LSTM(int(lcfg.get("units", 32))),
                tf.keras.layers.Dropout(float(lcfg.get("dropout", 0.15))),
                tf.keras.layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(
                X3, y,
                epochs=int(lcfg.get("epochs", 80)),
                batch_size=int(lcfg.get("batch_size", 32)),
                validation_split=float(lcfg.get("validation_split", 0.10)),
                verbose=0,
            )
            self.model_ = model
        else:
            self.backend_ = "fallback_Ridge_residual_not_real_LSTM"
            model = Ridge(alpha=1.0)
            if len(X) == 0:
                X = np.zeros((1, self.seq_len_)); y = np.zeros(1)
            model.fit(X, y)
            self.model_ = model
        return self

    def predict_in_sample(self, residual_train: pd.Series) -> pd.Series:
        series = residual_train.fillna(0.0).values.astype(float)
        preds = np.zeros(len(series))
        if len(series) <= self.seq_len_:
            return pd.Series(preds, index=residual_train.index)
        X, _, idx_offset = [], [], []
        for i in range(self.seq_len_, len(series)):
            X.append(series[i - self.seq_len_:i])
            idx_offset.append(i)
        X = np.asarray(X)
        if self.backend_.startswith("tensorflow"):
            pred = self.model_.predict(X.reshape((X.shape[0], X.shape[1], 1)), verbose=0).reshape(-1)  # type: ignore[union-attr]
        else:
            pred = self.model_.predict(X)  # type: ignore[union-attr]
        preds[idx_offset] = pred
        return pd.Series(preds, index=residual_train.index)

    def predict_one_step_rolling(
        self,
        residual_train: pd.Series,
        y_test: pd.Series,
        sarimax_test_pred: pd.Series,
    ) -> pd.Series:
        history = list(residual_train.fillna(0.0).values.astype(float))
        preds = []
        for dt in y_test.index:
            window = np.asarray(history[-self.seq_len_:], dtype=float)
            if len(window) < self.seq_len_:
                window = np.pad(window, (self.seq_len_ - len(window), 0))
            if self.backend_.startswith("tensorflow"):
                pred = float(self.model_.predict(window.reshape((1, self.seq_len_, 1)), verbose=0).reshape(-1)[0])  # type: ignore[union-attr]
            else:
                pred = float(self.model_.predict(window.reshape(1, -1))[0])  # type: ignore[union-attr]
            preds.append(pred)
            # In one-step-ahead backtest, yesterday's actual residual becomes known before the next forecast.
            actual_resid = float(y_test.loc[dt] - sarimax_test_pred.loc[dt])
            history.append(actual_resid)
        return pd.Series(preds, index=y_test.index, name="LSTM_residual")
