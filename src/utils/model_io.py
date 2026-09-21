from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle
import joblib


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pickle(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f)


def save_joblib(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)


def save_keras_model_if_possible(model: Any, path: str | Path) -> bool:
    """Save a TensorFlow/Keras model if the object supports .save().

    Returns True if a Keras-style save succeeded, otherwise False.
    """
    if model is None or not hasattr(model, "save"):
        return False
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(p)
        return True
    except Exception:
        return False
