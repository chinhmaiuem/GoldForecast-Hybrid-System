from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    root = project_root()
    path = Path(config_path) if config_path else root / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(root)
    return cfg


def resolve_path(cfg: Dict[str, Any], relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return Path(cfg["_root"]) / path


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
