from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402
from src.pipeline import run_all  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "config.yaml")
    run_all(cfg)
