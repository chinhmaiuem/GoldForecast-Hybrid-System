from src.utils.config import load_config
from src.pipeline import run_all


if __name__ == "__main__":
    cfg = load_config()
    run_all(cfg)
