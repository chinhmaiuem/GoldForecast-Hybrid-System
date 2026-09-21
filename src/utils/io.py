from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index, encoding="utf-8-sig")


def read_csv_date_index(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0])
    df = df.rename(columns={df.columns[0]: "date"}).set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()
