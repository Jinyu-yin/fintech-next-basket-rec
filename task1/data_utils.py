from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import USER_COL, TIME_COL


def load_csv(path: str | Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gbk", "gb18030"):
        try:
            df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(path, sep=None, engine="python", encoding_errors="ignore")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    if len(df.columns) and df.columns[0] == "":
        df = df.rename(columns={df.columns[0]: "row_id"})
    return df


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if USER_COL in out.columns:
        out[USER_COL] = out[USER_COL].astype(str).str.strip()
    if TIME_COL in out.columns:
        out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
    for col in ["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]:
        if col in out.columns:
            out[col] = out[col].fillna("UNK").astype(str).str.strip()
    sort_cols = [c for c in [USER_COL, TIME_COL] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out
