from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import USER_COL
from .fast_light_cross import COLS, SUBMISSION_COLUMNS


def quad_key_df(df: pd.DataFrame) -> pd.Series:
    return df["action_typ"].astype(str).str.cat(
        [df["prod_typ"].astype(str), df["prod_sub_typ"].astype(str), df["rsk_lvl"].astype(str)],
        sep="|",
    )


def split_quads(value: object) -> list[str]:
    if pd.isna(value) or str(value) == "":
        return []
    return str(value).split(" || ")


def write_prediction_format_report(pred_df: pd.DataFrame, out_dir: Path, prefix: str = "prediction") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_user = pred_df.groupby(USER_COL).size()
    rows = [
        {"metric": "rows", "value": len(pred_df)},
        {"metric": "unique_users", "value": pred_df[USER_COL].nunique()},
        {"metric": "index_min", "value": pred_df["index"].min() if "index" in pred_df else np.nan},
        {"metric": "index_max", "value": pred_df["index"].max() if "index" in pred_df else np.nan},
        {"metric": "per_user_min", "value": per_user.min() if not per_user.empty else np.nan},
        {"metric": "per_user_max", "value": per_user.max() if not per_user.empty else np.nan},
        {"metric": "per_user_mean", "value": per_user.mean() if not per_user.empty else np.nan},
        {"metric": "null_total", "value": int(pred_df[SUBMISSION_COLUMNS].isna().sum().sum()) if set(SUBMISSION_COLUMNS).issubset(pred_df.columns) else int(pred_df.isna().sum().sum())},
    ]
    for col in pred_df.columns:
        rows.append({"metric": f"null_{col}", "value": int(pred_df[col].isna().sum())})
    path = out_dir / f"{prefix}_format_report.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_prediction_distribution(pred_df: pd.DataFrame, out_dir: Path, prefix: str = "prediction") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for col in ["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]:
        counts = pred_df[col].astype(str).value_counts()
        total = counts.sum()
        for value, count in counts.items():
            rows.append({"section": col, "value": value, "count": int(count), "share": float(count / total) if total else 0.0})
    quad_counts = quad_key_df(pred_df).value_counts().head(100)
    total_rows = len(pred_df)
    for value, count in quad_counts.items():
        rows.append({"section": "quad_top100", "value": value, "count": int(count), "share": float(count / total_rows) if total_rows else 0.0})
    path = out_dir / f"{prefix}_distribution_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_holdout_compare(pred_df: pd.DataFrame, true_df: pd.DataFrame, out_dir: Path, prefix: str) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{prefix}_holdout_pred.csv"
    true_path = out_dir / f"{prefix}_holdout_true.csv"
    compare_path = out_dir / f"{prefix}_holdout_compare.csv"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    true_df.to_csv(true_path, index=False, encoding="utf-8-sig")

    true_work = true_df.copy()
    true_work["quadruple"] = quad_key_df(true_work)
    target_map = true_work.groupby(USER_COL)["quadruple"].apply(lambda s: set(s.tolist())).to_dict()

    rows = []
    for uid, g in pred_df.sort_values([USER_COL, "index"]).groupby(USER_COL, sort=False):
        true_quads = target_map.get(uid, set())
        pred_quads = quad_key_df(g).tolist()
        pred_set = set(pred_quads[:20])
        hit_ranks = [i + 1 for i, q in enumerate(pred_quads[:20]) if q in true_quads]
        same_aps_wrong_rsk = []
        pred_action_prod_sub = {"|".join(q.split("|")[:3]) for q in pred_quads[:20]}
        for true_quad in true_quads:
            parts = true_quad.split("|")
            if len(parts) == 4 and "|".join(parts[:3]) in pred_action_prod_sub and true_quad not in pred_set:
                same_aps_wrong_rsk.append(true_quad)
        rows.append(
            {
                USER_COL: uid,
                "true_count": len(true_quads),
                "hit_count": len(pred_set & true_quads),
                "user_recall": len(pred_set & true_quads) / len(true_quads) if true_quads else 0.0,
                "hit_ranks": ";".join(map(str, hit_ranks)),
                "same_action_prod_sub_wrong_rsk_count": len(same_aps_wrong_rsk),
                "same_action_prod_sub_wrong_rsk_quads": " || ".join(sorted(same_aps_wrong_rsk)),
                "true_quads": " || ".join(sorted(true_quads)),
                "pred_top20": " || ".join(pred_quads[:20]),
                "missing_from_final": " || ".join(sorted(true_quads - pred_set)),
            }
        )
    pd.DataFrame(rows).to_csv(compare_path, index=False, encoding="utf-8-sig")
    return pred_path, true_path, compare_path


def write_candidate_recall_diagnostic(
    candidate_rows: Iterable[dict],
    true_df: pd.DataFrame,
    final_pred_df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    true_work = true_df.copy()
    true_work["quadruple"] = quad_key_df(true_work)
    true_map = true_work.groupby(USER_COL)["quadruple"].apply(lambda s: set(s.tolist())).to_dict()

    final_work = final_pred_df.copy()
    final_work["quadruple"] = quad_key_df(final_work)
    final_map = final_work.groupby(USER_COL)["quadruple"].apply(lambda s: set(s.tolist())).to_dict()

    cand_map: dict[str, set[str]] = {}
    cand_count: Counter[str] = Counter()
    for row in candidate_rows:
        uid = str(row[USER_COL])
        quad = row.get("quadruple")
        if quad is None:
            quad_tuple = row.get("quad")
            quad = "|".join(quad_tuple) if quad_tuple else ""
        if not quad:
            continue
        cand_map.setdefault(uid, set()).add(str(quad))
        cand_count[uid] += 1

    rows = []
    all_users = sorted(set(true_map) | set(final_map) | set(cand_map))
    for uid in all_users:
        true_quads = true_map.get(uid, set())
        cand_quads = cand_map.get(uid, set())
        final_quads = final_map.get(uid, set())
        cand_hits = true_quads & cand_quads
        final_hits = true_quads & final_quads
        rows.append(
            {
                USER_COL: uid,
                "true_count": len(true_quads),
                "candidate_count_unique": len(cand_quads),
                "candidate_count_raw": int(cand_count.get(uid, len(cand_quads))),
                "candidate_hit_count": len(cand_hits),
                "candidate_recall": len(cand_hits) / len(true_quads) if true_quads else 0.0,
                "final_hit_count": len(final_hits),
                "final_recall": len(final_hits) / len(true_quads) if true_quads else 0.0,
                "missing_from_candidates": " || ".join(sorted(true_quads - cand_quads)),
                "missing_from_final": " || ".join(sorted(true_quads - final_quads)),
                "candidate_only_hits": " || ".join(sorted(cand_hits - final_hits)),
            }
        )
    path = out_dir / f"{prefix}_candidate_recall_diagnostic.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def summarize_holdout_files(compare_df: pd.DataFrame, candidate_diag_df: pd.DataFrame | None = None) -> dict[str, float]:
    true_total = float(compare_df["true_count"].sum()) if not compare_df.empty else 0.0
    hit_total = float(compare_df["hit_count"].sum()) if not compare_df.empty else 0.0
    out = {
        "holdout_users": float(len(compare_df)),
        "final_micro_recall": hit_total / true_total if true_total else 0.0,
        "final_mean_user_recall": float(compare_df["user_recall"].mean()) if not compare_df.empty else 0.0,
        "zero_hit_users": float((compare_df["hit_count"] == 0).sum()) if not compare_df.empty else 0.0,
        "full_hit_users": float((compare_df["hit_count"] == compare_df["true_count"]).sum()) if not compare_df.empty else 0.0,
    }
    if candidate_diag_df is not None and not candidate_diag_df.empty:
        cand_true_total = float(candidate_diag_df["true_count"].sum())
        cand_hit_total = float(candidate_diag_df["candidate_hit_count"].sum())
        out["candidate_micro_recall"] = cand_hit_total / cand_true_total if cand_true_total else 0.0
        out["candidate_mean_user_recall"] = float(candidate_diag_df["candidate_recall"].mean())
        out["candidate_to_final_gap"] = out["candidate_micro_recall"] - out["final_micro_recall"]
    return out
