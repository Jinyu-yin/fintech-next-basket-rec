from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from .config import HISTORY_END, SUBMIT_DIR, TIME_COL, USER_COL
from .diagnostic_outputs import write_prediction_distribution, write_prediction_format_report
from .fast_light_cross import (
    COLS,
    SUBMISSION_COLUMNS,
    _build_global_stats,
    _read_behavior,
    score_one_user_candidates,
)


def _quad_key(quad: tuple[str, str, str, str]) -> str:
    return "|".join(quad)


def _safe_log1p(value: float) -> float:
    return float(np.log1p(max(value, 0.0)))


def _candidate_feature_rows(
    g: pd.DataFrame,
    stats: dict[str, Any],
    top_k: int,
) -> tuple[list[tuple[str, str, str, str]], pd.DataFrame]:
    scored = score_one_user_candidates(g, stats, top_k=top_k)
    g = g.sort_values(TIME_COL).copy()
    g["quad"] = list(zip(g["action_typ"], g["prod_typ"], g["prod_sub_typ"], g["rsk_lvl"]))
    cutoff = pd.Timestamp(HISTORY_END)
    g["days_diff"] = (cutoff - g[TIME_COL]).dt.total_seconds().clip(lower=0) / 86400
    recent_1 = g[g["days_diff"] <= 1]
    recent_3 = g[g["days_diff"] <= 3]
    recent_7 = g[g["days_diff"] <= 7]
    recent_base = recent_7 if not recent_7.empty else g

    last_quad = g["quad"].iloc[-1]
    last_action, last_prod, last_sub, last_rsk = last_quad
    user_rsk_counts = g["rsk_lvl"].value_counts(normalize=True)
    user_rsk_mode = str(user_rsk_counts.index[0])
    user_rsk_mode_share = float(user_rsk_counts.iloc[0])
    recent_rsk_mode = str(recent_base["rsk_lvl"].value_counts().index[0])

    quad_counts_all = g["quad"].value_counts()
    quad_counts_1 = recent_1["quad"].value_counts() if not recent_1.empty else pd.Series(dtype=float)
    quad_counts_3 = recent_3["quad"].value_counts() if not recent_3.empty else pd.Series(dtype=float)
    quad_counts_7 = recent_7["quad"].value_counts() if not recent_7.empty else pd.Series(dtype=float)
    action_counts = g["action_typ"].value_counts()
    action_counts_7 = recent_base["action_typ"].value_counts()
    prod_counts = g["prod_typ"].value_counts()
    sub_counts = g["prod_sub_typ"].value_counts()
    prod_sub_counts = g[["prod_typ", "prod_sub_typ"]].value_counts()
    prod_sub_rsk_counts = g[["prod_typ", "prod_sub_typ", "rsk_lvl"]].value_counts()
    action_prod_counts = g[["action_typ", "prod_typ"]].value_counts()
    action_prod_sub_counts = g[["action_typ", "prod_typ", "prod_sub_typ"]].value_counts()
    quad_global_counts = stats["quad_counts"]
    prod_sub_rsk_values = stats["prod_sub_rsk_values"]

    rows = []
    quads = []
    for rank, (quad, rule_score) in enumerate(scored, 1):
        act, prod, sub, rsk = quad
        qdf = g[g["quad"] == quad]
        if qdf.empty:
            days_since_last = 99.0
            decay_sum = 0.0
        else:
            days_since_last = float(qdf["days_diff"].min())
            decay_sum = float(np.exp(-qdf["days_diff"] / 3.0).sum())
        prod_sub_key = (prod, sub)
        prod_sub_rsk_key = (prod, sub, rsk)
        action_prod_key = (act, prod)
        action_prod_sub_key = (act, prod, sub)
        prod_sub_mode_rsks = prod_sub_rsk_values.get(prod_sub_key, [])
        rows.append(
            {
                "rule_score": float(rule_score),
                "rule_rank_inv": 1.0 / np.log2(rank + 1),
                "quad_cnt_all": _safe_log1p(float(quad_counts_all.get(quad, 0))),
                "quad_cnt_1d": _safe_log1p(float(quad_counts_1.get(quad, 0))),
                "quad_cnt_3d": _safe_log1p(float(quad_counts_3.get(quad, 0))),
                "quad_cnt_7d": _safe_log1p(float(quad_counts_7.get(quad, 0))),
                "days_since_last": days_since_last,
                "decay_sum": decay_sum,
                "is_last_quad": float(quad == last_quad),
                "action_cnt_all": _safe_log1p(float(action_counts.get(act, 0))),
                "action_cnt_7d": _safe_log1p(float(action_counts_7.get(act, 0))),
                "is_last_action": float(act == last_action),
                "prod_cnt_all": _safe_log1p(float(prod_counts.get(prod, 0))),
                "sub_cnt_all": _safe_log1p(float(sub_counts.get(sub, 0))),
                "prod_sub_cnt": _safe_log1p(float(prod_sub_counts.get(prod_sub_key, 0))),
                "prod_sub_rsk_cnt": _safe_log1p(float(prod_sub_rsk_counts.get(prod_sub_rsk_key, 0))),
                "action_prod_cnt": _safe_log1p(float(action_prod_counts.get(action_prod_key, 0))),
                "action_prod_sub_cnt": _safe_log1p(float(action_prod_sub_counts.get(action_prod_sub_key, 0))),
                "is_last_prod": float(prod == last_prod),
                "is_last_sub": float(sub == last_sub),
                "is_last_rsk": float(rsk == last_rsk),
                "is_user_rsk_mode": float(rsk == user_rsk_mode),
                "is_recent_rsk_mode": float(rsk == recent_rsk_mode),
                "user_rsk_mode_share": user_rsk_mode_share,
                "is_prod_sub_mode_rsk": float(bool(prod_sub_mode_rsks) and rsk == prod_sub_mode_rsks[0]),
                "global_quad_cnt": _safe_log1p(float(quad_global_counts.get(quad, 0))),
            }
        )
        quads.append(quad)
    return quads, pd.DataFrame(rows)


def build_ranker_training_data(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    stats: dict[str, Any],
    max_train_users: int,
    candidate_k: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    history = train_df[train_df[TIME_COL] <= pd.Timestamp("2025-03-31 23:59:59")].copy()
    target = train_df[(train_df[TIME_COL] >= pd.Timestamp("2025-04-01")) & (train_df[TIME_COL] < pd.Timestamp("2025-04-02"))].copy()
    target["quad_key"] = target["action_typ"].astype(str).str.cat(
        [target["prod_typ"].astype(str), target["prod_sub_typ"].astype(str), target["rsk_lvl"].astype(str)],
        sep="|",
    )
    target_map = target.groupby(USER_COL)["quad_key"].apply(lambda s: set(s.tolist())).to_dict()
    users = [uid for uid in history[USER_COL].drop_duplicates().tolist() if uid in target_map]
    if max_train_users and len(users) > max_train_users:
        users = rng.choice(np.array(users, dtype=object), size=max_train_users, replace=False).tolist()

    eval_users = set(eval_df[USER_COL].drop_duplicates().tolist())
    hist_groups = {uid: g for uid, g in history[history[USER_COL].isin(users)].groupby(USER_COL, sort=False)}
    X_list = []
    y_list = []
    groups = []
    for n, uid in enumerate(users, 1):
        if uid in eval_users:
            continue
        g = hist_groups.get(uid)
        if g is None or g.empty:
            continue
        true_quads = target_map.get(uid, set())
        quads, feats = _candidate_feature_rows(g, stats, top_k=candidate_k)
        labels = np.array([1 if _quad_key(q) in true_quads else 0 for q in quads], dtype=np.int32)
        if labels.sum() == 0:
            continue
        X_list.append(feats)
        y_list.append(labels)
        groups.append(len(labels))
        if n % 2000 == 0:
            print(f"ranker 训练样本用户 {n}/{len(users)}")
    if not X_list:
        raise RuntimeError("没有构造出 ranker 训练样本。")
    return pd.concat(X_list, ignore_index=True), np.concatenate(y_list), groups


def predict_with_ranker(
    model: LGBMRanker,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stats: dict[str, Any],
    candidate_k: int = 100,
    top_k: int = 20,
    return_candidates: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict]]:
    rows = []
    candidate_rows: list[dict] = []
    user_count = test_df[USER_COL].nunique()
    for n, (uid, g) in enumerate(test_df.groupby(USER_COL, sort=False), 1):
        quads, feats = _candidate_feature_rows(g, stats, top_k=candidate_k)
        pred = model.predict(feats)
        if return_candidates:
            for cand_rank, (quad, model_score) in enumerate(zip(quads, pred), 1):
                candidate_rows.append({
                    USER_COL: uid,
                    "quadruple": _quad_key(quad),
                    "candidate_rank": cand_rank,
                    "model_score": float(model_score),
                })
        order = np.argsort(-pred)[:top_k]
        selected = [quads[i] for i in order]
        for idx, (act, prod, sub, rsk) in enumerate(selected, 1):
            rows.append({USER_COL: uid, "action_typ": act, "prod_typ": prod, "prod_sub_typ": sub, "rsk_lvl": rsk, "index": idx})
        if n % 10000 == 0:
            print(f"fast-light-ranker 已预测 {n}/{user_count} 用户")
    pred_df = pd.DataFrame(rows)[SUBMISSION_COLUMNS]
    if return_candidates:
        return pred_df, candidate_rows
    return pred_df


def train_fast_light_ranker(
    train_df: pd.DataFrame,
    eval_history_df: pd.DataFrame,
    max_ranker_train_users: int = 3000,
    candidate_k: int = 100,
    seed: int = 42,
) -> tuple[LGBMRanker, dict[str, Any]]:
    stats = _build_global_stats(train_df[train_df[TIME_COL] <= pd.Timestamp(HISTORY_END)], eval_history_df)
    X, y, groups = build_ranker_training_data(
        train_df=train_df,
        eval_df=eval_history_df,
        stats=stats,
        max_train_users=max_ranker_train_users,
        candidate_k=candidate_k,
        seed=seed,
    )
    print(f"ranker 训练行数：{len(X)}，query 数：{len(groups)}，正样本：{int(y.sum())}")
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y, group=groups, eval_at=[20])
    return model, stats


def run_task1_fast_light_ranker(
    train_path: Path,
    test_path: Path,
    output_name: str = "prediction.csv",
    max_ranker_train_users: int = 30000,
    candidate_k: int = 100,
    seed: int = 42,
) -> Path:
    train_df = _read_behavior(train_path, is_train=True, include_target_day=True)
    test_df = _read_behavior(test_path, is_train=False)
    model, stats = train_fast_light_ranker(train_df, test_df, max_ranker_train_users, candidate_k, seed)
    pred_df = predict_with_ranker(model, train_df, test_df, stats, candidate_k=candidate_k, top_k=20)
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMIT_DIR / output_name
    pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    format_path = write_prediction_format_report(pred_df, SUBMIT_DIR, prefix="prediction")
    dist_path = write_prediction_distribution(pred_df, SUBMIT_DIR, prefix="prediction")
    print(f"✅ fast-light-ranker 提交文件已保存：{out_path}")
    print(f"诊断格式报告：{format_path}")
    print(f"诊断分布报告：{dist_path}")
    return out_path
