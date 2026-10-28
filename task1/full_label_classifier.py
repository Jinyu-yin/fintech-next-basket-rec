from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .config import HISTORY_END, SUBMIT_DIR, TIME_COL, USER_COL
from .diagnostic_outputs import write_prediction_distribution, write_prediction_format_report
from .fast_light_cross import COLS, SUBMISSION_COLUMNS, _read_behavior

TARGET_START = pd.Timestamp("2025-04-01 00:00:00")
TARGET_END = pd.Timestamp("2025-04-02 00:00:00")
HISTORY_CUTOFF = pd.Timestamp("2025-03-31 23:59:59")
TRAIN_WINDOWS = [
    (pd.Timestamp("2025-03-24 23:59:59"), pd.Timestamp("2025-03-25"), pd.Timestamp("2025-03-26"), 0.7),
    (pd.Timestamp("2025-03-27 23:59:59"), pd.Timestamp("2025-03-28"), pd.Timestamp("2025-03-29"), 0.9),
    (pd.Timestamp("2025-03-29 23:59:59"), pd.Timestamp("2025-03-30"), pd.Timestamp("2025-03-31"), 1.1),
    (pd.Timestamp("2025-03-31 23:59:59"), pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-02"), 1.8),
]


def quad_key(quad: tuple[str, str, str, str]) -> str:
    return "|".join(quad)


def add_quad_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quad_key"] = df["action_typ"].astype(str).str.cat(
        [df["prod_typ"].astype(str), df["prod_sub_typ"].astype(str), df["rsk_lvl"].astype(str)],
        sep="|",
    )
    return df


def build_label_space(*dfs: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    rows = []
    for df in dfs:
        if df.empty:
            continue
        tmp = add_quad_key(df)
        rows.extend(tmp[["quad_key", *COLS]].drop_duplicates().itertuples(index=False, name=None))
    label_df = pd.DataFrame(rows, columns=["quad_key", *COLS]).drop_duplicates("quad_key")
    if top_n:
        counts = Counter()
        for df in dfs:
            if not df.empty:
                counts.update(add_quad_key(df)["quad_key"].tolist())
        label_df["_cnt"] = label_df["quad_key"].map(counts).fillna(0)
        label_df = label_df.sort_values("_cnt", ascending=False).head(top_n).drop(columns="_cnt")
    return label_df.reset_index(drop=True)


def _safe_log(value: float) -> float:
    return float(np.log1p(max(value, 0.0)))


def _user_stats(g: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any]:
    g = g.sort_values(TIME_COL).copy()
    g = add_quad_key(g)
    days = (cutoff - g[TIME_COL]).dt.total_seconds().clip(lower=0) / 86400
    g["days_diff"] = days
    recent_1 = g[g["days_diff"] <= 1]
    recent_3 = g[g["days_diff"] <= 3]
    recent_7 = g[g["days_diff"] <= 7]
    recent_base = recent_7 if not recent_7.empty else g
    last = g.iloc[-1]
    return {
        "event_count": len(g),
        "unique_quads": g["quad_key"].nunique(),
        "last_quad": str(last["quad_key"]),
        "last_action": str(last["action_typ"]),
        "last_prod": str(last["prod_typ"]),
        "last_sub": str(last["prod_sub_typ"]),
        "last_rsk": str(last["rsk_lvl"]),
        "quad_all": g["quad_key"].value_counts().to_dict(),
        "quad_1": recent_1["quad_key"].value_counts().to_dict(),
        "quad_3": recent_3["quad_key"].value_counts().to_dict(),
        "quad_7": recent_7["quad_key"].value_counts().to_dict(),
        "action_all": g["action_typ"].value_counts().to_dict(),
        "action_7": recent_base["action_typ"].value_counts().to_dict(),
        "prod_all": g["prod_typ"].value_counts().to_dict(),
        "prod_7": recent_base["prod_typ"].value_counts().to_dict(),
        "sub_all": g["prod_sub_typ"].value_counts().to_dict(),
        "sub_7": recent_base["prod_sub_typ"].value_counts().to_dict(),
        "rsk_all": g["rsk_lvl"].value_counts(normalize=False).to_dict(),
        "rsk_7": recent_base["rsk_lvl"].value_counts().to_dict(),
        "prod_sub": g[["prod_typ", "prod_sub_typ"]].value_counts().to_dict(),
        "prod_sub_rsk": g[["prod_typ", "prod_sub_typ", "rsk_lvl"]].value_counts().to_dict(),
        "action_prod": g[["action_typ", "prod_typ"]].value_counts().to_dict(),
        "action_prod_sub": g[["action_typ", "prod_typ", "prod_sub_typ"]].value_counts().to_dict(),
        "quad_last_days": g.groupby("quad_key")["days_diff"].min().to_dict(),
        "action_last_days": g.groupby("action_typ")["days_diff"].min().to_dict(),
        "prod_sub_last_days": g.groupby(["prod_typ", "prod_sub_typ"])["days_diff"].min().to_dict(),
        "quad_decay": g.groupby("quad_key")["days_diff"].apply(lambda s: float(np.exp(-s / 3.0).sum())).to_dict(),
    }


@dataclass(frozen=True)
class LabelMeta:
    keys: list[str]
    actions: list[str]
    prods: list[str]
    subs: list[str]
    rsks: list[str]
    key_to_idx: dict[str, int]
    global_hot_idx: list[int]
    action_to_idx: dict[str, set[int]]
    prod_to_idx: dict[str, set[int]]
    sub_to_idx: dict[str, set[int]]
    rsk_to_idx: dict[str, set[int]]


def build_label_meta(label_df: pd.DataFrame, global_counts: dict[str, int]) -> LabelMeta:
    keys = label_df["quad_key"].astype(str).tolist()
    actions = label_df["action_typ"].astype(str).tolist()
    prods = label_df["prod_typ"].astype(str).tolist()
    subs = label_df["prod_sub_typ"].astype(str).tolist()
    rsks = label_df["rsk_lvl"].astype(str).tolist()
    key_to_idx = {q: i for i, q in enumerate(keys)}
    global_hot_idx = [key_to_idx[q] for q, _ in sorted(global_counts.items(), key=lambda x: x[1], reverse=True) if q in key_to_idx][:80]
    action_to_idx: dict[str, set[int]] = {}
    prod_to_idx: dict[str, set[int]] = {}
    sub_to_idx: dict[str, set[int]] = {}
    rsk_to_idx: dict[str, set[int]] = {}
    for i, (act, prod, sub, rsk) in enumerate(zip(actions, prods, subs, rsks)):
        action_to_idx.setdefault(act, set()).add(i)
        prod_to_idx.setdefault(prod, set()).add(i)
        sub_to_idx.setdefault(sub, set()).add(i)
        rsk_to_idx.setdefault(rsk, set()).add(i)
    return LabelMeta(keys, actions, prods, subs, rsks, key_to_idx, global_hot_idx, action_to_idx, prod_to_idx, sub_to_idx, rsk_to_idx)


def _stable_code(value: str, modulo: int) -> float:
    return float(abs(hash(value)) % modulo) / float(modulo)


def _features_for_user_labels(
    stats: dict[str, Any],
    label_df: pd.DataFrame,
    global_counts: dict[str, int],
    global_target_rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    global_target_rates = global_target_rates or {}
    rows = []
    event_count = max(float(stats["event_count"]), 1.0)
    unique_quads = max(float(stats["unique_quads"]), 1.0)
    for row in label_df.itertuples(index=False):
        q = row.quad_key
        act = row.action_typ
        prod = row.prod_typ
        sub = row.prod_sub_typ
        rsk = row.rsk_lvl
        prod_sub = (prod, sub)
        prod_sub_rsk = (prod, sub, rsk)
        action_prod = (act, prod)
        action_prod_sub = (act, prod, sub)
        q_cnt = float(stats["quad_all"].get(q, 0))
        act_cnt = float(stats["action_all"].get(act, 0))
        prod_cnt = float(stats["prod_all"].get(prod, 0))
        sub_cnt = float(stats["sub_all"].get(sub, 0))
        rsk_cnt = float(stats["rsk_all"].get(rsk, 0))
        ps_cnt = float(stats["prod_sub"].get(prod_sub, 0))
        psr_cnt = float(stats["prod_sub_rsk"].get(prod_sub_rsk, 0))
        ap_cnt = float(stats["action_prod"].get(action_prod, 0))
        aps_cnt = float(stats["action_prod_sub"].get(action_prod_sub, 0))
        rows.append(
            {
                "quad_cnt_all": _safe_log(q_cnt),
                "quad_cnt_1d": _safe_log(float(stats["quad_1"].get(q, 0))),
                "quad_cnt_3d": _safe_log(float(stats["quad_3"].get(q, 0))),
                "quad_cnt_7d": _safe_log(float(stats["quad_7"].get(q, 0))),
                "quad_share": q_cnt / event_count,
                "quad_days_since": float(stats["quad_last_days"].get(q, 99.0)),
                "quad_decay": float(stats["quad_decay"].get(q, 0.0)),
                "action_cnt_all": _safe_log(act_cnt),
                "action_cnt_7d": _safe_log(float(stats["action_7"].get(act, 0))),
                "action_share": act_cnt / event_count,
                "action_days_since": float(stats["action_last_days"].get(act, 99.0)),
                "prod_cnt_all": _safe_log(prod_cnt),
                "prod_cnt_7d": _safe_log(float(stats["prod_7"].get(prod, 0))),
                "prod_share": prod_cnt / event_count,
                "sub_cnt_all": _safe_log(sub_cnt),
                "sub_cnt_7d": _safe_log(float(stats["sub_7"].get(sub, 0))),
                "sub_share": sub_cnt / event_count,
                "rsk_cnt_all": _safe_log(rsk_cnt),
                "rsk_cnt_7d": _safe_log(float(stats["rsk_7"].get(rsk, 0))),
                "rsk_share": rsk_cnt / event_count,
                "prod_sub_cnt": _safe_log(ps_cnt),
                "prod_sub_share": ps_cnt / event_count,
                "prod_sub_days_since": float(stats["prod_sub_last_days"].get(prod_sub, 99.0)),
                "prod_sub_rsk_cnt": _safe_log(psr_cnt),
                "action_prod_cnt": _safe_log(ap_cnt),
                "action_prod_sub_cnt": _safe_log(aps_cnt),
                "is_last_quad": float(q == stats["last_quad"]),
                "is_last_action": float(act == stats["last_action"]),
                "is_last_prod": float(prod == stats["last_prod"]),
                "is_last_sub": float(sub == stats["last_sub"]),
                "is_last_rsk": float(rsk == stats["last_rsk"]),
                "user_event_count_log": _safe_log(event_count),
                "user_unique_quads_log": _safe_log(unique_quads),
                "user_repetition": 1.0 - unique_quads / event_count,
                "global_quad_cnt": _safe_log(float(global_counts.get(q, 0))),
                "global_target_rate": float(global_target_rates.get(q, 0.0)),
                "label_action_code": _stable_code(act, 1000),
                "label_prod_code": _stable_code(prod, 1000),
                "label_sub_code": _stable_code(sub, 2000),
                "label_rsk_code": _stable_code(rsk, 100),
                "is_nonwealth_label": float(prod == "非财富"),
                "is_offline_label": float(act == "线下接触"),
                "is_browse_detail_label": float(act == "浏览详情"),
                "user_has_nonwealth": float(stats["prod_all"].get("非财富", 0) > 0),
                "user_has_offline": float(stats["action_all"].get("线下接触", 0) > 0),
            }
        )
    return pd.DataFrame(rows)


def _build_dataset(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    label_df: pd.DataFrame,
    global_counts: dict[str, int],
    global_target_rates: dict[str, float],
    max_users: int,
    neg_per_user: int,
    seed: int,
    variant: str = "v2",
    cutoff: pd.Timestamp = HISTORY_CUTOFF,
    sample_weight: float = 1.0,
    use_global_hot_negatives: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target = add_quad_key(target_df)
    target_map = target.groupby(USER_COL)["quad_key"].apply(lambda s: set(s.tolist())).to_dict()
    users = [uid for uid in history_df[USER_COL].drop_duplicates().tolist() if uid in target_map]
    if max_users and len(users) > max_users:
        users = rng.choice(np.array(users, dtype=object), size=max_users, replace=False).tolist()
    label_meta = build_label_meta(label_df, global_counts)
    label_keys = label_meta.keys
    key_to_idx = label_meta.key_to_idx
    all_idx_set = set(range(len(label_keys)))
    groups = {uid: g for uid, g in history_df[history_df[USER_COL].isin(users)].groupby(USER_COL, sort=False)}
    X_parts = []
    y_parts = []
    w_parts = []
    for n, uid in enumerate(users, 1):
        g = groups.get(uid)
        if g is None or g.empty:
            continue
        positives = [q for q in target_map.get(uid, set()) if q in key_to_idx]
        if not positives:
            continue
        pos_idx = [key_to_idx[q] for q in positives]
        pos_set = set(pos_idx)
        stats = _user_stats(g, cutoff)
        if variant == "v1":
            neg_pool = np.array([i for i in range(len(label_keys)) if i not in pos_set], dtype=np.int32)
            neg_size = min(neg_per_user, len(neg_pool))
            neg_idx = rng.choice(neg_pool, size=neg_size, replace=False).tolist() if neg_size else []
        else:
            # Hard negatives：用户历史出现过但目标日没出现、字段相似、全局热门，再混少量随机负样本。
            hard_idx = {key_to_idx[q] for q in stats["quad_all"].keys() if q in key_to_idx} - pos_set
            user_actions = {str(x) for x in stats["action_all"].keys()}
            user_prods = {str(x) for x in stats["prod_all"].keys()}
            user_subs = {str(x) for x in stats["sub_all"].keys()}
            user_rsks = {str(x) for x in stats["rsk_all"].keys()}
            action_idx = set().union(*(label_meta.action_to_idx.get(x, set()) for x in user_actions)) if user_actions else set()
            prod_idx = set().union(*(label_meta.prod_to_idx.get(x, set()) for x in user_prods)) if user_prods else set()
            sub_idx = set().union(*(label_meta.sub_to_idx.get(x, set()) for x in user_subs)) if user_subs else set()
            rsk_idx = set().union(*(label_meta.rsk_to_idx.get(x, set()) for x in user_rsks)) if user_rsks else set()
            similar_idx = (action_idx & prod_idx & (sub_idx | rsk_idx)) - pos_set
            global_hot_idx = set(label_meta.global_hot_idx) - pos_set if use_global_hot_negatives else set()
            selected_neg_idx: set[int] = set()
            sources = [(hard_idx, neg_per_user // 3), (similar_idx, neg_per_user // 3)]
            if use_global_hot_negatives:
                sources.append((global_hot_idx, neg_per_user // 5))
            for source_idx, quota in sources:
                available = list(source_idx - selected_neg_idx - pos_set)
                if available:
                    take = min(quota, len(available))
                    selected_neg_idx.update(rng.choice(np.array(available, dtype=np.int32), size=take, replace=False).tolist())
            remaining = list(all_idx_set - pos_set - selected_neg_idx)
            if len(selected_neg_idx) < neg_per_user and remaining:
                take = min(neg_per_user - len(selected_neg_idx), len(remaining))
                selected_neg_idx.update(rng.choice(np.array(remaining, dtype=np.int32), size=take, replace=False).tolist())
            neg_idx = list(selected_neg_idx)
        selected_idx = pos_idx + neg_idx
        selected_labels = label_df.iloc[selected_idx].reset_index(drop=True)
        feats = _features_for_user_labels(stats, selected_labels, global_counts, global_target_rates)
        labels = np.array([1] * len(pos_idx) + [0] * len(neg_idx), dtype=np.int8)
        X_parts.append(feats)
        y_parts.append(labels)
        w_parts.append(np.full(len(labels), sample_weight, dtype=np.float32))
        if n % 2000 == 0:
            print(f"full-label 训练样本用户 {n}/{len(users)}")
    if not X_parts:
        raise RuntimeError("没有构造出 full-label 训练样本。")
    return pd.concat(X_parts, ignore_index=True), np.concatenate(y_parts), np.concatenate(w_parts)


def _build_multi_window_dataset(
    train_df: pd.DataFrame,
    label_df: pd.DataFrame,
    global_counts: dict[str, int],
    max_train_users: int,
    neg_per_user: int,
    seed: int,
    variant: str,
    use_target_prior: bool = True,
    use_global_hot_negatives: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, float]]:
    X_parts = []
    y_parts = []
    w_parts = []
    all_target_counts: Counter[str] = Counter()
    all_target_users = 0
    for i, (cutoff, target_start, target_end, weight) in enumerate(TRAIN_WINDOWS):
        history = train_df[train_df[TIME_COL] <= cutoff].copy()
        target = train_df[(train_df[TIME_COL] >= target_start) & (train_df[TIME_COL] < target_end)].copy()
        if history.empty or target.empty:
            continue
        all_target_counts.update(add_quad_key(target)["quad_key"].tolist())
        all_target_users += int(target[USER_COL].nunique())
    target_user_count = max(float(all_target_users), 1.0)
    global_target_rates = {q: float(c) / target_user_count for q, c in all_target_counts.items()} if use_target_prior else {}
    for i, (cutoff, target_start, target_end, weight) in enumerate(TRAIN_WINDOWS):
        history = train_df[train_df[TIME_COL] <= cutoff].copy()
        target = train_df[(train_df[TIME_COL] >= target_start) & (train_df[TIME_COL] < target_end)].copy()
        if history.empty or target.empty:
            continue
        print(f"multi-window {target_start.date()} 样本构造，权重={weight}")
        X, y, w = _build_dataset(
            history,
            target,
            label_df,
            global_counts,
            global_target_rates,
            max_train_users,
            neg_per_user,
            seed + i,
            variant,
            cutoff,
            weight,
            use_global_hot_negatives,
        )
        X_parts.append(X)
        y_parts.append(y)
        w_parts.append(w)
    if not X_parts:
        raise RuntimeError("没有构造出 multi-window 训练样本。")
    return pd.concat(X_parts, ignore_index=True), np.concatenate(y_parts), np.concatenate(w_parts), global_target_rates


def train_full_label_classifier(
    train_df: pd.DataFrame,
    test_history_df: pd.DataFrame,
    max_train_users: int = 30000,
    neg_per_user: int = 120,
    seed: int = 42,
    variant: str = "v2",
    multi_window: bool = False,
    n_jobs: int = -1,
    use_target_prior: bool = True,
    use_global_hot_negatives: bool = True,
) -> tuple[LGBMClassifier, pd.DataFrame, dict[str, int]]:
    history = train_df[train_df[TIME_COL] <= HISTORY_CUTOFF].copy()
    target = train_df[(train_df[TIME_COL] >= TARGET_START) & (train_df[TIME_COL] < TARGET_END)].copy()
    label_df = build_label_space(history, target, test_history_df)
    global_counts = add_quad_key(history)["quad_key"].value_counts().to_dict()
    print(f"full-label 标签空间：{len(label_df)}")
    if multi_window:
        X, y, sample_weight, global_target_rates = _build_multi_window_dataset(
            train_df, label_df, global_counts, max_train_users, neg_per_user, seed, variant, use_target_prior, use_global_hot_negatives
        )
    else:
        target_counts = add_quad_key(target)["quad_key"].value_counts().to_dict()
        target_user_count = max(float(target[USER_COL].nunique()), 1.0)
        global_target_rates = {q: float(c) / target_user_count for q, c in target_counts.items()} if use_target_prior else {}
        X, y, sample_weight = _build_dataset(
            history,
            target,
            label_df,
            global_counts,
            global_target_rates,
            max_train_users,
            neg_per_user,
            seed,
            variant,
            HISTORY_CUTOFF,
            1.0,
            use_global_hot_negatives,
        )
    print(f"full-label 训练行数：{len(X)}，正样本：{int(y.sum())}，特征数：{X.shape[1]}")
    model = LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
        class_weight="balanced",
    )
    model.fit(X, y, sample_weight=sample_weight)
    model.global_target_rates_ = global_target_rates
    return model, label_df, global_counts


def _apply_postprocess(
    uid: str,
    g: pd.DataFrame,
    label_df: pd.DataFrame,
    scores: np.ndarray,
    top_k: int,
) -> list[int]:
    g = g.sort_values(TIME_COL)
    user_actions = set(g["action_typ"].astype(str).tolist())
    user_prods = set(g["prod_typ"].astype(str).tolist())
    has_nonwealth = "非财富" in user_prods or bool(user_actions & {"参与活动", "线下接触"})
    has_offline = "线下接触" in user_actions
    adjusted = scores.copy()
    nonwealth_mask = label_df["prod_typ"].astype(str).eq("非财富").to_numpy()
    offline_mask = label_df["action_typ"].astype(str).eq("线下接触").to_numpy()
    browse_mask = label_df["action_typ"].astype(str).eq("浏览详情").to_numpy()
    if not has_nonwealth:
        adjusted[nonwealth_mask] -= 0.08
    if not has_offline:
        adjusted[offline_mask] -= 0.12

    order = np.argsort(-adjusted)
    selected: list[int] = []
    nonwealth_count = 0
    offline_count = 0
    browse_count = 0
    for idx in order:
        is_nonwealth = bool(nonwealth_mask[idx])
        is_offline = bool(offline_mask[idx])
        is_browse = bool(browse_mask[idx])
        if is_nonwealth:
            limit = 2 if has_nonwealth else 1
            if nonwealth_count >= limit:
                continue
            if not has_nonwealth and len(selected) < 10:
                continue
        if is_offline:
            limit = 1 if not has_offline else 2
            if offline_count >= limit:
                continue
            if not has_offline and len(selected) < 12:
                continue
        if is_browse and browse_count >= 12:
            continue
        selected.append(int(idx))
        nonwealth_count += int(is_nonwealth)
        offline_count += int(is_offline)
        browse_count += int(is_browse)
        if len(selected) >= top_k:
            return selected
    for idx in order:
        if int(idx) not in selected:
            selected.append(int(idx))
        if len(selected) >= top_k:
            break
    return selected[:top_k]


def predict_full_label(
    model: LGBMClassifier,
    label_df: pd.DataFrame,
    global_counts: dict[str, int],
    test_df: pd.DataFrame,
    top_k: int = 20,
    postprocess: bool = True,
) -> pd.DataFrame:
    rows = []
    user_count = test_df[USER_COL].nunique()
    for n, (uid, g) in enumerate(test_df.groupby(USER_COL, sort=False), 1):
        feats = _features_for_user_labels(
            _user_stats(g, HISTORY_CUTOFF),
            label_df,
            global_counts,
            getattr(model, "global_target_rates_", {}),
        )
        scores = model.predict_proba(feats)[:, 1]
        order = _apply_postprocess(str(uid), g, label_df, scores, top_k) if postprocess else np.argsort(-scores)[:top_k].tolist()
        for idx, label_idx in enumerate(order, 1):
            row = label_df.iloc[int(label_idx)]
            rows.append({
                USER_COL: uid,
                "action_typ": row["action_typ"],
                "prod_typ": row["prod_typ"],
                "prod_sub_typ": row["prod_sub_typ"],
                "rsk_lvl": row["rsk_lvl"],
                "index": idx,
            })
        if n % 10000 == 0:
            print(f"full-label 已预测 {n}/{user_count} 用户")
    return pd.DataFrame(rows)[SUBMISSION_COLUMNS]


def run_task1_full_label_classifier(
    train_path: Path,
    test_path: Path,
    output_name: str = "prediction.csv",
    max_train_users: int = 30000,
    neg_per_user: int = 120,
    seed: int = 42,
    variant: str = "v2",
    multi_window: bool = False,
    n_jobs: int = -1,
    use_target_prior: bool = True,
    use_global_hot_negatives: bool = True,
) -> Path:
    train_df = _read_behavior(train_path, is_train=True, include_target_day=True)
    test_df = _read_behavior(test_path, is_train=False)
    model, label_df, global_counts = train_full_label_classifier(
        train_df,
        test_df,
        max_train_users,
        neg_per_user,
        seed,
        variant,
        multi_window,
        n_jobs,
        use_target_prior,
        use_global_hot_negatives,
    )
    pred_df = predict_full_label(model, label_df, global_counts, test_df, top_k=20, postprocess=(variant != "v1"))
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMIT_DIR / output_name
    pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ full-label 提交文件已保存：{out_path}")
    print(f"诊断格式报告：{write_prediction_format_report(pred_df, SUBMIT_DIR, prefix='prediction')}")
    print(f"诊断分布报告：{write_prediction_distribution(pred_df, SUBMIT_DIR, prefix='prediction')}")
    return out_path
