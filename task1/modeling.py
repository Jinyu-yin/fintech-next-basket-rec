from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRanker, early_stopping

from .config import RANDOM_STATE
from .feature_engineering import PROD_RSK_VALID, ACTION_PROD_VALID, PROD_SUB_MAP


# NBR/FPMC-inspired 轻量混合排序参数：短期重复 + 长期偏好 + 行为转移 + 全局热度 + LGBM。
# 这些不是论文代码复刻，而是把 next-basket / repeat-aware / Markov transition 思想落成可解释特征分。
HYBRID_WEIGHTS = {
    "lgbm": 0.25,
    "repeat_1d": 18.0,
    "repeat_3d": 10.0,
    "repeat_7d": 6.0,
    "user_weighted": 4.0,
    "user_all": 1.5,
    "field_pref": 3.0,
    "transition": 5.0,
    "risk": 2.0,
    "april1_hot": 2.0,
}

ACTION_TRANSITIONS = {
    "浏览详情": [("浏览详情", 1.0), ("收藏", 0.75), ("购买", 0.45)],
    "收藏": [("浏览详情", 0.65), ("收藏", 1.0), ("购买", 0.8)],
    "购买": [("浏览详情", 0.45), ("浏览持仓", 1.0), ("购买", 0.6)],
    "浏览持仓": [("浏览持仓", 1.0), ("赎回", 0.75), ("购买", 0.35)],
    "赎回": [("浏览持仓", 0.8), ("赎回", 0.7), ("浏览详情", 0.35)],
    "参与活动": [("参与活动", 1.0), ("线下接触", 0.45)],
    "线下接触": [("线下接触", 1.0), ("参与活动", 0.5)],
}


FEATURE_COLUMNS = [
    "quad_count_all",
    "quad_count_3d",
    "quad_count_7d",
    "quad_weighted_count",
    "quad_april1_count",
    "act_prod_count",
    "prod_sub_count",
    "rsk_count",
    "recent_act_switch",
    "recent_prod_switch",
    "recent_sub_switch",
    "recent_rsk_switch",
    "act_unique_7d",
    "prod_unique_7d",
    "sub_unique_7d",
    "act_match_last_1d",
    "prod_match_last_1d",
    "sub_match_last_1d",
    "act_match_last_1d_mode",
    "prod_match_last_1d_mode",
    "act_match_last_3d_mode",
    "prod_match_last_3d_mode",
    "last_act_hours_ago",
    "cnt_last_1d",
    "last_week_ratio",
    "rsk_mean",
    "rsk_std",
    "rsk_diff",
    "action_typ",
    "prod_typ",
]


@dataclass
class NextDayRankModel:
    ranker: LGBMRanker
    cat_encoders: dict[str, LabelEncoder]
    global_top_quads: list[tuple]
    april1_top_quads: list[tuple]
    quad_april1_count: dict[str, int]
    feature_columns: list[str]


def align_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """确保训练和预测使用完全一致的特征列。"""
    out = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0
    return out[FEATURE_COLUMNS]


def train_next_day_ranker(
    X_train,
    y_train,
    groups_train,
    X_val,
    y_val,
    groups_val,
    sample_weight=None,
    val_sample_weight=None,
):
    """训练次日排序模型。"""
    X_train = align_feature_frame(X_train)
    X_val = align_feature_frame(X_val)
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([X_train[col].astype(str), X_val[col].astype(str)], ignore_index=True))
        X_train[col] = le.transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        encoders[col] = le

    model = LGBMRanker(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=0.8,
        min_child_samples=25,
        objective="lambdarank",
        metric="ndcg",
        random_state=RANDOM_STATE,
        verbosity=-1,
    )

    callbacks = [early_stopping(stopping_rounds=50, verbose=False)]
    fit_kwargs = dict(
        X=X_train,
        y=y_train,
        group=groups_train,
        eval_set=[(X_val, y_val)],
        eval_group=[groups_val],
        categorical_feature=cat_cols,
        callbacks=callbacks,
    )
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    if val_sample_weight is not None:
        fit_kwargs["eval_sample_weight"] = [val_sample_weight]
    model.fit(**fit_kwargs)

    return NextDayRankModel(
        ranker=model,
        cat_encoders=encoders,
        global_top_quads=[],
        april1_top_quads=[],
        quad_april1_count={},
        feature_columns=FEATURE_COLUMNS,
    )


def _is_valid_quad(quad: tuple) -> bool:
    if len(quad) != 4:
        return False
    act, prod, sub, rsk = quad
    return (
        prod in PROD_SUB_MAP
        and sub in PROD_SUB_MAP[prod]
        and rsk in PROD_RSK_VALID.get(prod, [])
        and prod in ACTION_PROD_VALID.get(act, [])
    )


def _unique_extend(candidates: list[tuple], new_items: list[tuple], limit: int) -> None:
    for item in new_items:
        if len(candidates) >= limit:
            break
        if item not in candidates and _is_valid_quad(item):
            candidates.append(item)


def _quads_from_count_dict(count_dict: dict, top_n: int) -> list[tuple]:
    if not isinstance(count_dict, dict) or not count_dict:
        return []
    items = sorted(count_dict.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [tuple(str(k).split("|")) for k, _ in items]


def _transition_candidates(user_feat) -> list[tuple]:
    """基于最近行为构造 Markov-style 行为转移候选。"""
    seeds = [
        (user_feat.get("last_1d_act"), user_feat.get("last_1d_prod"), user_feat.get("last_1d_sub"), user_feat.get("last_1d_rsk")),
        (user_feat.get("last_3d_act_mode"), user_feat.get("last_3d_prod_mode"), user_feat.get("last_3d_sub_mode"), user_feat.get("wgt_rsk")),
        (user_feat.get("last_7d_act_mode"), user_feat.get("last_7d_prod_mode"), user_feat.get("last_7d_sub_mode"), user_feat.get("wgt_rsk")),
    ]
    out = []
    for act, prod, sub, rsk in seeds:
        if not act or not prod or not sub or not rsk:
            continue
        for next_act, _weight in ACTION_TRANSITIONS.get(act, [(act, 1.0)]):
            out.append((next_act, prod, sub, rsk))
    return out


def _factorized_candidates(user_feat) -> list[tuple]:
    """把 action/product/sub/risk 偏好拆开后重新组合，缓解四元组稀疏。"""
    actions = [user_feat.get("last_1d_act"), user_feat.get("last_1d_act_mode"), user_feat.get("last_3d_act_mode"), user_feat.get("wgt_act")]
    products = [user_feat.get("last_1d_prod"), user_feat.get("last_1d_prod_mode"), user_feat.get("last_3d_prod_mode"), user_feat.get("wgt_prod")]
    subs = [user_feat.get("last_1d_sub"), user_feat.get("last_3d_sub_mode"), user_feat.get("last_7d_sub_mode"), user_feat.get("wgt_sub")]
    risks = [user_feat.get("last_1d_rsk"), user_feat.get("wgt_rsk")]

    out = []
    for prod in dict.fromkeys([p for p in products if p]):
        valid_subs = [s for s in dict.fromkeys([s for s in subs if s]) if s in PROD_SUB_MAP.get(prod, [])]
        if not valid_subs:
            valid_subs = PROD_SUB_MAP.get(prod, [])[:2]
        valid_rsks = [r for r in dict.fromkeys([r for r in risks if r]) if r in PROD_RSK_VALID.get(prod, [])]
        if not valid_rsks:
            valid_rsks = PROD_RSK_VALID.get(prod, [])[:1]
        valid_actions = [a for a in dict.fromkeys([a for a in actions if a]) if prod in ACTION_PROD_VALID.get(a, [])]
        if not valid_actions:
            valid_actions = [a for a, prods in ACTION_PROD_VALID.items() if prod in prods][:2]
        for act in valid_actions[:3]:
            for sub in valid_subs[:3]:
                for rsk in valid_rsks[:2]:
                    out.append((act, prod, sub, rsk))
    return out


def generate_candidates(user_feat, global_top_quads, april1_top_quads) -> list[tuple]:
    """NBR-inspired 分层召回：历史重复、行为转移、字段拆分重组、4/1热门、全局热门。"""
    candidates = []
    _unique_extend(candidates, _quads_from_count_dict(user_feat.get("quad_count_3d", {}), 30), 40)
    _unique_extend(candidates, _quads_from_count_dict(user_feat.get("quad_count_7d", {}), 30), 60)
    _unique_extend(candidates, list(user_feat.get("top_quads", [])), 80)
    _unique_extend(candidates, _transition_candidates(user_feat), 110)
    _unique_extend(candidates, _factorized_candidates(user_feat), 140)
    _unique_extend(candidates, list(april1_top_quads), 180)
    _unique_extend(candidates, list(global_top_quads), 220)
    return candidates[:220]


def _candidate_feature_row(model: NextDayRankModel, user_feat, quad: tuple[str, str, str, str]) -> dict:
    act, prod, sub, rsk = quad
    quad_str = "|".join(quad)
    return {
        "quad_count_all": user_feat["quad_count_all"].get(quad_str, 0),
        "quad_count_3d": user_feat["quad_count_3d"].get(quad_str, 0),
        "quad_count_7d": user_feat["quad_count_7d"].get(quad_str, 0),
        "quad_weighted_count": user_feat["quad_weighted_count"].get(quad_str, 0),
        "quad_april1_count": model.quad_april1_count.get(quad_str, 0),
        "act_prod_count": user_feat.get("act_prod_count", {}).get((act, prod), 0),
        "prod_sub_count": user_feat.get("prod_sub_count", {}).get((prod, sub), 0),
        "rsk_count": user_feat.get("rsk_count", {}).get((prod, rsk), 0),
        "recent_act_switch": user_feat.get("recent_act_switch", 0),
        "recent_prod_switch": user_feat.get("recent_prod_switch", 0),
        "recent_sub_switch": user_feat.get("recent_sub_switch", 0),
        "recent_rsk_switch": user_feat.get("recent_rsk_switch", 0),
        "act_unique_7d": user_feat.get("act_unique_7d", 0),
        "prod_unique_7d": user_feat.get("prod_unique_7d", 0),
        "sub_unique_7d": user_feat.get("sub_unique_7d", 0),
        "act_match_last_1d": 1 if act == user_feat["last_1d_act"] else 0,
        "prod_match_last_1d": 1 if prod == user_feat["last_1d_prod"] else 0,
        "sub_match_last_1d": 1 if sub == user_feat["last_1d_sub"] else 0,
        "act_match_last_1d_mode": 1 if act == user_feat["last_1d_act_mode"] else 0,
        "prod_match_last_1d_mode": 1 if prod == user_feat["last_1d_prod_mode"] else 0,
        "act_match_last_3d_mode": 1 if act == user_feat["last_3d_act_mode"] else 0,
        "prod_match_last_3d_mode": 1 if prod == user_feat["last_3d_prod_mode"] else 0,
        "last_act_hours_ago": user_feat["last_act_hours_ago"],
        "cnt_last_1d": user_feat["cnt_last_1d"],
        "last_week_ratio": user_feat["last_week_ratio"],
        "rsk_mean": user_feat["rsk_mean"],
        "rsk_std": user_feat.get("rsk_std", 0),
        "rsk_diff": abs(int(rsk[1]) - user_feat["rsk_mean"]),
        "action_typ": act,
        "prod_typ": prod,
    }


def _safe_label_transform(le: LabelEncoder, values: pd.Series) -> np.ndarray:
    """安全编码预测阶段类别，未见过的类别映射到第一个已知类别。"""
    str_values = values.astype(str)
    known = set(le.classes_)
    fallback = le.classes_[0]
    safe_values = str_values.where(str_values.isin(known), fallback)
    return le.transform(safe_values)


def rank_candidates(model: NextDayRankModel, user_feat, candidates) -> list[tuple]:
    """对候选排序。"""
    if not candidates:
        return model.april1_top_quads[:20]

    X = pd.DataFrame([_candidate_feature_row(model, user_feat, quad) for quad in candidates])
    X = align_feature_frame(X)
    for col, le in model.cat_encoders.items():
        if col in X.columns:
            X[col] = _safe_label_transform(le, X[col])

    # 训练时 LightGBM 使用的是带列名的 DataFrame，预测也必须保留列名；
    # 否则 sklearn 会对每个用户反复输出 “X does not have valid feature names” 警告。
    expected_n_features = len(model.feature_columns)
    if X.shape[1] != expected_n_features:
        raise ValueError(f"预测特征列数异常：{X.shape[1]} != {expected_n_features}，列名：{X.columns.tolist()}")

    lgb_scores = model.ranker.predict(X, validate_features=False)
    if len(lgb_scores) > 1 and np.std(lgb_scores) > 1e-12:
        lgb_scores = (lgb_scores - np.mean(lgb_scores)) / np.std(lgb_scores)
    else:
        lgb_scores = np.zeros_like(lgb_scores, dtype=float)

    heuristic_scores = []
    for quad in candidates:
        act, prod, _sub, rsk = quad
        row = _candidate_feature_row(model, user_feat, quad)

        repeat_score = (
            HYBRID_WEIGHTS["repeat_1d"] * (row["act_match_last_1d"] + row["prod_match_last_1d"] + row["sub_match_last_1d"])
            + HYBRID_WEIGHTS["repeat_3d"] * row["quad_count_3d"]
            + HYBRID_WEIGHTS["repeat_7d"] * row["quad_count_7d"]
            + HYBRID_WEIGHTS["user_weighted"] * row["quad_weighted_count"]
            + HYBRID_WEIGHTS["user_all"] * row["quad_count_all"]
        )

        field_score = HYBRID_WEIGHTS["field_pref"] * (
            np.log1p(row["act_prod_count"])
            + np.log1p(row["prod_sub_count"])
            + np.log1p(row["rsk_count"])
            + row["act_match_last_1d_mode"]
            + row["prod_match_last_1d_mode"]
            + row["act_match_last_3d_mode"]
            + row["prod_match_last_3d_mode"]
        )

        transition_score = 0.0
        for next_act, weight in ACTION_TRANSITIONS.get(user_feat.get("last_1d_act"), []):
            if act == next_act and prod == user_feat.get("last_1d_prod"):
                transition_score = max(transition_score, weight)
        transition_score *= HYBRID_WEIGHTS["transition"]

        user_risk = user_feat.get("last_1d_rsk", user_feat.get("wgt_rsk"))
        risk_score = HYBRID_WEIGHTS["risk"] * (1.0 if rsk == user_risk else 0.0) - 0.7 * row["rsk_diff"]
        hot_score = HYBRID_WEIGHTS["april1_hot"] * np.log1p(row["quad_april1_count"])

        heuristic_scores.append(repeat_score + field_score + transition_score + risk_score + hot_score)
    heuristic_scores = np.array(heuristic_scores, dtype=float)
    if len(heuristic_scores) > 1 and np.std(heuristic_scores) > 1e-12:
        heuristic_scores = (heuristic_scores - np.mean(heuristic_scores)) / np.std(heuristic_scores)

    scores = HYBRID_WEIGHTS["lgbm"] * lgb_scores + (1.0 - HYBRID_WEIGHTS["lgbm"]) * heuristic_scores
    ranked_candidates = [candidates[i] for i in np.argsort(scores)[::-1]]
    final_top20 = []
    seen = set()
    for q in ranked_candidates:
        if q in seen:
            continue
        seen.add(q)
        final_top20.append(q)
        if len(final_top20) >= 20:
            break
    return final_top20
