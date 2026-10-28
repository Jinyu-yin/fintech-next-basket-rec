"""
客户次日行为预测：召回 + Learning-to-Rank 精简方案
=================================================

核心思路
--------
1. 把需要预测的四个字段拼成一个候选行为四元组：
   (action_typ, prod_typ, prod_sub_typ, rsk_lvl)。NDCG@20 只认四字段全匹配，
   因此建模目标不是分别预测四列，而是直接给“四元组候选”排序。

2. 用历史窗口构造 next-day 训练样本：
   - 用 3/1~cutoff 的历史生成每个用户的特征。
   - 用 cutoff 后一天的真实行为作为正样本。
   - 对每个用户召回一批候选四元组，命中次日真实四元组标为 1，否则为 0。
   - group 为每个用户的候选数量，训练排序模型。

3. 候选召回优先覆盖高概率行为：
   - 用户自己最近 1/3/7/全量出现过的四元组。
   - 用户最近 action/prod/sub/rsk 的组合。
   - 全局热门四元组。
   - 4/1 热门四元组，利用训练集比测试集多一天的特殊信息。

4. 排序特征强调“重复性 + 近因效应”：
   - 候选四元组在用户历史中的总次数、近 3 天次数、近 7 天次数、时间衰减次数。
   - action/prod/sub/rsk 是否匹配用户最近一次、近 1 天众数、近 3 天众数。
   - 用户近 1 天活跃次数、近 7 天活跃占比、最近行为距 cutoff 小时数。
   - 风险等级均值/波动，以及候选风险与用户历史风险均值的距离。

5. 最终提交：
   对测试集每个用户基于 3/1~3/31 历史构造特征，召回候选，模型排序，输出 top20。
   不足 20 个时用 4/1 热门和全局热门补齐。

依赖
----
推荐安装 lightgbm：
    pip install lightgbm pandas numpy scikit-learn

如果没有 lightgbm，代码会自动退化为 sklearn HistGradientBoostingClassifier，
但排序效果通常不如 LambdaRank。

运行示例
--------
    python task1/solution_ranker.py --train train_dataset.csv --test test_dataset.csv --out prediction.csv
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMRanker
except Exception:  # pragma: no cover
    LGBMRanker = None
    from sklearn.ensemble import HistGradientBoostingClassifier


USER_COL = "user_id"
# 平台字段可能写成 act_tm，也可能写成 acs_tm；读入时会统一重命名为 act_tm。
TIME_COL = "act_tm"
TIME_COL_CANDIDATES = ["act_tm", "acs_tm", "access_tm", "time", "datetime"]
TARGET_COLS = ["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]
SUBMISSION_COLUMNS = ["user_id", "action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl", "index"]

TASK_START = "2025-03-01"
HISTORY_END = "2025-03-31 23:59:59"
TARGET_START = "2025-04-01"
TARGET_END = "2025-04-02"
RANDOM_STATE = 2026
TOP_K = 20


DEFAULT_QUADS = [
    ("浏览详情", "理财", "现金", "R1"),
    ("浏览详情", "基金", "货币", "R1"),
    ("浏览持仓", "理财", "现金", "R1"),
    ("购买", "理财", "现金", "R1"),
    ("浏览详情", "存款", "一般性", "R1"),
]


class RankModel:
    """保存模型及候选补齐所需的热门四元组。"""

    def __init__(self, model, feature_cols: list[str], global_top: list[tuple], target_day_top: list[tuple]):
        self.model = model
        self.feature_cols = feature_cols
        self.global_top = global_top
        self.target_day_top = target_day_top

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_cols]
        if LGBMRanker is not None:
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]


def quad_to_str(q: tuple[str, str, str, str]) -> str:
    return "|".join(q)


def str_to_quad(s: str) -> tuple[str, str, str, str]:
    a, b, c, d = s.split("|", 3)
    return a, b, c, d


def safe_mode(values: Iterable[str], default: str = "") -> str:
    values = [v for v in values if pd.notna(v)]
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """兼容平台可能出现的字段名差异。"""
    df = df.copy()
    time_source = next((col for col in TIME_COL_CANDIDATES if col in df.columns), None)
    if time_source is None:
        raise KeyError(f"找不到时间列，候选列名：{TIME_COL_CANDIDATES}，当前列：{df.columns.tolist()}")
    if time_source != TIME_COL:
        df = df.rename(columns={time_source: TIME_COL})
    return df


def read_events(path: Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[USER_COL, TIME_COL]).copy()
    for col in TARGET_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("未知").astype(str)
    if start is not None:
        df = df[df[TIME_COL] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df[TIME_COL] < pd.Timestamp(end)]
    df = df.sort_values([USER_COL, TIME_COL]).reset_index(drop=True)
    return df


def add_quad_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quad"] = df[TARGET_COLS].astype(str).agg("|".join, axis=1)
    return df


def top_quads(df: pd.DataFrame, n: int = 200) -> list[tuple[str, str, str, str]]:
    if df.empty:
        return DEFAULT_QUADS[:]
    q = add_quad_col(df)["quad"].value_counts().head(n).index.tolist()
    out = [str_to_quad(x) for x in q]
    for item in DEFAULT_QUADS:
        if item not in out:
            out.append(item)
    return out


def weighted_quad_counts(group: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float]:
    hours = (cutoff - group[TIME_COL]).dt.total_seconds().clip(lower=0) / 3600
    # 7 天半衰期。越近权重越高。
    weights = np.exp(-hours / (24 * 7))
    result: dict[str, float] = defaultdict(float)
    for quad, w in zip(group["quad"], weights):
        result[quad] += float(w)
    return dict(result)


def build_user_features(history: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """把 cutoff 之前的历史压成 user-level 特征。"""
    hist = add_quad_col(history[history[TIME_COL] <= cutoff].copy())
    rows = []

    for uid, g in hist.groupby(USER_COL, sort=False):
        g = g.sort_values(TIME_COL)
        last = g.iloc[-1]
        last_1d = g[g[TIME_COL] > cutoff - pd.Timedelta(days=1)]
        last_3d = g[g[TIME_COL] > cutoff - pd.Timedelta(days=3)]
        last_7d = g[g[TIME_COL] > cutoff - pd.Timedelta(days=7)]

        quad_all = Counter(g["quad"])
        quad_3d = Counter(last_3d["quad"])
        quad_7d = Counter(last_7d["quad"])
        action_prod = Counter(zip(g["action_typ"], g["prod_typ"]))
        prod_sub = Counter(zip(g["prod_typ"], g["prod_sub_typ"]))
        prod_rsk = Counter(zip(g["prod_typ"], g["rsk_lvl"]))

        rsk_num = g["rsk_lvl"].str.extract(r"R(\d)")[0].astype(float)
        rows.append(
            {
                USER_COL: uid,
                "last_quad": last["quad"],
                "last_action": last["action_typ"],
                "last_prod": last["prod_typ"],
                "last_sub": last["prod_sub_typ"],
                "last_rsk": last["rsk_lvl"],
                "mode_1d_action": safe_mode(last_1d["action_typ"], last["action_typ"]),
                "mode_1d_prod": safe_mode(last_1d["prod_typ"], last["prod_typ"]),
                "mode_3d_action": safe_mode(last_3d["action_typ"], last["action_typ"]),
                "mode_3d_prod": safe_mode(last_3d["prod_typ"], last["prod_typ"]),
                "cnt_all": len(g),
                "cnt_1d": len(last_1d),
                "cnt_3d": len(last_3d),
                "cnt_7d": len(last_7d),
                "last_week_ratio": len(last_7d) / max(len(g), 1),
                "hours_since_last": max((cutoff - last[TIME_COL]).total_seconds() / 3600, 0),
                "quad_all": dict(quad_all),
                "quad_3d": dict(quad_3d),
                "quad_7d": dict(quad_7d),
                "quad_decay": weighted_quad_counts(g, cutoff),
                "action_prod": dict(action_prod),
                "prod_sub": dict(prod_sub),
                "prod_rsk": dict(prod_rsk),
                "rsk_mean": float(rsk_num.mean()) if len(rsk_num) else 1.0,
                "rsk_std": float(rsk_num.std(ddof=0)) if len(rsk_num) else 0.0,
                "unique_quad_7d": int(last_7d["quad"].nunique()),
                "unique_action_7d": int(last_7d["action_typ"].nunique()),
                "unique_prod_7d": int(last_7d["prod_typ"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def recall_candidates(user_row: pd.Series, global_top: list[tuple], target_day_top: list[tuple], max_candidates: int = 80) -> list[tuple]:
    """每个用户召回候选四元组，先个性化，后热门补齐。"""
    candidates: list[tuple] = []
    seen = set()

    def add(q):
        if isinstance(q, str):
            q = str_to_quad(q)
        if q not in seen:
            seen.add(q)
            candidates.append(q)

    add(user_row["last_quad"])

    for key in ["quad_3d", "quad_7d", "quad_all", "quad_decay"]:
        items = sorted(user_row[key].items(), key=lambda kv: kv[1], reverse=True)
        for quad_str, _ in items[:30]:
            add(quad_str)

    # 最近 action/prod/rsk 与热门细类做轻量交叉，增加命中未完整重复的机会。
    recent_actions = [user_row["last_action"], user_row["mode_1d_action"], user_row["mode_3d_action"]]
    recent_prods = [user_row["last_prod"], user_row["mode_1d_prod"], user_row["mode_3d_prod"]]
    recent_rsk = [user_row["last_rsk"]]
    for act in recent_actions:
        for prod in recent_prods:
            for q in global_top[:60]:
                if q[0] == act and q[1] == prod:
                    add(q)
            for q in target_day_top[:60]:
                if q[0] == act and q[1] == prod:
                    add(q)
        for q in global_top[:80]:
            if q[0] == act and q[3] in recent_rsk:
                add(q)

    for q in target_day_top[:80] + global_top[:80] + DEFAULT_QUADS:
        add(q)
        if len(candidates) >= max_candidates:
            break
    return candidates[:max_candidates]


def candidate_features(user_row: pd.Series, q: tuple[str, str, str, str]) -> dict:
    q_str = quad_to_str(q)
    act, prod, sub, rsk = q
    rsk_num = float(rsk[1]) if len(rsk) >= 2 and rsk[1].isdigit() else user_row["rsk_mean"]
    return {
        "quad_count_all": user_row["quad_all"].get(q_str, 0),
        "quad_count_3d": user_row["quad_3d"].get(q_str, 0),
        "quad_count_7d": user_row["quad_7d"].get(q_str, 0),
        "quad_decay_count": user_row["quad_decay"].get(q_str, 0.0),
        "action_prod_count": user_row["action_prod"].get((act, prod), 0),
        "prod_sub_count": user_row["prod_sub"].get((prod, sub), 0),
        "prod_rsk_count": user_row["prod_rsk"].get((prod, rsk), 0),
        "match_last_action": int(act == user_row["last_action"]),
        "match_last_prod": int(prod == user_row["last_prod"]),
        "match_last_sub": int(sub == user_row["last_sub"]),
        "match_last_rsk": int(rsk == user_row["last_rsk"]),
        "match_mode_1d_action": int(act == user_row["mode_1d_action"]),
        "match_mode_1d_prod": int(prod == user_row["mode_1d_prod"]),
        "match_mode_3d_action": int(act == user_row["mode_3d_action"]),
        "match_mode_3d_prod": int(prod == user_row["mode_3d_prod"]),
        "cnt_all": user_row["cnt_all"],
        "cnt_1d": user_row["cnt_1d"],
        "cnt_3d": user_row["cnt_3d"],
        "cnt_7d": user_row["cnt_7d"],
        "last_week_ratio": user_row["last_week_ratio"],
        "hours_since_last": user_row["hours_since_last"],
        "rsk_mean": user_row["rsk_mean"],
        "rsk_std": user_row["rsk_std"],
        "rsk_diff": abs(rsk_num - user_row["rsk_mean"]),
        "unique_quad_7d": user_row["unique_quad_7d"],
        "unique_action_7d": user_row["unique_action_7d"],
        "unique_prod_7d": user_row["unique_prod_7d"],
    }


def build_rank_samples(
    train_df: pd.DataFrame,
    cutoff: str,
    target_day: str,
    global_top: list[tuple],
    target_day_top: list[tuple],
    max_users: int | None = 30000,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    cutoff_ts = pd.Timestamp(cutoff)
    target_ts = pd.Timestamp(target_day)

    features = build_user_features(train_df, cutoff_ts)
    target = add_quad_col(train_df[(train_df[TIME_COL] >= target_ts) & (train_df[TIME_COL] < target_ts + pd.Timedelta(days=1))])
    target_map = target.groupby(USER_COL)["quad"].apply(lambda x: set(x)).to_dict()

    if max_users is not None and len(features) > max_users:
        active_users = set(target[USER_COL].unique())
        active = features[features[USER_COL].isin(active_users)]
        inactive = features[~features[USER_COL].isin(active_users)]
        need = max(0, max_users - len(active))
        inactive = inactive.sample(min(need, len(inactive)), random_state=RANDOM_STATE)
        features = pd.concat([active, inactive], ignore_index=True)

    rows, labels, groups = [], [], []
    for _, row in features.iterrows():
        uid = row[USER_COL]
        positives = target_map.get(uid, set())
        candidates = recall_candidates(row, global_top, target_day_top)
        if not candidates:
            continue
        for q in candidates:
            rows.append(candidate_features(row, q))
            labels.append(int(quad_to_str(q) in positives))
        groups.append(len(candidates))

    return pd.DataFrame(rows), np.array(labels, dtype=np.int8), groups


def train_model(train_df: pd.DataFrame) -> RankModel:
    global_top = top_quads(train_df, 300)
    target_day_df = train_df[(train_df[TIME_COL] >= pd.Timestamp(TARGET_START)) & (train_df[TIME_COL] < pd.Timestamp(TARGET_END))]
    target_day_top = top_quads(target_day_df, 300) if not target_day_df.empty else global_top[:]

    windows = [
        ("2025-03-24 23:59:59", "2025-03-25"),
        ("2025-03-27 23:59:59", "2025-03-28"),
        ("2025-03-29 23:59:59", "2025-03-30"),
        ("2025-03-31 23:59:59", "2025-04-01"),
    ]

    X_parts, y_parts, groups = [], [], []
    for cutoff, target_day in windows:
        print(f"构造训练窗口：{cutoff} -> {target_day}")
        X, y, g = build_rank_samples(train_df, cutoff, target_day, global_top, target_day_top)
        if len(X):
            X_parts.append(X)
            y_parts.append(y)
            groups.extend(g)
            print(f"  samples={len(X)}, positives={int(y.sum())}, groups={len(g)}")

    X_train = pd.concat(X_parts, ignore_index=True)
    y_train = np.concatenate(y_parts)
    feature_cols = X_train.columns.tolist()

    if LGBMRanker is not None:
        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[20],
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train[feature_cols], y_train, group=groups)
    else:
        print("未检测到 lightgbm，使用 sklearn 分类模型兜底。建议安装 lightgbm 提升 NDCG。")
        model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, random_state=RANDOM_STATE)
        model.fit(X_train[feature_cols], y_train)

    return RankModel(model, feature_cols, global_top, target_day_top)


def predict_for_user(rank_model: RankModel, user_row: pd.Series) -> list[tuple[str, str, str, str]]:
    candidates = recall_candidates(user_row, rank_model.global_top, rank_model.target_day_top, max_candidates=100)
    X = pd.DataFrame([candidate_features(user_row, q) for q in candidates])
    scores = rank_model.predict_score(X)
    ranked = [q for q, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)]

    seen = set(ranked)
    for q in rank_model.target_day_top + rank_model.global_top + DEFAULT_QUADS:
        if q not in seen:
            ranked.append(q)
            seen.add(q)
        if len(ranked) >= TOP_K:
            break
    return ranked[:TOP_K]


def make_submission(rank_model: RankModel, test_df: pd.DataFrame, out_path: Path) -> Path:
    cutoff = pd.Timestamp(HISTORY_END)
    user_features = build_user_features(test_df, cutoff)
    test_users = test_df[USER_COL].drop_duplicates().tolist()
    feature_map = {row[USER_COL]: row for _, row in user_features.iterrows()}

    rows = []
    fallback = (rank_model.target_day_top + rank_model.global_top + DEFAULT_QUADS)[:TOP_K]
    for i, uid in enumerate(test_users, 1):
        row = feature_map.get(uid)
        top20 = fallback if row is None else predict_for_user(rank_model, row)
        for rank, (act, prod, sub, rsk) in enumerate(top20, 1):
            rows.append(
                {
                    "user_id": uid,
                    "action_typ": act,
                    "prod_typ": prod,
                    "prod_sub_typ": sub,
                    "rsk_lvl": rsk,
                    "index": rank,
                }
            )
        if i % 10000 == 0:
            print(f"已预测 {i}/{len(test_users)} 个用户")

    sub = pd.DataFrame(rows)[SUBMISSION_COLUMNS]
    sub.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"保存提交文件：{out_path}")
    print(f"行数={len(sub)}，用户数={sub['user_id'].nunique()}，每用户20条={sub.groupby('user_id').size().eq(20).all()}")
    return out_path


def ndcg_at_20(pred: list[str], true_set: set[str]) -> float:
    if not true_set:
        return 0.0
    dcg = 0.0
    for i, q in enumerate(pred[:20], 1):
        if q in true_set:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(true_set), 20)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True, help="训练集 CSV")
    parser.add_argument("--test", type=Path, required=True, help="测试集 CSV")
    parser.add_argument("--out", type=Path, default=Path("prediction.csv"), help="输出 prediction.csv 路径")
    args = parser.parse_args()

    train_df = read_events(args.train, TASK_START, TARGET_END)
    test_df = read_events(args.test, TASK_START, TARGET_END)
    print(f"训练记录：{len(train_df)}，测试记录：{len(test_df)}")

    rank_model = train_model(train_df)
    make_submission(rank_model, test_df, args.out)


if __name__ == "__main__":
    main()
