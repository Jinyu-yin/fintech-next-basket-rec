from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .config import SUBMIT_DIR, TIME_COL, USER_COL
from .diagnostic_outputs import write_prediction_distribution, write_prediction_format_report
from .fast_light_cross import COLS, SUBMISSION_COLUMNS, _read_behavior

TRANSITIONS = {
    "浏览详情": ["购买", "收藏", "浏览持仓"],
    "收藏": ["购买", "浏览详情"],
    "购买": ["浏览持仓", "浏览详情"],
    "浏览持仓": ["赎回", "购买", "浏览详情"],
    "赎回": ["浏览详情", "购买"],
    "线下接触": ["浏览详情", "购买"],
    "参与活动": ["浏览详情", "购买"],
}

DEFAULT_NONWEALTH_SUB = "其他"
DEFAULT_NONWEALTH_RISK = "R2"


def add_quad_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quad_key"] = df["action_typ"].astype(str).str.cat(
        [df["prod_typ"].astype(str), df["prod_sub_typ"].astype(str), df["rsk_lvl"].astype(str)], sep="|"
    )
    return df


def split_quad(q: str) -> tuple[str, str, str, str]:
    parts = q.split("|")
    if len(parts) != 4:
        return ("浏览详情", "理财", "短债", "R2")
    return parts[0], parts[1], parts[2], parts[3]


def build_global_popular(*dfs: pd.DataFrame) -> list[str]:
    counts: Counter[str] = Counter()
    for df in dfs:
        if not df.empty:
            counts.update(add_quad_key(df)["quad_key"].tolist())
    return [q for q, _ in counts.most_common()]


def _score_user_history(g: pd.DataFrame, global_popular: list[str], top_k: int = 20) -> list[str]:
    g = add_quad_key(g.sort_values(TIME_COL))
    max_time = g[TIME_COL].max()
    days = (max_time - g[TIME_COL]).dt.total_seconds().clip(lower=0) / 86400
    score: defaultdict[str, float] = defaultdict(float)

    for q, d in zip(g["quad_key"].tolist(), days.tolist(), strict=False):
        weight = 1.0
        if d <= 1:
            weight += 8.0
        if d <= 3:
            weight += 5.0
        if d <= 7:
            weight += 3.0
        if d <= 14:
            weight += 1.5
        score[q] += weight

    recent = g.tail(10)
    for rank, q in enumerate(reversed(recent["quad_key"].tolist()), 1):
        score[q] += 10.0 / rank

    # Transition candidates: keep product/sub/risk, vary action according to common funnels.
    for recency_rank, row in enumerate(reversed(list(recent.itertuples(index=False))), 1):
        act = str(row.action_typ)
        prod = str(row.prod_typ)
        sub = str(row.prod_sub_typ)
        rsk = str(row.rsk_lvl)
        for next_act in TRANSITIONS.get(act, []):
            if prod == "非财富":
                cand = f"{next_act}|非财富|{DEFAULT_NONWEALTH_SUB}|{DEFAULT_NONWEALTH_RISK}"
            else:
                cand = f"{next_act}|{prod}|{sub}|{rsk}"
            score[cand] += 6.0 / recency_rank

    # Product-sub fallback with common actions.
    top_prod_sub = g[["prod_typ", "prod_sub_typ", "rsk_lvl"]].value_counts().head(8)
    for (prod, sub, rsk), cnt in top_prod_sub.items():
        for act, boost in [("浏览详情", 3.0), ("购买", 2.2), ("浏览持仓", 1.8), ("赎回", 1.0)]:
            score[f"{act}|{prod}|{sub}|{rsk}"] += float(cnt) * boost

    selected: list[str] = []
    selected_set = set()
    for q, _ in sorted(score.items(), key=lambda x: x[1], reverse=True):
        if q in selected_set:
            continue
        selected.append(q)
        selected_set.add(q)
        if len(selected) >= top_k:
            return selected
    for q in global_popular:
        if q not in selected_set:
            selected.append(q)
            selected_set.add(q)
        if len(selected) >= top_k:
            break
    return selected[:top_k]


def predict_recent_repeat(test_df: pd.DataFrame, global_popular: list[str], top_k: int = 20) -> pd.DataFrame:
    rows = []
    user_count = test_df[USER_COL].nunique()
    for n, (uid, g) in enumerate(test_df.groupby(USER_COL, sort=False), 1):
        quads = _score_user_history(g, global_popular, top_k=top_k)
        for idx, q in enumerate(quads, 1):
            action, prod, sub, rsk = split_quad(q)
            rows.append(
                {
                    USER_COL: uid,
                    "action_typ": action,
                    "prod_typ": prod,
                    "prod_sub_typ": sub,
                    "rsk_lvl": rsk,
                    "index": idx,
                }
            )
        if n % 10000 == 0:
            print(f"recent-repeat 已预测 {n}/{user_count} 用户")
    return pd.DataFrame(rows)[SUBMISSION_COLUMNS]


def run_recent_repeat_model(train_path: Path, test_path: Path, output_name: str = "prediction_recent.csv") -> Path:
    train_df = _read_behavior(train_path, is_train=True, include_target_day=True)
    test_df = _read_behavior(test_path, is_train=False)
    global_popular = build_global_popular(train_df, test_df)
    pred = predict_recent_repeat(test_df, global_popular, top_k=20)
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMIT_DIR / output_name
    pred.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"recent-repeat 提交文件已保存：{out}")
    print(f"格式报告：{write_prediction_format_report(pred, SUBMIT_DIR, prefix=out.stem)}")
    print(f"分布报告：{write_prediction_distribution(pred, SUBMIT_DIR, prefix=out.stem)}")
    return out


if __name__ == "__main__":
    run_recent_repeat_model(Path("/work/data/task1/dataset_train.csv"), Path("/work/data/task1/dataset_test.csv"))
