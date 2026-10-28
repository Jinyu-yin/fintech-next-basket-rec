from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import SUBMIT_DIR
from .diagnostic_outputs import write_prediction_distribution, write_prediction_format_report
from .fast_light_cross import COLS, SUBMISSION_COLUMNS, _build_global_stats, _quad_series, _read_behavior

RECENT_WEIGHTS = [(1, 8.0), (3, 5.0), (7, 3.0), (14, 1.5), (999, 0.6)]
ACTION_TRANSITIONS = {
    "浏览详情": [("购买", 2.5), ("收藏", 1.8), ("浏览持仓", 1.5)],
    "收藏": [("购买", 3.0), ("浏览详情", 1.5)],
    "购买": [("浏览持仓", 2.8), ("浏览详情", 1.2)],
    "浏览持仓": [("赎回", 2.2), ("浏览详情", 1.5), ("购买", 1.0)],
    "赎回": [("浏览详情", 1.5), ("购买", 0.8)],
    "参与活动": [],
    "线下接触": [],
}


def _add(scores: dict[tuple[str, str, str, str], float], q: tuple[str, str, str, str], score: float) -> None:
    act, prod, sub, rsk = q
    if prod == "非财富":
        q = (act, prod, "", "")
    elif not sub or not rsk:
        return
    scores[q] = scores.get(q, 0.0) + score


def _user_scores(g: pd.DataFrame, global_hot: list[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str]]:
    g = g.sort_values("acs_tm").copy()
    g["quad"] = _quad_series(g)
    max_time = g["acs_tm"].max()
    days = (max_time - g["acs_tm"]).dt.total_seconds().clip(lower=0) / 86400
    g["days_diff"] = days
    scores: dict[tuple[str, str, str, str], float] = {}

    for limit, weight in RECENT_WEIGHTS:
        part = g[g["days_diff"] <= limit] if limit < 999 else g
        if part.empty:
            continue
        counts = part["quad"].value_counts()
        for q, c in counts.items():
            _add(scores, q, float(c) * weight)

    last_rows = g.tail(10).iloc[::-1]
    for pos, row in enumerate(last_rows.itertuples(index=False), 1):
        q = row.quad
        _add(scores, q, 12.0 / pos)
        act, prod, sub, rsk = q
        if prod != "非财富":
            for next_act, w in ACTION_TRANSITIONS.get(act, []):
                _add(scores, (next_act, prod, sub, rsk), w / pos)

    # Conservative nonwealth: only keep if user has recent nonwealth evidence; cap handled after ranking.
    has_recent_nonwealth = bool((g.tail(20)["prod_typ"].astype(str) == "非财富").any())
    if not has_recent_nonwealth:
        scores = {q: s for q, s in scores.items() if q[1] != "非财富"}

    for rank, q in enumerate(global_hot[:80], 1):
        if q[1] == "非财富":
            continue
        _add(scores, q, 0.15 / rank)

    ranked = [q for q, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    selected = []
    seen = set()
    nonwealth_count = 0
    for q in ranked:
        if q in seen:
            continue
        if q[1] == "非财富":
            if nonwealth_count >= 1 or len(selected) < 10:
                continue
            nonwealth_count += 1
        selected.append(q)
        seen.add(q)
        if len(selected) >= 20:
            break
    for q in global_hot:
        if len(selected) >= 20:
            break
        if q[1] == "非财富" or q in seen:
            continue
        selected.append(q)
        seen.add(q)
    return selected[:20]


def run_recent_conservative(test_path: Path, train_path: Path, output_name: str = "prediction_recent_conservative.csv") -> Path:
    train_df = _read_behavior(train_path, is_train=True, include_target_day=False)
    test_df = _read_behavior(test_path, is_train=False)
    stats = _build_global_stats(train_df, test_df)
    global_hot = stats["hot_quads"]
    rows = []
    users = test_df["user_id"].drop_duplicates().tolist()
    for n, (uid, g) in enumerate(test_df.groupby("user_id", sort=False), 1):
        labels = _user_scores(g, global_hot)
        for idx, q in enumerate(labels, 1):
            rows.append({"user_id": uid, "action_typ": q[0], "prod_typ": q[1], "prod_sub_typ": q[2], "rsk_lvl": q[3], "index": idx})
        if n % 10000 == 0:
            print(f"recent-conservative 已预测 {n}/{len(users)} 用户")
    pred = pd.DataFrame(rows)[SUBMISSION_COLUMNS]
    out = SUBMIT_DIR / output_name
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"recent-conservative 保存：{out}")
    print(write_prediction_format_report(pred, out.parent, prefix=out.stem))
    print(write_prediction_distribution(pred, out.parent, prefix=out.stem))
    return out


if __name__ == "__main__":
    run_recent_conservative(Path("/work/data/task1/dataset_test.csv"), Path("/work/data/task1/dataset_train.csv"))
