from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HISTORY_END, SUBMIT_DIR, TASK_START, TIME_COL, USER_COL

COLS = ["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]
SUBMISSION_COLUMNS = [USER_COL, "action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl", "index"]

# 行为循环只作为用户内产品对象扩展，不使用全局热门强推。
ACTION_TRANSITIONS: dict[str, list[tuple[str, float]]] = {
    "浏览详情": [("浏览持仓", 5.0), ("收藏", 3.0), ("购买", 2.5)],
    "收藏": [("浏览详情", 2.5), ("购买", 4.5)],
    "购买": [("浏览持仓", 4.0), ("赎回", 3.2), ("浏览详情", 1.5)],
    "浏览持仓": [("赎回", 3.0), ("浏览详情", 2.5), ("购买", 1.5)],
    "赎回": [("浏览详情", 2.0), ("购买", 1.0)],
    # 非财富类只在用户历史真实出现时保留，不做强转移泛化，避免挤占金融产品候选。
    "参与活动": [],
    "线下接触": [],
}

PRODUCT_FAMILIES: dict[str, list[str]] = {
    "理财": ["短债", "周周宝", "月月宝", "季季宝", "半年宝", "多月宝", "固收+", "定期宝", "现金", "非短债固收"],
    "基金": ["灵活配置", "TMT行业", "港股行业", "消费行业", "周期行业", "海外行业", "制造行业", "FOF", "固收+", "固收", "金融地产行业", "医药生物行业"],
    "存款": ["一般性", "结构性"],
}


def _detect_sep(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        header = f.readline()
    return "\t" if "\t" in header else ","


def _read_behavior(path: Path, is_train: bool, include_target_day: bool = False) -> pd.DataFrame:
    print(f"读取{'训练' if is_train else '测试'}集：{path}")
    sep = _detect_sep(path)
    keep_cols = [USER_COL, *COLS, TIME_COL]
    frames: list[pd.DataFrame] = []
    task_start = pd.Timestamp(TASK_START)
    history_end = pd.Timestamp("2025-04-01 23:59:59") if include_target_day else pd.Timestamp(HISTORY_END)

    for chunk in pd.read_csv(path, sep=sep, encoding="utf-8-sig", chunksize=200_000):
        chunk.columns = [str(col).replace("\ufeff", "").strip() for col in chunk.columns]
        unnamed = [col for col in chunk.columns if col == "" or col.startswith("Unnamed")]
        if unnamed:
            chunk = chunk.drop(columns=unnamed)
        missing = [col for col in keep_cols if col not in chunk.columns]
        if missing:
            raise KeyError(f"{path} 缺少必要列：{missing}，实际列名：{chunk.columns.tolist()}")
        chunk = chunk[keep_cols].copy()
        chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], errors="coerce")
        chunk = chunk[(chunk[TIME_COL] >= task_start) & (chunk[TIME_COL] <= history_end)]
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=[USER_COL, TIME_COL, "action_typ", "prod_typ"])
        nonwealth_mask = chunk["prod_typ"].astype(str).eq("非财富")
        chunk.loc[nonwealth_mask, "prod_sub_typ"] = ""
        chunk.loc[nonwealth_mask, "rsk_lvl"] = ""
        chunk = chunk.dropna(subset=["action_typ", "prod_typ"])
        chunk = chunk[nonwealth_mask | chunk[["prod_sub_typ", "rsk_lvl"]].notna().all(axis=1)]
        if not chunk.empty:
            frames.append(chunk)

    if not frames:
        raise RuntimeError(f"{path} 在题目历史窗口内没有可用记录。")
    out = pd.concat(frames, ignore_index=True)
    for col in [USER_COL, *COLS]:
        out[col] = out[col].astype(str)
    return out.sort_values([USER_COL, TIME_COL]).reset_index(drop=True)


def _quad_series(df: pd.DataFrame) -> pd.Series:
    return list(zip(df["action_typ"], df["prod_typ"], df["prod_sub_typ"], df["rsk_lvl"]))  # type: ignore[return-value]


def _build_global_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    all_hist = pd.concat([train_df, test_df], ignore_index=True)
    all_hist["quad"] = _quad_series(all_hist)
    quad_counts = Counter(all_hist["quad"].tolist())
    hot_quads = []
    nonwealth_count = 0
    for q, _ in quad_counts.most_common(120):
        if q[1] == "非财富":
            nonwealth_count += 1
            if nonwealth_count > 1:
                continue
        hot_quads.append(q)
        if len(hot_quads) >= 80:
            break

    prod_sub_rsk = defaultdict(Counter)
    prod_sub_rsk_values: dict[tuple[str, str], list[str]] = {}
    for prod, sub, rsk in zip(all_hist["prod_typ"], all_hist["prod_sub_typ"], all_hist["rsk_lvl"]):
        prod_sub_rsk[(prod, sub)].update([rsk])
    for key, counter in prod_sub_rsk.items():
        prod_sub_rsk_values[key] = [r for r, _ in counter.most_common(3)]

    observed_prod_sub_rsk = set(zip(all_hist["prod_typ"], all_hist["prod_sub_typ"], all_hist["rsk_lvl"]))
    return {
        "quad_counts": quad_counts,
        "hot_quads": hot_quads,
        "prod_sub_rsk_values": prod_sub_rsk_values,
        "observed_prod_sub_rsk": observed_prod_sub_rsk,
    }


def _add_candidate(
    scores: dict[tuple[str, str, str, str], float],
    quad: tuple[str, str, str, str],
    score: float,
    observed_prod_sub_rsk: set[tuple[str, str, str]],
) -> None:
    act, prod, sub, rsk = quad
    if not act or not prod:
        return
    if prod == "非财富":
        sub = ""
        rsk = ""
        quad = (act, prod, sub, rsk)
    elif not sub or not rsk:
        return
    # 不手写产品-风险规则，只要求 prod/sub/rsk 在历史数据中出现过，避免乱造极端组合。
    if prod != "非财富" and (prod, sub, rsk) not in observed_prod_sub_rsk:
        return
    scores[quad] = scores.get(quad, 0.0) + score


def _mode_values(series: pd.Series, n: int) -> list[str]:
    if series.empty:
        return []
    return series.astype(str).value_counts().head(n).index.tolist()


def score_one_user_candidates(
    g: pd.DataFrame,
    stats: dict,
    top_k: int = 80,
) -> list[tuple[tuple[str, str, str, str], float]]:
    cutoff = pd.Timestamp(HISTORY_END)
    g = g.sort_values(TIME_COL).copy()
    g["quad"] = _quad_series(g)
    g["days_diff"] = (cutoff - g[TIME_COL]).dt.total_seconds().clip(lower=0) / 86400

    observed_prod_sub_rsk = stats["observed_prod_sub_rsk"]
    prod_sub_rsk_values = stats["prod_sub_rsk_values"]
    quad_global_counts: Counter = stats["quad_counts"]
    scores: dict[tuple[str, str, str, str], float] = {}

    last_quad = g["quad"].iloc[-1]
    last_action, _last_prod, _last_sub, last_rsk = last_quad

    recent_1 = g[g["days_diff"] <= 1]
    recent_3 = g[g["days_diff"] <= 3]
    recent_7 = g[g["days_diff"] <= 7]
    recent_base = recent_7 if not recent_7.empty else g

    # 1) 完整四元组 repeat：最近、频次、衰减。
    for i, quad in enumerate(reversed(g["quad"].tolist())):
        _add_candidate(scores, quad, 28.0 / np.log2(i + 2), observed_prod_sub_rsk)
        if i >= 80:
            break

    for days, recent, weight in [(1, recent_1, 18.0), (3, recent_3, 11.0), (7, recent_7, 7.0), (31, g, 3.0)]:
        if recent.empty:
            continue
        counts = recent["quad"].value_counts()
        for quad, cnt in counts.head(60).items():
            _add_candidate(scores, quad, weight * np.log1p(float(cnt)), observed_prod_sub_rsk)

    for quad, sub_g in g.groupby("quad", sort=False):
        decay = float(np.exp(-sub_g["days_diff"] / 3.0).sum())
        _add_candidate(scores, quad, 4.0 * decay, observed_prod_sub_rsk)

    # 2) 用户内 action x object 交叉。
    actions = _mode_values(recent_base["action_typ"], 5)
    if last_action not in actions:
        actions.insert(0, last_action)
    objects_df = recent_base[["prod_typ", "prod_sub_typ", "rsk_lvl"]].value_counts().head(18)
    objects = list(objects_df.index)
    for rank_a, act in enumerate(actions):
        for rank_o, (prod, sub, rsk) in enumerate(objects):
            score = 10.0 / np.log2(rank_a + 2) + 8.0 / np.log2(rank_o + 2)
            _add_candidate(scores, (act, prod, sub, rsk), score, observed_prod_sub_rsk)

    # 3) 行为转移：只沿用户自己的产品对象扩展，不拿别人产品硬刷。
    base_objects = list(g[["prod_typ", "prod_sub_typ", "rsk_lvl"]].value_counts().head(24).index)
    source_actions = _mode_values(recent_base["action_typ"], 5)
    for source_act in source_actions:
        for next_act, trans_w in ACTION_TRANSITIONS.get(source_act, []):
            for rank_o, (prod, sub, rsk) in enumerate(base_objects):
                _add_candidate(scores, (next_act, prod, sub, rsk), trans_w + 4.0 / np.log2(rank_o + 2), observed_prod_sub_rsk)

    # 4) 购买扩展：金融产品对象如果被浏览/持仓/收藏过，补用户内购买候选。
    finance_objects = [obj for obj in base_objects if obj[0] != "非财富"]
    user_actions = set(g["action_typ"].tolist())
    buy_signal = bool(user_actions & {"浏览详情", "浏览持仓", "收藏", "购买"})
    if buy_signal:
        for rank_o, (prod, sub, rsk) in enumerate(finance_objects[:30]):
            buy_w = 7.0 if prod in {"理财", "存款"} else 4.5
            _add_candidate(scores, ("购买", prod, sub, rsk), buy_w + 5.0 / np.log2(rank_o + 2), observed_prod_sub_rsk)

    # 5) 产品族邻域扩展：只在用户已出现同产品大类时，用用户风险偏好补同族子类。
    family_actions = [a for a in actions if a not in {"参与活动", "线下接触"}]
    user_prod_rsk = g.groupby("prod_typ")["rsk_lvl"].agg(lambda s: s.value_counts().head(2).index.astype(str).tolist()).to_dict()
    seen_prod_sub = set(zip(g["prod_typ"], g["prod_sub_typ"]))
    for prod, family_subs in PRODUCT_FAMILIES.items():
        if prod not in set(g["prod_typ"].tolist()):
            continue
        risks = user_prod_rsk.get(prod, [])
        if not risks:
            continue
        for sub_rank, sub in enumerate(family_subs):
            if (prod, sub) in seen_prod_sub:
                continue
            for risk in risks:
                if (prod, sub, risk) not in observed_prod_sub_rsk:
                    continue
                for action_rank, act in enumerate(family_actions[:4]):
                    base_w = 4.0 if prod == "理财" else 3.2
                    _add_candidate(
                        scores,
                        (act, prod, sub, risk),
                        base_w / np.log2(sub_rank + 2) + 2.0 / np.log2(action_rank + 2),
                        observed_prod_sub_rsk,
                    )
                if buy_signal and prod in {"理财", "存款"}:
                    _add_candidate(scores, ("购买", prod, sub, risk), 3.8 / np.log2(sub_rank + 2), observed_prod_sub_rsk)

    # 6) 风险慢变量：对已有 action/prod/sub 换成用户/产品子类常见风险。
    user_risks = []
    user_risks.extend(_mode_values(recent_base["rsk_lvl"], 3))
    user_risks.extend(_mode_values(g["rsk_lvl"], 3))
    seed_quads = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:80]
    for (act, prod, sub, _rsk), base_score in seed_quads:
        risks = []
        risks.extend(user_risks)
        risks.extend(prod_sub_rsk_values.get((prod, sub), []))
        for risk in dict.fromkeys(risks):
            _add_candidate(scores, (act, prod, sub, risk), min(6.0, base_score * 0.2), observed_prod_sub_rsk)

    # 7) 风险、最近 action、全局弱先验重排。
    user_rsk_mode = _mode_values(g["rsk_lvl"], 1)[0]
    recent_rsk_mode = _mode_values(recent_base["rsk_lvl"], 1)[0]
    final_scores: list[tuple[float, tuple[str, str, str, str]]] = []
    for quad, score in scores.items():
        act, _prod, _sub, rsk = quad
        if quad == last_quad:
            score += 16.0
        if act == last_action:
            score += 3.0
        if prod == "非财富" and quad != last_quad:
            score -= 8.0
        if act == "线下接触" and last_action != "线下接触":
            score -= 6.0
        if act == "购买" and prod in {"理财", "存款"}:
            score += 2.5
        if rsk == last_rsk:
            score += 3.5
        if rsk == recent_rsk_mode:
            score += 2.5
        if rsk == user_rsk_mode:
            score += 1.5
        score += 0.15 * np.log1p(quad_global_counts.get(quad, 0))
        final_scores.append((score, quad))

    final_scores.sort(key=lambda x: x[0], reverse=True)
    out: list[tuple[tuple[str, str, str, str], float]] = []
    seen = set()
    min_fallback_score = final_scores[-1][0] - 1.0 if final_scores else 0.0
    for score, quad in final_scores:
        if quad in seen:
            continue
        seen.add(quad)
        out.append((quad, score))
        if len(out) >= top_k:
            return out

    # 6) 全局热门仅最低优先级兜底。
    for rank, quad in enumerate(stats["hot_quads"]):
        if quad not in seen:
            seen.add(quad)
            out.append((quad, min_fallback_score - rank * 0.01))
        if len(out) >= top_k:
            break
    while len(out) < top_k:
        out.append((stats["hot_quads"][0], min_fallback_score - 99.0))
    return out[:top_k]


def _predict_one_user(g: pd.DataFrame, stats: dict, top_k: int = 20) -> list[tuple[str, str, str, str]]:
    return [quad for quad, _score in score_one_user_candidates(g, stats, top_k=top_k)]


def build_fast_light_cross_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    print("构造全局弱统计/风险分布...")
    stats = _build_global_stats(train_df, test_df)
    print(
        f"全局兜底热门 {len(stats['hot_quads'])} 个，"
        f"观测 prod-sub-rsk {len(stats['observed_prod_sub_rsk'])} 个"
    )

    rows = []
    user_count = test_df[USER_COL].nunique()
    for n, (uid, g) in enumerate(test_df.groupby(USER_COL, sort=False), 1):
        top_quads = _predict_one_user(g, stats, top_k=top_k)
        for idx, (act, prod, sub, rsk) in enumerate(top_quads, 1):
            rows.append({
                USER_COL: uid,
                "action_typ": act,
                "prod_typ": prod,
                "prod_sub_typ": sub,
                "rsk_lvl": rsk,
                "index": idx,
            })
        if n % 10000 == 0:
            print(f"fast-light-cross 已预测 {n}/{user_count} 用户")
    return pd.DataFrame(rows)[SUBMISSION_COLUMNS]


def run_task1_fast_light_cross(train_path: Path, test_path: Path, output_name: str = "prediction.csv") -> Path:
    print("📥 加载历史窗口训练/测试数据...")
    train_df = _read_behavior(train_path, is_train=True)
    test_df = _read_behavior(test_path, is_train=False)
    print(f"训练历史：{len(train_df)} 行，{train_df[USER_COL].nunique()} 用户")
    print(f"测试历史：{len(test_df)} 行，{test_df[USER_COL].nunique()} 用户")

    print("⚡ 生成 fast-light-cross top20 预测...")
    submission_df = build_fast_light_cross_predictions(train_df, test_df, top_k=20)
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMIT_DIR / output_name
    submission_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 提交文件已保存：{out_path}")
    print(f"总行数：{len(submission_df)}，用户数：{submission_df[USER_COL].nunique()}")
    return out_path


if __name__ == "__main__":
    run_task1_fast_light_cross(
        Path("/work/data/task1/dataset_train.csv"),
        Path("/work/data/task1/dataset_test.csv"),
    )
