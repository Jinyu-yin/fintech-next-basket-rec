from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HISTORY_END, SUBMIT_DIR, TASK_START, TIME_COL, USER_COL
from .feature_engineering import COLS, _fill_missing_values

SUBMISSION_COLUMNS = ["user_id", "action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl", "index"]


def _read_behavior_csv(path: Path) -> pd.DataFrame:
    """读取比赛 CSV，兼容 BOM、逗号/制表符分隔和首列无名行号。"""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    unnamed_cols = [col for col in df.columns if col == "" or col.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def _prepare_frames(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = _read_behavior_csv(train_path)
    test_df = _read_behavior_csv(test_path)
    train_df[TIME_COL] = pd.to_datetime(train_df[TIME_COL], errors="coerce")
    test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL], errors="coerce")

    task_start = pd.Timestamp(TASK_START)
    history_end = pd.Timestamp(HISTORY_END)
    train_df = train_df[(train_df[TIME_COL] >= task_start) & (train_df[TIME_COL] <= history_end)].copy()
    test_df = test_df[(test_df[TIME_COL] >= task_start) & (test_df[TIME_COL] <= history_end)].copy()

    train_df = _fill_missing_values(train_df)
    test_df = _fill_missing_values(test_df)
    train_df = train_df.dropna(subset=[USER_COL, TIME_COL] + COLS).reset_index(drop=True)
    test_df = test_df.dropna(subset=[USER_COL, TIME_COL]).reset_index(drop=True)
    return train_df, test_df



@dataclass
class ObservedSpace:
    quads: set[tuple[str, str, str, str]]
    action_prod: set[tuple[str, str]]
    prod_sub: set[tuple[str, str]]
    prod_rsk: set[tuple[str, str]]
    prod_sub_rsk_values: dict[tuple[str, str], list[str]]
    prod_sub_rsk_mode: dict[tuple[str, str], str]


def _build_observed_space(*frames: pd.DataFrame) -> ObservedSpace:
    """从真实历史数据学习可出现空间，替代手写 prod-risk 强规则。"""
    base = pd.concat([df[COLS] for df in frames if not df.empty], ignore_index=True).dropna(subset=COLS)
    quads = set(zip(base["action_typ"], base["prod_typ"], base["prod_sub_typ"], base["rsk_lvl"]))
    action_prod = set(zip(base["action_typ"], base["prod_typ"]))
    prod_sub = set(zip(base["prod_typ"], base["prod_sub_typ"]))
    prod_rsk = set(zip(base["prod_typ"], base["rsk_lvl"]))

    prod_sub_rsk_values: dict[tuple[str, str], list[str]] = {}
    prod_sub_rsk_mode: dict[tuple[str, str], str] = {}
    for key, s in base.groupby(["prod_typ", "prod_sub_typ"])["rsk_lvl"]:
        counts = s.value_counts()
        prod_sub_rsk_values[key] = counts.index.astype(str).tolist()[:4]
        prod_sub_rsk_mode[key] = str(counts.index[0])

    return ObservedSpace(
        quads=quads,
        action_prod=action_prod,
        prod_sub=prod_sub,
        prod_rsk=prod_rsk,
        prod_sub_rsk_values=prod_sub_rsk_values,
        prod_sub_rsk_mode=prod_sub_rsk_mode,
    )


def _is_observed_compatible(quad: tuple[str, str, str, str], observed: ObservedSpace) -> bool:
    act, prod, sub, rsk = quad
    if quad in observed.quads:
        return True
    return (act, prod) in observed.action_prod and (prod, sub) in observed.prod_sub and (prod, rsk) in observed.prod_rsk


def _unique_extend(out: list[tuple[str, str, str, str]], items, limit: int = 20, observed: ObservedSpace | None = None) -> None:
    seen = set(out)
    for item in items:
        if len(out) >= limit:
            break
        if item in seen:
            continue
        if observed is not None and not _is_observed_compatible(item, observed):
            continue
        seen.add(item)
        out.append(item)


def _build_history_hot_quads(train_df: pd.DataFrame, observed: ObservedSpace, top_n: int = 300) -> list[tuple[str, str, str, str]]:
    """只基于历史窗口构造全局兜底热门，禁止使用 4/1 目标日信息。"""
    history_df = train_df[train_df[TIME_COL] <= pd.Timestamp(HISTORY_END)].copy()
    history_df["quadruple"] = history_df[COLS].astype(str).agg("|".join, axis=1)
    quads = []
    for q in history_df["quadruple"].value_counts().nlargest(top_n).index.tolist():
        quad = tuple(q.split("|"))
        if len(quad) == 4 and _is_observed_compatible(quad, observed):
            quads.append(quad)  # type: ignore[arg-type]
    return quads


def _build_history_transition_map(
    train_df: pd.DataFrame,
    observed: ObservedSpace,
    top_n: int = 8,
) -> dict[tuple[str, str, str, str], list[tuple[str, str, str, str]]]:
    """只基于历史窗口学习相邻行为转移，禁止使用 4/1 目标日标签。"""
    history = train_df[train_df[TIME_COL] <= pd.Timestamp(HISTORY_END)].sort_values([USER_COL, TIME_COL]).copy()
    if history.empty:
        return {}
    history["quad"] = list(zip(history["action_typ"], history["prod_typ"], history["prod_sub_typ"], history["rsk_lvl"]))

    transition_counter: dict[tuple[str, str, str, str], Counter] = {}
    for _uid, g in history.groupby(USER_COL, sort=False):
        quads = g["quad"].tolist()
        for prev_quad, next_quad in zip(quads[:-1], quads[1:]):
            if prev_quad == next_quad:
                continue
            counter = transition_counter.setdefault(prev_quad, Counter())
            counter.update([next_quad])

    return {
        prev_quad: [quad for quad, _cnt in counter.most_common(top_n) if _is_observed_compatible(quad, observed)]
        for prev_quad, counter in transition_counter.items()
    }


def _user_field_combo_candidates(g: pd.DataFrame, observed: ObservedSpace, limit: int = 80) -> list[tuple[str, str, str, str]]:
    """字段分解重组候选：提升 recall，但只用用户历史中高频字段，避免过度模板化。"""
    if g.empty:
        return []
    recent = g[g["days_diff"] <= 7]
    base = recent if not recent.empty else g
    actions = base["action_typ"].value_counts().head(4).index.tolist()
    prods = base["prod_typ"].value_counts().head(4).index.tolist()
    prod_subs = base.groupby("prod_typ")["prod_sub_typ"].agg(lambda s: s.value_counts().head(3).index.tolist()).to_dict()
    prod_rsks = base.groupby("prod_typ")["rsk_lvl"].agg(lambda s: s.value_counts().head(3).index.tolist()).to_dict()

    out = []
    seen = set()
    for act in actions:
        for prod in prods:
            for sub in prod_subs.get(prod, []):
                for rsk in prod_rsks.get(prod, []):
                    quad = (act, prod, sub, rsk)
                    if quad in seen or not _is_observed_compatible(quad, observed):
                        continue
                    seen.add(quad)
                    out.append(quad)
                    if len(out) >= limit:
                        return out
    return out


def _risk_variant_candidates(
    candidates: list[tuple[str, str, str, str]],
    user_df: pd.DataFrame,
    observed: ObservedSpace,
    limit: int = 80,
) -> list[tuple[str, str, str, str]]:
    """把 rsk_lvl 当慢变量：为已有 action/prod/sub 生成用户风险众数和产品子类常见风险变体。"""
    if not candidates or user_df.empty:
        return []
    recent = user_df[user_df["days_diff"] <= 7]
    base = recent if not recent.empty else user_df
    user_risks = []
    for source in (base, user_df):
        user_risks.extend(source["rsk_lvl"].value_counts().head(3).index.astype(str).tolist())

    out = []
    seen = set()
    for act, prod, sub, _rsk in candidates:
        risk_values = []
        risk_values.extend(user_risks)
        risk_values.extend(observed.prod_sub_rsk_values.get((prod, sub), []))
        for risk in risk_values:
            quad = (act, prod, sub, risk)
            if quad in seen or not _is_observed_compatible(quad, observed):
                continue
            seen.add(quad)
            out.append(quad)
            if len(out) >= limit:
                return out
    return out


def _rank_user_repeat_quads(
    user_df: pd.DataFrame,
    cutoff: pd.Timestamp,
    hot_quads: list[tuple[str, str, str, str]],
    observed: ObservedSpace,
    transition_map: dict[tuple[str, str, str, str], list[tuple[str, str, str, str]]] | None = None,
) -> list[tuple[str, str, str, str]]:
    """召回尽量宽、排序尽量个性化：目标是先把 Recall@20 拉高，再兼顾 NDCG。"""
    if user_df.empty:
        return hot_quads[:20]
    g = user_df.sort_values(TIME_COL).copy()
    g["quad"] = list(zip(g["action_typ"], g["prod_typ"], g["prod_sub_typ"], g["rsk_lvl"]))
    g["days_diff"] = (cutoff - g[TIME_COL]).dt.total_seconds() / 86400
    last_quad = g["quad"].iloc[-1]
    last_action, last_prod, last_sub, last_rsk = last_quad
    recent_7d = g[g["days_diff"] <= 7]
    risk_base = recent_7d if not recent_7d.empty else g
    user_rsk_mode = str(g["rsk_lvl"].value_counts().index[0])
    user_recent_rsk_mode = str(risk_base["rsk_lvl"].value_counts().index[0])
    user_rsk_mode_share = float(g["rsk_lvl"].value_counts(normalize=True).iloc[0])

    candidates: list[tuple[str, str, str, str]] = []
    _unique_extend(candidates, g.sort_values(TIME_COL, ascending=False)["quad"].tolist(), limit=160, observed=observed)

    for days in (1, 3, 7, 14, 31):
        recent = g[g["days_diff"] <= days]
        if recent.empty:
            continue
        stats = (
            recent.groupby("quad", sort=False)
            .agg(cnt=("quad", "size"), last_tm=(TIME_COL, "max"))
            .sort_values(["cnt", "last_tm"], ascending=[False, False])
        )
        _unique_extend(candidates, stats.index.tolist(), limit=180, observed=observed)

    if transition_map:
        _unique_extend(candidates, transition_map.get(last_quad, []), limit=190, observed=observed)
    _unique_extend(candidates, _user_field_combo_candidates(g, observed), limit=210, observed=observed)
    _unique_extend(candidates, _risk_variant_candidates(candidates, g, observed), limit=230, observed=observed)

    # 注意：全局历史热门不参与用户候选打分，只在最终不足 20 条时作为最低优先级补齐。
    scores = []
    for quad in candidates:
        qdf = g[g["quad"] == quad]
        if qdf.empty:
            cnt_1d = cnt_3d = cnt_7d = cnt_all = decay_sum = 0.0
            last_seen_days = 99.0
        else:
            cnt_1d = float((qdf["days_diff"] <= 1).sum())
            cnt_3d = float((qdf["days_diff"] <= 3).sum())
            cnt_7d = float((qdf["days_diff"] <= 7).sum())
            cnt_all = float(len(qdf))
            decay_sum = float(np.exp(-qdf["days_diff"].clip(lower=0) / 3.0).sum())
            last_seen_days = float(qdf["days_diff"].min())

        act, prod, sub, rsk = quad
        trans_score = 1.0 if transition_map and quad in transition_map.get(last_quad, []) else 0.0
        score = 0.0
        score += 22.0 if quad == last_quad else 0.0
        score += 15.0 * cnt_1d
        score += 9.0 * cnt_3d
        score += 5.0 * cnt_7d
        score += 3.5 * decay_sum
        score += 1.0 * np.log1p(cnt_all)
        score += max(0.0, 5.0 - last_seen_days) * 1.5
        score += 7.0 * trans_score
        prod_sub_mode_rsk = observed.prod_sub_rsk_mode.get((prod, sub))
        score += 3.0 if prod == last_prod else 0.0
        score += 2.0 if act == last_action else 0.0
        score += 1.0 if sub == last_sub else 0.0
        score += 2.5 if rsk == last_rsk else 0.0
        score += 2.0 if rsk == user_recent_rsk_mode else 0.0
        score += (1.0 + 2.0 * user_rsk_mode_share) if rsk == user_rsk_mode else 0.0
        score += 2.0 if prod_sub_mode_rsk is not None and rsk == prod_sub_mode_rsk else 0.0
        scores.append((score, quad))

    scores.sort(key=lambda x: x[0], reverse=True)
    ranked = []
    seen = set()
    for _score, quad in scores:
        if quad not in seen and _is_observed_compatible(quad, observed):
            seen.add(quad)
            ranked.append(quad)
        if len(ranked) >= 20:
            break
    _unique_extend(ranked, hot_quads, limit=20, observed=observed)
    return ranked


def build_repeat_first_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_users: list[str] | None = None,
    top_k: int = 20,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(HISTORY_END)
    observed = _build_observed_space(train_df, test_df)
    hot_quads = _build_history_hot_quads(train_df, observed, top_n=60)
    if not hot_quads:
        raise RuntimeError("无法构造历史热门四元组，请检查训练数据和时间窗口。")
    transition_map = _build_history_transition_map(train_df, observed, top_n=12)
    print(
        "repeat-first 候选增强："
        f"历史热门 {len(hot_quads)} 个，历史相邻转移 {len(transition_map)} 个状态，"
        f"观测四元组 {len(observed.quads)} 个"
    )

    if test_users is None:
        test_users = test_df[USER_COL].dropna().drop_duplicates().tolist()

    user_groups = {uid: g for uid, g in test_df.groupby(USER_COL, sort=False)}
    rows = []
    for n, uid in enumerate(test_users, 1):
        user_df = user_groups.get(uid)
        top_quads = hot_quads[:top_k] if user_df is None else _rank_user_repeat_quads(user_df, cutoff, hot_quads, observed, transition_map)
        _unique_extend(top_quads, hot_quads, limit=top_k, observed=observed)
        while len(top_quads) < top_k:
            top_quads.append(hot_quads[0])

        for idx, (act, prod, sub, rsk) in enumerate(top_quads[:top_k], 1):
            rows.append({
                "user_id": uid,
                "action_typ": act,
                "prod_typ": prod,
                "prod_sub_typ": sub,
                "rsk_lvl": rsk,
                "index": idx,
            })
        if n % 10000 == 0:
            print(f"  repeat-first 已预测 {n}/{len(test_users)} 个用户")

    return pd.DataFrame(rows)[SUBMISSION_COLUMNS]


def run_task1_repeat_first(train_path: Path, test_path: Path, output_name: str = "prediction.csv") -> Path:
    print("📥 加载并裁剪训练集/测试集...")
    train_df, test_df = _prepare_frames(train_path, test_path)
    print(f"训练集裁剪后：{len(train_df)} 行，{train_df[USER_COL].nunique()} 用户")
    print(f"测试集裁剪后：{len(test_df)} 行，{test_df[USER_COL].nunique()} 用户")

    print("🔁 生成 repeat-first top20 预测...")
    submission_df = build_repeat_first_predictions(train_df, test_df)
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMIT_DIR / output_name
    submission_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ repeat-first 提交已保存：{out_path}")
    print(f"总行数：{len(submission_df)}，用户数：{submission_df[USER_COL].nunique()}")
    return out_path
