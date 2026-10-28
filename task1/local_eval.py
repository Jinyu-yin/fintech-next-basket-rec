from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import LOCAL_TRAIN_PATHS, RANDOM_STATE, TIME_COL, USER_COL, TARGET_COLS, SUBMIT_DIR
from .feature_engineering import (
    ACTION_PROD_VALID,
    COLS,
    PROD_RSK_VALID,
    _fill_missing_values,
    build_global_top_quads,
    build_next_day_features,
    build_quad_april1_count,
)
from .modeling import generate_candidates, rank_candidates, train_next_day_ranker
from .pipeline import build_rank_samples
from .diagnostic_outputs import (
    summarize_holdout_files,
    write_candidate_recall_diagnostic,
    write_holdout_compare,
    write_prediction_distribution,
    write_prediction_format_report,
)
from .fast_light_ranker import predict_with_ranker, train_fast_light_ranker
from .full_label_classifier import predict_full_label, train_full_label_classifier
from .repeat_first import build_repeat_first_predictions


EVAL_START = pd.Timestamp("2025-03-01 00:00:00")
EVAL_CUTOFF = pd.Timestamp("2025-03-31 23:59:59")
EVAL_TARGET = pd.Timestamp("2025-04-01 00:00:00")
EVAL_END = pd.Timestamp("2025-04-02 00:00:00")


def load_local_train(train_paths: list[Path], max_users: int = 0, seed: int = RANDOM_STATE) -> pd.DataFrame:
    frames = []
    keep_cols = [USER_COL, *TARGET_COLS, TIME_COL]
    selected_users: set[str] = set()
    rng = np.random.default_rng(seed)

    for path in train_paths:
        print(f"读取并分块裁剪：{path}")
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            header = f.readline()
        sep = "\t" if "\t" in header else ","
        try:
            chunk_iter = pd.read_csv(
                path,
                sep=sep,
                encoding="utf-8-sig",
                usecols=lambda col: str(col).replace("\ufeff", "").strip() in keep_cols
                or str(col).replace("\ufeff", "").strip().startswith("Unnamed"),
                chunksize=25_000,
                low_memory=True,
            )
        except Exception:
            chunk_iter = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                encoding="utf-8-sig",
                chunksize=10_000,
                on_bad_lines="skip",
            )

        for chunk in chunk_iter:
            chunk.columns = [str(col).replace("\ufeff", "").strip() for col in chunk.columns]
            unnamed_cols = [col for col in chunk.columns if col == "" or col.startswith("Unnamed")]
            if unnamed_cols:
                chunk = chunk.drop(columns=unnamed_cols)
            missing_cols = [col for col in keep_cols if col not in chunk.columns]
            if missing_cols:
                raise KeyError(f"{path} 缺少必要列：{missing_cols}，实际列名：{chunk.columns.tolist()}")

            chunk = chunk[keep_cols].copy()
            chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], errors="coerce")
            mask = (chunk[TIME_COL] >= EVAL_START) & (chunk[TIME_COL] < EVAL_END)
            chunk = chunk.loc[mask]
            if chunk.empty:
                continue

            if max_users:
                chunk_users = chunk[USER_COL].dropna().astype(str).unique().tolist()
                if len(selected_users) < max_users:
                    remaining = max_users - len(selected_users)
                    new_users = [u for u in chunk_users if u not in selected_users]
                    if len(new_users) > remaining:
                        new_users = rng.choice(np.array(new_users, dtype=object), size=remaining, replace=False).tolist()
                    selected_users.update(new_users)
                chunk = chunk[chunk[USER_COL].astype(str).isin(selected_users)]
                if chunk.empty:
                    continue

            frames.append(chunk.copy())

    if not frames:
        raise RuntimeError("题目时间窗口内没有读到训练数据，请检查日期范围和文件路径。")

    train_df = pd.concat(frames, ignore_index=True)
    train_df = _fill_missing_values(train_df)
    train_df = train_df.dropna(subset=[USER_COL, TIME_COL] + COLS).reset_index(drop=True)
    print(f"本地评估实际载入记录数：{len(train_df)}，用户数：{train_df[USER_COL].nunique()}")
    return train_df


def ndcg_at_20(pred_quads: list[tuple], true_quads: set[str]) -> float | None:
    if not true_quads:
        return None
    dcg = 0.0
    for rank, quad in enumerate(pred_quads[:20], start=1):
        if "|".join(quad) in true_quads:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(20, len(true_quads))
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def build_fast_training_samples(
    train_df: pd.DataFrame,
    windows: list[tuple[str, str, float]],
    use_april1_boost: bool,
) -> tuple[pd.DataFrame, np.ndarray, list[int], pd.DataFrame, np.ndarray, list[int], np.ndarray, np.ndarray]:
    """轻量版训练样本：只取少量近日期窗口，优先快速验证算法有效性。"""
    train_X_list = []
    train_y_list = []
    train_groups = []
    train_w_list = []

    for cutoff_date, target_date, weight in windows:
        print(f"构造训练窗口：{cutoff_date} -> {target_date}，权重 {weight}")
        X, y, groups, weights = build_rank_samples(train_df, cutoff_date, target_date, sample_weight=weight)
        if len(X) == 0:
            continue
        train_X_list.append(X)
        train_y_list.append(y)
        train_groups.extend(groups)
        train_w_list.append(weights)

    if use_april1_boost:
        print("构造训练增强窗口：2025-03-31 23:59:59 -> 2025-04-01，权重 2.0")
        X, y, groups, weights = build_rank_samples(
            train_df,
            "2025-03-31 23:59:59",
            "2025-04-01",
            sample_weight=2.0,
        )
        if len(X) > 0:
            train_X_list.append(X)
            train_y_list.append(y)
            train_groups.extend(groups)
            train_w_list.append(weights)

    val_X, val_y, val_groups, val_w = build_rank_samples(
        train_df,
        "2025-03-30 23:59:59",
        "2025-03-31",
        sample_weight=1.0,
    )

    if not train_X_list:
        raise RuntimeError("没有构造出训练样本，请检查日期窗口和数据。")

    return (
        pd.concat(train_X_list, ignore_index=True),
        np.concatenate(train_y_list),
        train_groups,
        val_X,
        val_y,
        val_groups,
        np.concatenate(train_w_list),
        val_w,
    )


def _write_holdout_artifacts(pred_df: pd.DataFrame, eval_target: pd.DataFrame, prefix: str) -> tuple[Path, Path, Path]:
    """保存本地 holdout 预测、真实标签和逐用户命中明细，避免只看聚合指标。"""
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = SUBMIT_DIR / f"{prefix}_holdout_pred.csv"
    target_path = SUBMIT_DIR / f"{prefix}_holdout_true.csv"
    compare_path = SUBMIT_DIR / f"{prefix}_holdout_compare.csv"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    target_cols = [USER_COL, *COLS, TIME_COL]
    eval_target[target_cols].sort_values([USER_COL, TIME_COL]).to_csv(target_path, index=False, encoding="utf-8-sig")

    target_map = (
        eval_target.assign(quadruple=eval_target[COLS].astype(str).agg("|".join, axis=1))
        .groupby(USER_COL)["quadruple"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )
    rows = []
    for uid, g in pred_df.sort_values([USER_COL, "index"]).groupby(USER_COL, sort=False):
        true_quads = target_map.get(uid, set())
        pred_quads = ["|".join(q) for q in zip(g["action_typ"], g["prod_typ"], g["prod_sub_typ"], g["rsk_lvl"])]
        hit_ranks = [i + 1 for i, q in enumerate(pred_quads[:20]) if q in true_quads]
        same_aps_wrong_rsk = []
        pred_action_prod_sub = {"|".join(q.split("|")[:3]) for q in pred_quads[:20]}
        for true_quad in true_quads:
            parts = true_quad.split("|")
            if len(parts) == 4 and "|".join(parts[:3]) in pred_action_prod_sub and true_quad not in pred_quads[:20]:
                same_aps_wrong_rsk.append(true_quad)
        rows.append({
            USER_COL: uid,
            "true_count": len(true_quads),
            "hit_count": len(hit_ranks),
            "hit_ranks": ";".join(map(str, hit_ranks)),
            "same_action_prod_sub_wrong_rsk_count": len(same_aps_wrong_rsk),
            "same_action_prod_sub_wrong_rsk_quads": " || ".join(sorted(same_aps_wrong_rsk)),
            "true_quads": " || ".join(sorted(true_quads)),
            "pred_top20": " || ".join(pred_quads[:20]),
        })
    pd.DataFrame(rows).to_csv(compare_path, index=False, encoding="utf-8-sig")
    return pred_path, target_path, compare_path


def _score_top20_dataframe(pred_df: pd.DataFrame, target_map: dict[str, set[str]]) -> dict[str, float]:
    scores = []
    recall_hits = 0
    recall_total = 0
    grouped = pred_df.sort_values([USER_COL, "index"]).groupby(USER_COL, sort=False)
    for uid, g in grouped:
        true_quads = target_map.get(uid, set())
        if not true_quads:
            continue
        pred_quads = list(zip(g["action_typ"], g["prod_typ"], g["prod_sub_typ"], g["rsk_lvl"]))
        score = ndcg_at_20(pred_quads, true_quads)
        if score is not None:
            scores.append(score)
        pred_set = {"|".join(q) for q in pred_quads[:20]}
        recall_hits += len(pred_set & true_quads)
        recall_total += min(20, len(true_quads))
    return {
        "eval_users_with_target": float(len(scores)),
        "mean_ndcg20": float(np.mean(scores)) if scores else 0.0,
        "median_ndcg20": float(np.median(scores)) if scores else 0.0,
        "hit_recall20": float(recall_hits / recall_total) if recall_total else 0.0,
    }


def evaluate_repeat_first_holdout(all_df: pd.DataFrame, train_users: set[str], eval_users: set[str], save_artifacts: bool = True) -> dict[str, float]:
    train_df = all_df[all_df[USER_COL].isin(train_users)].copy()
    eval_df = all_df[all_df[USER_COL].isin(eval_users)].copy()
    eval_history = eval_df[eval_df[TIME_COL] <= EVAL_CUTOFF].copy()
    eval_target = eval_df[(eval_df[TIME_COL] >= EVAL_TARGET) & (eval_df[TIME_COL] < EVAL_END)].copy()
    eval_target["quadruple"] = (
        eval_target["action_typ"].astype(str).str.cat(
            [eval_target["prod_typ"].astype(str), eval_target["prod_sub_typ"].astype(str), eval_target["rsk_lvl"].astype(str)],
            sep="|",
        )
    )
    target_map = eval_target.groupby(USER_COL)["quadruple"].apply(lambda s: set(s.tolist())).to_dict()
    pred_df = build_repeat_first_predictions(
        train_df,
        eval_history,
        test_users=eval_history[USER_COL].drop_duplicates().tolist(),
        top_k=20,
    )
    metrics = _score_top20_dataframe(pred_df, target_map)
    if save_artifacts:
        pred_path, target_path, compare_path = _write_holdout_artifacts(pred_df, eval_target, "repeat_first")
        print(f"本地 holdout 预测明细：{pred_path}")
        print(f"本地 holdout 真实标签：{target_path}")
        print(f"本地 holdout 对比明细：{compare_path}")
    metrics["lgb_valid_ndcg20"] = 0.0
    metrics["legal_rate"] = 1.0
    return metrics


def evaluate_fast_light_ranker_holdout(
    all_df: pd.DataFrame,
    train_users: set[str],
    eval_users: set[str],
    max_ranker_train_users: int = 3000,
    candidate_k: int = 80,
    seed: int = RANDOM_STATE,
    save_artifacts: bool = True,
) -> dict[str, float]:
    train_df = all_df[all_df[USER_COL].isin(train_users)].copy()
    eval_df = all_df[all_df[USER_COL].isin(eval_users)].copy()
    eval_history = eval_df[eval_df[TIME_COL] <= EVAL_CUTOFF].copy()
    eval_target = eval_df[(eval_df[TIME_COL] >= EVAL_TARGET) & (eval_df[TIME_COL] < EVAL_END)].copy()
    eval_target["quadruple"] = eval_target[COLS].astype(str).agg("|".join, axis=1)
    target_map = eval_target.groupby(USER_COL)["quadruple"].apply(lambda s: set(s.tolist())).to_dict()

    model, stats = train_fast_light_ranker(
        train_df=train_df,
        eval_history_df=eval_history,
        max_ranker_train_users=max_ranker_train_users,
        candidate_k=candidate_k,
        seed=seed,
    )
    pred_result = predict_with_ranker(
        model,
        train_df,
        eval_history,
        stats,
        candidate_k=candidate_k,
        top_k=20,
        return_candidates=True,
    )
    pred_df, candidate_rows = pred_result
    metrics = _score_top20_dataframe(pred_df, target_map)
    if save_artifacts:
        pred_path, target_path, compare_path = write_holdout_compare(pred_df, eval_target, SUBMIT_DIR, "fast_light_ranker")
        cand_path = write_candidate_recall_diagnostic(candidate_rows, eval_target, pred_df, SUBMIT_DIR, "fast_light_ranker")
        fmt_path = write_prediction_format_report(pred_df, SUBMIT_DIR, "fast_light_ranker_holdout_pred")
        dist_path = write_prediction_distribution(pred_df, SUBMIT_DIR, "fast_light_ranker_holdout_pred")
        compare_df = pd.read_csv(compare_path, encoding="utf-8-sig")
        cand_df = pd.read_csv(cand_path, encoding="utf-8-sig")
        metrics.update(summarize_holdout_files(compare_df, cand_df))
        print(f"本地 holdout 预测明细：{pred_path}")
        print(f"本地 holdout 真实标签：{target_path}")
        print(f"本地 holdout 对比明细：{compare_path}")
        print(f"候选召回诊断：{cand_path}")
        print(f"预测格式诊断：{fmt_path}")
        print(f"预测分布诊断：{dist_path}")
    metrics["lgb_valid_ndcg20"] = 0.0
    metrics["legal_rate"] = 1.0
    return metrics


def evaluate_full_label_holdout(
    all_df: pd.DataFrame,
    train_users: set[str],
    eval_users: set[str],
    max_train_users: int = 5000,
    neg_per_user: int = 100,
    seed: int = RANDOM_STATE,
    save_artifacts: bool = True,
) -> dict[str, float]:
    train_df = all_df[all_df[USER_COL].isin(train_users)].copy()
    eval_df = all_df[all_df[USER_COL].isin(eval_users)].copy()
    eval_history = eval_df[eval_df[TIME_COL] <= EVAL_CUTOFF].copy()
    eval_target = eval_df[(eval_df[TIME_COL] >= EVAL_TARGET) & (eval_df[TIME_COL] < EVAL_END)].copy()
    eval_target["quadruple"] = eval_target["action_typ"].astype(str).str.cat(
        [eval_target["prod_typ"].astype(str), eval_target["prod_sub_typ"].astype(str), eval_target["rsk_lvl"].astype(str)],
        sep="|",
    )
    target_map = eval_target.groupby(USER_COL)["quadruple"].apply(lambda s: set(s.tolist())).to_dict()
    model, label_df, global_counts = train_full_label_classifier(
        train_df=train_df,
        test_history_df=eval_history,
        max_train_users=max_train_users,
        neg_per_user=neg_per_user,
        seed=seed,
    )
    pred_df = predict_full_label(model, label_df, global_counts, eval_history, top_k=20)
    metrics = _score_top20_dataframe(pred_df, target_map)
    if save_artifacts:
        pred_path, target_path, compare_path = write_holdout_compare(pred_df, eval_target, SUBMIT_DIR, "full_label")
        fmt_path = write_prediction_format_report(pred_df, SUBMIT_DIR, "full_label_holdout_pred")
        dist_path = write_prediction_distribution(pred_df, SUBMIT_DIR, "full_label_holdout_pred")
        print(f"本地 holdout 预测明细：{pred_path}")
        print(f"本地 holdout 真实标签：{target_path}")
        print(f"本地 holdout 对比明细：{compare_path}")
        print(f"预测格式诊断：{fmt_path}")
        print(f"预测分布诊断：{dist_path}")
    metrics["lgb_valid_ndcg20"] = 0.0
    metrics["legal_rate"] = 1.0
    return metrics


def evaluate_april1_holdout(
    all_df: pd.DataFrame,
    train_users: set[str],
    eval_users: set[str],
    windows: list[tuple[str, str, float]],
    use_april1_boost: bool,
    save_artifacts: bool = True,
) -> dict[str, float]:
    train_df = all_df[all_df[USER_COL].isin(train_users)].copy()
    eval_df = all_df[all_df[USER_COL].isin(eval_users)].copy()

    global_top_quads = build_global_top_quads(train_df, top_n=100)
    april1_df = train_df[train_df[TIME_COL] >= EVAL_TARGET].copy()
    april1_top_quads = build_global_top_quads(april1_df, top_n=100) if not april1_df.empty else global_top_quads[:]
    if not april1_top_quads:
        april1_top_quads = global_top_quads[:]
    quad_april1_count = build_quad_april1_count(train_df)

    X_train, y_train, groups_train, X_val, y_val, groups_val, train_w, val_w = build_fast_training_samples(
        train_df,
        windows=windows,
        use_april1_boost=use_april1_boost,
    )

    print(f"训练样本：{len(X_train)}，训练 query：{len(groups_train)}")
    print(f"验证样本：{len(X_val)}，验证 query：{len(groups_val)}")

    model = train_next_day_ranker(
        X_train,
        y_train,
        groups_train,
        X_val,
        y_val,
        groups_val,
        sample_weight=train_w,
        val_sample_weight=val_w,
    )
    model.global_top_quads = global_top_quads
    model.april1_top_quads = april1_top_quads
    model.quad_april1_count = quad_april1_count

    eval_history = eval_df[eval_df[TIME_COL] <= EVAL_CUTOFF].copy()
    eval_target = eval_df[(eval_df[TIME_COL] >= EVAL_TARGET) & (eval_df[TIME_COL] < EVAL_END)].copy()
    eval_target["quadruple"] = eval_target[COLS].astype(str).agg("|".join, axis=1)
    target_map = eval_target.groupby(USER_COL)["quadruple"].apply(lambda s: set(s.tolist())).to_dict()

    eval_feats = build_next_day_features(eval_history, EVAL_CUTOFF)

    scores = []
    recall_hits = 0
    recall_total = 0
    legal_rows = 0
    total_rows = 0
    pred_rows = []

    for _, user_row in eval_feats.iterrows():
        uid = user_row[USER_COL]
        true_quads = target_map.get(uid, set())
        if not true_quads:
            continue
        candidates = generate_candidates(user_row, global_top_quads, april1_top_quads)
        top20 = rank_candidates(model, user_row, candidates)

        score = ndcg_at_20(top20, true_quads)
        if score is not None:
            scores.append(score)
        pred_set = {"|".join(q) for q in top20[:20]}
        recall_hits += len(pred_set & true_quads)
        recall_total += min(20, len(true_quads))
        for idx, (act, prod, sub, rsk) in enumerate(top20[:20], 1):
            total_rows += 1
            if prod in ACTION_PROD_VALID.get(act, []) and rsk in PROD_RSK_VALID.get(prod, []):
                legal_rows += 1
            pred_rows.append({
                USER_COL: uid,
                "action_typ": act,
                "prod_typ": prod,
                "prod_sub_typ": sub,
                "rsk_lvl": rsk,
                "index": idx,
            })

    if save_artifacts and pred_rows:
        pred_df = pd.DataFrame(pred_rows)
        pred_path, target_path, compare_path = _write_holdout_artifacts(pred_df, eval_target, "ranker")
        print(f"本地 holdout 预测明细：{pred_path}")
        print(f"本地 holdout 真实标签：{target_path}")
        print(f"本地 holdout 对比明细：{compare_path}")

    return {
        "eval_users_with_target": float(len(scores)),
        "mean_ndcg20": float(np.mean(scores)) if scores else 0.0,
        "median_ndcg20": float(np.median(scores)) if scores else 0.0,
        "hit_recall20": float(recall_hits / recall_total) if recall_total else 0.0,
        "legal_rate": float(legal_rows / total_rows) if total_rows else 0.0,
        "lgb_valid_ndcg20": float(model.ranker.best_score_.get("valid_0", {}).get("ndcg@20", 0.0)) if model.ranker.best_score_ else 0.0,
    }


def parse_windows(mode: str) -> list[tuple[str, str, float]]:
    if mode == "tiny":
        return [
            ("2025-03-29 23:59:59", "2025-03-30", 1.5),
        ]
    if mode == "fast":
        return [
            ("2025-03-24 23:59:59", "2025-03-25", 1.0),
            ("2025-03-27 23:59:59", "2025-03-28", 1.3),
            ("2025-03-29 23:59:59", "2025-03-30", 1.6),
        ]
    return [
        ("2025-03-21 23:59:59", "2025-03-22", 1.0),
        ("2025-03-24 23:59:59", "2025-03-25", 1.1),
        ("2025-03-26 23:59:59", "2025-03-27", 1.2),
        ("2025-03-28 23:59:59", "2025-03-29", 1.5),
        ("2025-03-29 23:59:59", "2025-03-30", 1.7),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 holdout 评估：默认快速模式，预测 4/1。")
    parser.add_argument("--train-paths", nargs="*", type=Path, default=LOCAL_TRAIN_PATHS)
    parser.add_argument("--eval-user-ratio", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--max-train-users", type=int, default=5000, help="默认只抽样 5000 个训练用户；设为 0 表示全量。")
    parser.add_argument("--max-eval-users", type=int, default=2000, help="默认只抽样 2000 个验证用户；设为 0 表示全量。")
    parser.add_argument("--window-mode", choices=["tiny", "fast", "fullish"], default="fast")
    parser.add_argument("--model-mode", choices=["ranker", "repeat_first", "fast_light_ranker", "full_label"], default="full_label", help="本地快速比较不同小版本。")
    parser.add_argument("--candidate-k", type=int, default=100, help="fast_light_ranker 每用户候选数量。")
    parser.add_argument("--max-ranker-train-users", type=int, default=3000, help="fast_light_ranker 训练用户抽样数。")
    parser.add_argument("--no-save-artifacts", action="store_true", help="不保存本地 holdout 预测/真实/对比明细。")
    parser.add_argument("--no-april1-boost", action="store_true", help="关闭训练用户 4/1 监督增强，用于对照实验。")
    args = parser.parse_args()

    preload_users = 0
    if args.max_train_users and args.max_eval_users:
        preload_users = int((args.max_train_users + args.max_eval_users) * 1.5)
    elif args.max_train_users:
        preload_users = int(args.max_train_users * 2)
    elif args.max_eval_users:
        preload_users = int(args.max_eval_users * 3)
    all_df = load_local_train(args.train_paths, max_users=preload_users, seed=args.seed)
    users = np.array(sorted(all_df[USER_COL].unique()))
    train_users, eval_users = train_test_split(
        users,
        test_size=args.eval_user_ratio,
        random_state=args.seed,
        shuffle=True,
    )

    rng = np.random.default_rng(args.seed)
    if args.max_train_users and len(train_users) > args.max_train_users:
        train_users = rng.choice(train_users, size=args.max_train_users, replace=False)
    if args.max_eval_users and len(eval_users) > args.max_eval_users:
        eval_users = rng.choice(eval_users, size=args.max_eval_users, replace=False)

    windows = parse_windows(args.window_mode)

    print("========== 本地 4/1 Holdout 快速评估 ==========")
    print(f"训练文件：{[str(p) for p in args.train_paths]}")
    print(f"裁剪后总用户数：{len(users)}")
    print(f"实际训练用户数：{len(train_users)}")
    print(f"实际验证用户数：{len(eval_users)}")
    print(f"时间范围：{all_df[TIME_COL].min()} ~ {all_df[TIME_COL].max()}")
    print(f"窗口模式：{args.window_mode}")
    print(f"模型模式：{args.model_mode}")
    print(f"使用 4/1 监督增强：{not args.no_april1_boost}")

    if args.model_mode == "repeat_first":
        metrics = evaluate_repeat_first_holdout(
            all_df,
            set(train_users),
            set(eval_users),
            save_artifacts=not args.no_save_artifacts,
        )
    elif args.model_mode == "fast_light_ranker":
        metrics = evaluate_fast_light_ranker_holdout(
            all_df,
            set(train_users),
            set(eval_users),
            max_ranker_train_users=args.max_ranker_train_users,
            candidate_k=args.candidate_k,
            seed=args.seed,
            save_artifacts=not args.no_save_artifacts,
        )
    elif args.model_mode == "full_label":
        metrics = evaluate_full_label_holdout(
            all_df,
            set(train_users),
            set(eval_users),
            max_train_users=args.max_ranker_train_users,
            neg_per_user=args.candidate_k,
            seed=args.seed,
            save_artifacts=not args.no_save_artifacts,
        )
    else:
        metrics = evaluate_april1_holdout(
            all_df,
            set(train_users),
            set(eval_users),
            windows=windows,
            use_april1_boost=not args.no_april1_boost,
            save_artifacts=not args.no_save_artifacts,
        )

    print("\n========== 评估结果 ==========")
    print(f"有 4/1 真实标签的验证用户数：{int(metrics['eval_users_with_target'])}")
    print(f"Mean NDCG@20：{metrics['mean_ndcg20']:.6f}")
    print(f"Median NDCG@20：{metrics['median_ndcg20']:.6f}")
    print(f"Hit Recall@20：{metrics['hit_recall20']:.6f}")
    if "candidate_micro_recall" in metrics:
        print(f"Candidate Recall@K：{metrics['candidate_micro_recall']:.6f}")
        print(f"Candidate→Final Recall Gap：{metrics['candidate_to_final_gap']:.6f}")
        print(f"Zero-hit users：{int(metrics.get('zero_hit_users', 0))}")
        print(f"Full-hit users：{int(metrics.get('full_hit_users', 0))}")
    print(f"提交合法组合率：{metrics['legal_rate']:.6f}")
    print(f"LightGBM 内部 valid NDCG@20：{metrics['lgb_valid_ndcg20']:.6f}")


if __name__ == "__main__":
    main()
