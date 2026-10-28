from __future__ import annotations
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from .config import RANDOM_STATE, USER_COL, TIME_COL, SUBMIT_DIR, MODEL_DIR, TASK_START, HISTORY_END, TARGET_START, TARGET_END
from .feature_engineering import (
    build_next_day_features,
    build_global_top_quads,
    build_quad_april1_count,
    build_training_cutoff_features,
    COLS,
    _fill_missing_values,
    PROD_RSK_VALID,
    ACTION_PROD_VALID,
)
from .modeling import train_next_day_ranker, generate_candidates, rank_candidates


SUBMISSION_COLUMNS = ["user_id", "action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl", "index"]
MAX_TRAIN_USERS = 30000
CHECKPOINT_PATH = MODEL_DIR / "task1_ranker_checkpoint.pkl"
PREDICT_READY_CHECKPOINT_PATH = MODEL_DIR / "task1_predict_ready_checkpoint.pkl"


def load_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载并清洗数据。"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df[TIME_COL] = pd.to_datetime(train_df[TIME_COL], errors="coerce")
    test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL], errors="coerce")

    task_start = pd.Timestamp(TASK_START)
    history_end = pd.Timestamp(HISTORY_END)
    target_end = pd.Timestamp(TARGET_END)
    train_df = train_df[(train_df[TIME_COL] >= task_start) & (train_df[TIME_COL] < target_end)].copy()
    test_df = test_df[(test_df[TIME_COL] >= task_start) & (test_df[TIME_COL] <= history_end)].copy()

    train_df = _fill_missing_values(train_df)
    test_df = _fill_missing_values(test_df)

    train_df = train_df.dropna(subset=COLS + [USER_COL, TIME_COL]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[USER_COL, TIME_COL]).reset_index(drop=True)
    print(f"训练集裁剪后时间范围：{train_df[TIME_COL].min()} ~ {train_df[TIME_COL].max()}，记录数 {len(train_df)}")
    print(f"测试集裁剪后时间范围：{test_df[TIME_COL].min()} ~ {test_df[TIME_COL].max()}，记录数 {len(test_df)}")
    return train_df, test_df


def _quad_to_row(user_row: pd.Series, quad: tuple[str, str, str, str], april1_count: dict[str, int] | None = None) -> dict:
    quad_str = "|".join(quad)
    return { 
        "quad_count_all": user_row["quad_count_all"].get(quad_str, 0),
        "quad_count_3d": user_row["quad_count_3d"].get(quad_str, 0),
        "quad_count_7d": user_row["quad_count_7d"].get(quad_str, 0),
        "quad_weighted_count": user_row["quad_weighted_count"].get(quad_str, 0),
        "quad_april1_count": 0 if april1_count is None else april1_count.get(quad_str, 0),
        "act_prod_count": user_row.get("act_prod_count", {}).get((quad[0], quad[1]), 0),
        "prod_sub_count": user_row.get("prod_sub_count", {}).get((quad[1], quad[2]), 0),
        "rsk_count": user_row.get("rsk_count", {}).get((quad[1], quad[3]), 0),
        "recent_act_switch": user_row.get("recent_act_switch", 0),
        "recent_prod_switch": user_row.get("recent_prod_switch", 0),
        "recent_sub_switch": user_row.get("recent_sub_switch", 0),
        "recent_rsk_switch": user_row.get("recent_rsk_switch", 0),
        "act_unique_7d": user_row.get("act_unique_7d", 0),
        "prod_unique_7d": user_row.get("prod_unique_7d", 0),
        "sub_unique_7d": user_row.get("sub_unique_7d", 0),
        "act_match_last_1d": 1 if quad[0] == user_row["last_1d_act"] else 0,
        "prod_match_last_1d": 1 if quad[1] == user_row["last_1d_prod"] else 0,
        "sub_match_last_1d": 1 if quad[2] == user_row["last_1d_sub"] else 0,
        "act_match_last_1d_mode": 1 if quad[0] == user_row["last_1d_act_mode"] else 0,
        "prod_match_last_1d_mode": 1 if quad[1] == user_row["last_1d_prod_mode"] else 0,
        "act_match_last_3d_mode": 1 if quad[0] == user_row["last_3d_act_mode"] else 0,
        "prod_match_last_3d_mode": 1 if quad[1] == user_row["last_3d_prod_mode"] else 0,
        "last_act_hours_ago": user_row["last_act_hours_ago"],
        "cnt_last_1d": user_row["cnt_last_1d"],
        "last_week_ratio": user_row["last_week_ratio"],
        "rsk_mean": user_row["rsk_mean"],
        "rsk_std": user_row.get("rsk_std", 0),
        "rsk_diff": abs(int(quad[3][1]) - user_row["rsk_mean"]),
        "action_typ": quad[0],
        "prod_typ": quad[1],
    }


def build_rank_samples(
    train_df: pd.DataFrame,
    cutoff_date: str,
    target_date: str,
    april1_count: dict[str, int] | None = None,
    sample_weight: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray, list[int], np.ndarray]:
    """构建单个 next-day 排序样本集合。"""
    cutoff_ts = pd.Timestamp(cutoff_date)
    target_ts = pd.Timestamp(target_date)

    user_feats = build_next_day_features(train_df, cutoff_ts)
    target_df = train_df[(train_df[TIME_COL] >= target_ts) & (train_df[TIME_COL] < target_ts + pd.Timedelta(days=1))].copy()
    target_df = _fill_missing_values(target_df)
    target_df["quadruple"] = target_df[COLS].astype(str).agg("|".join, axis=1)
    target_map = target_df.groupby(USER_COL)["quadruple"].apply(lambda s: set(s.tolist())).to_dict()

    rows = []
    labels = []
    groups = []
    weights = []

    for _, user_row in user_feats.iterrows():
        uid = user_row[USER_COL]
        user_targets = target_map.get(uid, set())
        candidates = generate_candidates(user_row, [], [])
        if not candidates:
            continue
        for quad in candidates:
            quad_str = "|".join(quad)
            rows.append(_quad_to_row(user_row, quad, april1_count))
            labels.append(1 if quad_str in user_targets else 0)
            weights.append(sample_weight)
        groups.append(len(candidates))

    return pd.DataFrame(rows), np.array(labels), groups, np.array(weights, dtype=float)


def sample_training_users(train_df: pd.DataFrame, max_users: int = MAX_TRAIN_USERS) -> pd.DataFrame:
    """抽样训练用户用于构造排序样本，最终预测仍使用全量测试用户。

    全量训练用户逐窗口构造复杂特征会非常慢且容易 OOM。这里优先保留 4/1
    有真实行为的用户，再补充近 7 天活跃用户，使训练分布尽量贴近最终任务。
    """
    all_users = pd.Index(train_df[USER_COL].dropna().unique())
    if len(all_users) <= max_users:
        print(f"训练用户数 {len(all_users)} <= {max_users}，使用全量训练用户构造样本。")
        return train_df

    april1_users = pd.Index(train_df.loc[train_df[TIME_COL] >= pd.Timestamp("2025-04-01"), USER_COL].dropna().unique())
    recent_users = pd.Index(
        train_df.loc[
            (train_df[TIME_COL] >= pd.Timestamp("2025-03-24"))
            & (train_df[TIME_COL] < pd.Timestamp("2025-04-02")),
            USER_COL,
        ].dropna().unique()
    )

    rng = np.random.default_rng(RANDOM_STATE)
    selected: list[str] = []
    selected_set: set[str] = set()

    def add_users(users: pd.Index | np.ndarray, limit: int) -> None:
        nonlocal selected, selected_set
        candidates = [u for u in users.tolist() if u not in selected_set]
        if len(candidates) > limit:
            candidates = rng.choice(np.array(candidates, dtype=object), size=limit, replace=False).tolist()
        selected.extend(candidates)
        selected_set.update(candidates)

    add_users(april1_users, min(len(april1_users), max_users // 2))
    add_users(recent_users, max_users - len(selected))
    if len(selected) < max_users:
        add_users(all_users, max_users - len(selected))

    sampled_df = train_df[train_df[USER_COL].isin(selected_set)].copy()
    print(
        f"训练用户抽样：全量 {len(all_users)} -> 抽样 {len(selected_set)}；"
        f"4/1用户候选 {len(april1_users)}；近日期用户候选 {len(recent_users)}；"
        f"训练记录 {len(train_df)} -> {len(sampled_df)}"
    )
    return sampled_df


def build_training_samples(train_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[int], pd.DataFrame, np.ndarray, list[int], np.ndarray, np.ndarray]:
    """构造最终提交使用的轻量训练集，并显式利用 4/1 的监督信号。

    任务目标是用测试集截至 3/31 的历史预测 4/1，因此正式训练优先保留靠近
    4/1 的 next-day 窗口，避免 3/7~3/29 全窗口在云端造成 OOM。最终预测仍
    覆盖全部测试用户，每个用户输出 20 条。
    """
    train_X_list = []
    train_y_list = []
    train_groups = []
    train_w_list = []

    # fast-final 主训练段：近日期稀疏窗口，已在 local_eval fast 中验证过。
    # 保留趋势信息，同时把训练样本量控制在 full 的约 1/5~1/6。
    fast_windows = [
        ("2025-03-24 23:59:59", "2025-03-25", 1.0),
        ("2025-03-27 23:59:59", "2025-03-28", 1.3),
        ("2025-03-29 23:59:59", "2025-03-30", 1.6),
    ]
    for cutoff_date, target_date, w in fast_windows:
        print(f"构造训练窗口：{cutoff_date} -> {target_date}，权重 {w}")
        X, y, groups, weights = build_rank_samples(train_df, cutoff_date, target_date, sample_weight=w)
        if len(X) == 0:
            print(f"跳过空训练窗口：{cutoff_date} -> {target_date}")
            continue
        train_X_list.append(X)
        train_y_list.append(y)
        train_groups.extend(groups)
        train_w_list.append(weights)

    # 4/1 增强段：显式利用“训练集多出的一天”
    april1_feats = build_training_cutoff_features(train_df, HISTORY_END, TARGET_START)
    april1_target_df = train_df[
        (train_df[TIME_COL] >= pd.Timestamp(TARGET_START))
        & (train_df[TIME_COL] < pd.Timestamp(TARGET_END))
    ].copy()
    april1_target_df = _fill_missing_values(april1_target_df)
    april1_target_df["quadruple"] = april1_target_df[COLS].astype(str).agg("|".join, axis=1)
    april1_target_map = april1_target_df.groupby(USER_COL)["quadruple"].apply(lambda s: set(s.tolist())).to_dict()

    april1_rows = []
    april1_labels = []
    april1_groups = []
    april1_weights = []
    print("构造训练增强窗口：2025-03-31 23:59:59 -> 2025-04-01，权重 2.0")
    for _, user_row in april1_feats.iterrows():
        uid = user_row[USER_COL]
        user_targets = april1_target_map.get(uid, set())
        candidates = generate_candidates(user_row, [], [])
        if not candidates:
            continue
        for quad in candidates:
            quad_str = "|".join(quad)
            april1_rows.append(_quad_to_row(user_row, quad, None))
            april1_labels.append(1 if quad_str in user_targets else 0)
            april1_weights.append(2.0)
        april1_groups.append(len(candidates))

    if not train_X_list and not april1_rows:
        raise RuntimeError("没有构造出训练样本，请检查日期窗口和数据。")

    if train_X_list:
        train_X = pd.concat(train_X_list + [pd.DataFrame(april1_rows)], ignore_index=True)
        y_train = np.concatenate(train_y_list + [np.array(april1_labels, dtype=int)])
        groups_train = train_groups + april1_groups
        sample_weight = np.concatenate(train_w_list + [np.array(april1_weights, dtype=float)])
    else:
        train_X = pd.DataFrame(april1_rows)
        y_train = np.array(april1_labels, dtype=int)
        groups_train = april1_groups
        sample_weight = np.array(april1_weights, dtype=float)

    # 验证集：3/30 -> 3/31，用于近似测试分布
    val_X, val_y, val_groups, val_weight = build_rank_samples(
        train_df,
        "2025-03-30 23:59:59",
        "2025-03-31",
        april1_count=None,
        sample_weight=1.0,
    )

    return train_X, y_train, groups_train, val_X, val_y, val_groups, sample_weight, val_weight


def validate_submission(submission_df: pd.DataFrame) -> list[str]:
    """题目要求自检。"""
    issues = []
    if list(submission_df.columns) != SUBMISSION_COLUMNS:
        issues.append("列顺序不符合题目要求")
    if submission_df.isna().any().any():
        issues.append("存在空值")
    if not all(submission_df.groupby("user_id").size() == 20):
        issues.append("不是每个 user_id 恰好 20 条")
    if set(submission_df["index"].unique()) - set(range(1, 21)):
        issues.append("index 存在 1~20 之外的值")
    bad_index = submission_df.groupby("user_id")["index"].nunique().ne(20).any()
    if bad_index:
        issues.append("同一 user_id 的 index 不唯一或数量不足 20")
    if not submission_df.apply(lambda x: x["prod_typ"] in ACTION_PROD_VALID.get(x["action_typ"], []), axis=1).all():
        issues.append("存在非法 action_typ-prod_typ 组合")
    if not submission_df.apply(lambda x: x["rsk_lvl"] in PROD_RSK_VALID.get(x["prod_typ"], []), axis=1).all():
        issues.append("存在非法 prod_typ-rsk_lvl 组合")
    return issues


def _save_predict_ready_checkpoint(model, test_user_feats: pd.DataFrame, test_users: list) -> None:
    """保存预测前 checkpoint，避免预测阶段失败后重新构建测试特征。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "test_user_feats": test_user_feats,
        "test_users": test_users,
    }
    with open(PREDICT_READY_CHECKPOINT_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"💾 预测前 checkpoint 已保存：{PREDICT_READY_CHECKPOINT_PATH}")


def _predict_and_save_submission_from_features(model, test_user_feats: pd.DataFrame, test_users: list) -> Path:
    """使用已构建好的测试用户特征预测并保存提交文件。"""
    print("🔮 预测测试集次日行为...")
    global_top_quads = model.global_top_quads
    april1_top_quads = model.april1_top_quads
    if not april1_top_quads:
        april1_top_quads = global_top_quads[:]
    if not april1_top_quads:
        raise RuntimeError("模型缺少可用于补齐 top20 的全局/4月1日热门四元组。")

    user_feat_map = {row[USER_COL]: row for _, row in test_user_feats.iterrows()}
    submission_rows = []
    for n, uid in enumerate(test_users, 1):
        user_feat = user_feat_map.get(uid)
        if user_feat is None:
            top20 = april1_top_quads[:20]
    else:
            candidates = generate_candidates(user_feat, global_top_quads, april1_top_quads)
            top20 = rank_candidates(model, user_feat, candidates)

        if n % 10000 == 0:
            print(f"  已预测 {n}/{len(test_users)} 个用户")

        seen = set(top20)
        idx = 0
        while len(top20) < 20 and idx < len(april1_top_quads):
            q = april1_top_quads[idx]
            if q not in seen:
                seen.add(q)
                top20.append(q)
            idx += 1
        while len(top20) < 20:
            top20.append(april1_top_quads[0])

        for i, (act, prod, sub, rsk) in enumerate(top20[:20], 1):
            submission_rows.append({
                "user_id": uid,
                "action_typ": act,
                "prod_typ": prod,
                "prod_sub_typ": sub,
                "rsk_lvl": rsk,
                "index": i,
            })

    submission_df = pd.DataFrame(submission_rows)[SUBMISSION_COLUMNS]
    out_path = SUBMIT_DIR / "prediction.csv"
    submission_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n✅ 提交文件生成完成，校验结果：")
    print(f"📊 总预测行数：{len(submission_df)}")
    print(f"👥 预测用户数：{len(test_users)}")
    print(f"🔍 每个用户严格20条：{all(submission_df.groupby('user_id').size() == 20)}")
    issues = validate_submission(submission_df)
    print(f"🧪 自检问题数：{len(issues)}")
    for issue in issues:
        print(f"- {issue}")
    print(f"📄 文件路径：{out_path}")
    return out_path


def _predict_and_save_submission(model, test_df: pd.DataFrame) -> Path:
    """使用已训练模型预测全量测试用户并保存提交文件。"""
    print("📊 构建测试集用户特征...")
    test_end_date = pd.Timestamp(HISTORY_END)
    test_user_feats = build_next_day_features(test_df, test_end_date)
    test_users = test_df[USER_COL].unique().tolist()
    _save_predict_ready_checkpoint(model, test_user_feats, test_users)
    return _predict_and_save_submission_from_features(model, test_user_feats, test_users)


def train_and_checkpoint(train_df: pd.DataFrame):
    """训练排序模型并立即保存 checkpoint。"""
    print("🌍 生成全局统计信息...")
    global_top_quads = build_global_top_quads(train_df, top_n=100)
    april1_df = train_df[
        (train_df[TIME_COL] >= pd.Timestamp(TARGET_START))
        & (train_df[TIME_COL] < pd.Timestamp(TARGET_END))
    ].copy()
    april1_top_quads = build_global_top_quads(april1_df, top_n=100) if not april1_df.empty else global_top_quads[:]
    if not april1_top_quads:
        april1_top_quads = global_top_quads[:]
    quad_april1_count = build_quad_april1_count(train_df)
    print(f"✅ 4月1日共出现 {len(quad_april1_count)} 种不同的四元组")

    train_sample_df = sample_training_users(train_df)

    print("🏋️ 构造训练集和验证集（sampled fast-final 时间切分 + 4/1增强）...")
    X_train, y_train, groups_train, X_val, y_val, groups_val, train_w, val_w = build_training_samples(train_sample_df)
    print(f"训练样本量：{len(X_train)}，验证样本量：{len(X_val)}")
    print(f"训练 query 数：{len(groups_train)}，验证 query 数：{len(groups_val)}")
    print(f"训练正样本数：{int(y_train.sum())}，验证正样本数：{int(y_val.sum())}")
    print(f"训练特征内存约：{X_train.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    print("🤖 训练排序模型...")
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 模型 checkpoint 已保存：{CHECKPOINT_PATH}")

    print(f"✅ 模型训练完成，最佳迭代次数：{model.ranker.best_iteration_}")
    if model.ranker.best_score_:
        valid_scores = model.ranker.best_score_.get("valid_0", {})
        if valid_scores:
            metric_text = ", ".join(f"{name}={score:.4f}" for name, score in valid_scores.items())
            print(f"📊 本地验证指标：{metric_text}")
        else:
            print(f"📊 本地验证指标：{model.ranker.best_score_}")
    return model


def run_task1(train_path: Path, test_path: Path) -> Path:
    """完整运行流程。"""
    print("📥 加载训练集和测试集...")
    train_df, test_df = load_data(train_path, test_path)
    model = train_and_checkpoint(train_df)
    return _predict_and_save_submission(model, test_df)


def run_task1_from_checkpoint(test_path: Path, checkpoint_path: Path = CHECKPOINT_PATH) -> Path:
    """从已保存模型 checkpoint 继续预测，避免重复训练。"""
    print(f"📦 加载模型 checkpoint：{checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        model = pickle.load(f)

    print("📥 加载测试集...")
    test_df = pd.read_csv(test_path)
    test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL], errors="coerce")
    test_df = test_df[
        (test_df[TIME_COL] >= pd.Timestamp(TASK_START))
        & (test_df[TIME_COL] <= pd.Timestamp(HISTORY_END))
    ].copy()
    test_df = _fill_missing_values(test_df)
    test_df = test_df.dropna(subset=[USER_COL, TIME_COL]).reset_index(drop=True)
    return _predict_and_save_submission(model, test_df)


def run_task1_from_predict_ready_checkpoint(checkpoint_path: Path = PREDICT_READY_CHECKPOINT_PATH) -> Path:
    """从预测前 checkpoint 继续，避免重复加载测试集和构建测试用户特征。"""
    print(f"📦 加载预测前 checkpoint：{checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"预测前 checkpoint 不存在：{checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        payload = pickle.load(f)
    return _predict_and_save_submission_from_features(
        payload["model"],
        payload["test_user_feats"],
        payload["test_users"],
    )


def find_exam_paths() -> tuple[Path, Path]:
    """自动探测离线环境中的训练/测试文件。"""
    data_dir = Path("/work/data/task1")
    train_candidates = [
        data_dir / "dataset_train.csv",
        data_dir / "train_dataset.csv",
        Path("/work/task1/dataset_train.csv"),
        Path("/work/task1/train_dataset.csv"),
    ]
    test_candidates = [
        data_dir / "dataset_test.csv",
        data_dir / "test_dataset.csv",
        Path("/work/task1/dataset_test.csv"),
        Path("/work/task1/test_dataset.csv"),
    ]

    train_path = next((path for path in train_candidates if path.exists()), None)
    test_path = next((path for path in test_candidates if path.exists()), None)
    if train_path is None:
        raise FileNotFoundError(f"找不到训练文件，候选路径：{[str(path) for path in train_candidates]}")
    if test_path is None:
        raise FileNotFoundError(f"找不到测试文件，候选路径：{[str(path) for path in test_candidates]}")
    return train_path, test_path


if __name__ == "__main__":
    TRAIN_PATH, TEST_PATH = find_exam_paths()
    print(f"使用训练集文件：{TRAIN_PATH}")
    print(f"使用测试集文件：{TEST_PATH}")
    if os.environ.get("TASK1_USE_CHECKPOINT", "0") == "1":
        run_task1_from_checkpoint(TEST_PATH)
    else:
        run_task1(TRAIN_PATH, TEST_PATH)
