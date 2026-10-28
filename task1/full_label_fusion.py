from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import SUBMIT_DIR
from .diagnostic_outputs import write_prediction_distribution, write_prediction_format_report
from .full_label_classifier import run_task1_full_label_classifier

COLS = ["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]
SUBMISSION_COLUMNS = ["user_id", *COLS, "index"]


def _add_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quad_key"] = df["action_typ"].astype(str).str.cat(
        [df["prod_typ"].astype(str), df["prod_sub_typ"].astype(str), df["rsk_lvl"].astype(str)], sep="|"
    )
    return df


def fuse_prediction_files(v1_path: Path, v2_path: Path, output_path: Path, v2_head: int = 6) -> Path:
    v1 = _add_key(pd.read_csv(v1_path))
    v2 = _add_key(pd.read_csv(v2_path))
    users = v2["user_id"].drop_duplicates().tolist()
    v1_groups = {uid: g.sort_values("index") for uid, g in v1.groupby("user_id", sort=False)}
    v2_groups = {uid: g.sort_values("index") for uid, g in v2.groupby("user_id", sort=False)}
    rows = []
    for uid in users:
        selected = []
        selected_keys = set()
        g2 = v2_groups.get(uid)
        if g2 is not None:
            for r in g2.itertuples(index=False):
                if len(selected) >= v2_head:
                    break
                if r.quad_key in selected_keys:
                    continue
                selected.append(r)
                selected_keys.add(r.quad_key)
        g1 = v1_groups.get(uid)
        if g1 is not None:
            for r in g1.itertuples(index=False):
                if len(selected) >= 20:
                    break
                if r.quad_key in selected_keys:
                    continue
                selected.append(r)
                selected_keys.add(r.quad_key)
        if len(selected) < 20 and g2 is not None:
            for r in g2.itertuples(index=False):
                if len(selected) >= 20:
                    break
                if r.quad_key in selected_keys:
                    continue
                selected.append(r)
                selected_keys.add(r.quad_key)
        for idx, r in enumerate(selected[:20], 1):
            rows.append(
                {
                    "user_id": uid,
                    "action_typ": r.action_typ,
                    "prod_typ": r.prod_typ,
                    "prod_sub_typ": r.prod_sub_typ,
                    "rsk_lvl": r.rsk_lvl,
                    "index": idx,
                }
            )
    pred = pd.DataFrame(rows)[SUBMISSION_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"融合提交文件已保存：{output_path}")
    print(f"格式报告：{write_prediction_format_report(pred, output_path.parent, prefix=output_path.stem)}")
    print(f"分布报告：{write_prediction_distribution(pred, output_path.parent, prefix=output_path.stem)}")
    return output_path


def run_full_label_fusion(
    train_path: Path,
    test_path: Path,
    max_train_users: int = 30000,
    neg_per_user: int = 150,
    v2_head: int = 6,
    seed: int = 42,
    multi_window: bool = False,
    n_jobs: int = -1,
    use_target_prior: bool = True,
    use_global_hot_negatives: bool = True,
) -> Path:
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    v1_path = run_task1_full_label_classifier(
        train_path,
        test_path,
        output_name="prediction_v1.csv",
        max_train_users=max_train_users,
        neg_per_user=neg_per_user,
        seed=seed,
        variant="v1",
        multi_window=multi_window,
        n_jobs=n_jobs,
        use_target_prior=use_target_prior,
        use_global_hot_negatives=use_global_hot_negatives,
    )
    v2_path = run_task1_full_label_classifier(
        train_path,
        test_path,
        output_name="prediction_v2.csv",
        max_train_users=max_train_users,
        neg_per_user=neg_per_user,
        seed=seed,
        variant="v2",
        multi_window=multi_window,
        n_jobs=n_jobs,
    )
    return fuse_prediction_files(v1_path, v2_path, SUBMIT_DIR / "prediction.csv", v2_head=v2_head)


if __name__ == "__main__":
    run_full_label_fusion(
        Path("/work/data/task1/dataset_train.csv"),
        Path("/work/data/task1/dataset_test.csv"),
        max_train_users=30000,
        neg_per_user=150,
        v2_head=6,
        seed=42,
    )
