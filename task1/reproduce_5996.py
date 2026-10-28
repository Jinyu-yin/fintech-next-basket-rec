from __future__ import annotations

import argparse
from pathlib import Path

from .make_fusion_candidates import splice_fuse
from .make_nonwealth_empty import make_nonwealth_empty
from .recent_repeat_model import run_recent_repeat_model
from .full_label_classifier import run_task1_full_label_classifier


def reproduce_5996(
    train_path: Path,
    test_path: Path,
    submit_dir: Path,
    max_train_users: int,
    neg_per_user: int,
    n_jobs: int,
    seed: int,
) -> Path:
    submit_dir.mkdir(parents=True, exist_ok=True)

    v1_path = run_task1_full_label_classifier(
        train_path,
        test_path,
        output_name="prediction_v1_no_prior.csv",
        max_train_users=max_train_users,
        neg_per_user=neg_per_user,
        seed=seed,
        variant="v1",
        multi_window=False,
        n_jobs=n_jobs,
        use_target_prior=False,
        use_global_hot_negatives=False,
    )
    v2_path = run_task1_full_label_classifier(
        train_path,
        test_path,
        output_name="prediction_v2_no_prior.csv",
        max_train_users=max_train_users,
        neg_per_user=neg_per_user,
        seed=seed,
        variant="v2",
        multi_window=False,
        n_jobs=n_jobs,
        use_target_prior=False,
        use_global_hot_negatives=False,
    )
    recent_path = run_recent_repeat_model(train_path, test_path, "prediction_recent.csv")

    raw_splice = submit_dir / "splice_v1_16_recent_2_v2_2.csv"
    splice_fuse(
        [(v1_path, 16), (recent_path, 2), (v2_path, 2)],
        raw_splice,
        [v1_path, v2_path, recent_path],
    )

    final_path = submit_dir / "prediction.csv"
    make_nonwealth_empty(raw_splice, final_path)
    print(f"5996 reproduction file saved: {final_path}")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the 0.5996 submission pipeline.")
    parser.add_argument("--train-path", type=Path, default=Path("/work/data/task1/dataset_train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("/work/data/task1/dataset_test.csv"))
    parser.add_argument("--submit-dir", type=Path, default=Path("/work/task1/submit"))
    parser.add_argument("--max-train-users", type=int, default=30000)
    parser.add_argument("--neg-per-user", type=int, default=150)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reproduce_5996(
        args.train_path,
        args.test_path,
        args.submit_dir,
        args.max_train_users,
        args.neg_per_user,
        args.n_jobs,
        args.seed,
    )


if __name__ == "__main__":
    main()
