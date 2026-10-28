from __future__ import annotations

import argparse
from pathlib import Path

from .full_label_classifier import run_task1_full_label_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one full-label variant for parallel training.")
    parser.add_argument("--train-path", type=Path, default=Path("/work/data/task1/dataset_train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("/work/data/task1/dataset_test.csv"))
    parser.add_argument("--variant", choices=["v1", "v2"], required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-train-users", type=int, default=30000)
    parser.add_argument("--neg-per-user", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-window", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-target-prior", action="store_true")
    parser.add_argument("--no-global-hot-negatives", action="store_true")
    args = parser.parse_args()

    output = args.output or f"prediction_{args.variant}.csv"
    out = run_task1_full_label_classifier(
        args.train_path,
        args.test_path,
        output_name=output,
        max_train_users=args.max_train_users,
        neg_per_user=args.neg_per_user,
        seed=args.seed,
        variant=args.variant,
        multi_window=args.multi_window,
        n_jobs=args.n_jobs,
        use_target_prior=not args.no_target_prior,
        use_global_hot_negatives=not args.no_global_hot_negatives,
    )
    print(out)


if __name__ == "__main__":
    main()
