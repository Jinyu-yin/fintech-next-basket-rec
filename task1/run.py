from __future__ import annotations

import sys
from pathlib import Path

# 离线平台直接执行 /work/task1/run.py 时，包相对导入可能失效。
# 显式把 /work 加入 sys.path，保证可以 import task1.pipeline。
WORK_DIR = Path("/work")
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

from task1.full_label_classifier import run_task1_full_label_classifier


def _first_existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            print(f"使用{label}文件：{path}")
            return path
    raise FileNotFoundError(f"找不到{label}文件，候选路径：{[str(p) for p in candidates]}")


if __name__ == "__main__":
    data_dir = Path("/work/data/task1")
    train_path = _first_existing(
        [
            data_dir / "dataset_train.csv",
            data_dir / "train_dataset.csv",
            Path("/work/task1/dataset_train.csv"),
            Path("/work/task1/train_dataset.csv"),
        ],
        "训练集",
    )
    test_path = _first_existing(
        [
            data_dir / "dataset_test.csv",
            data_dir / "test_dataset.csv",
            Path("/work/task1/dataset_test.csv"),
            Path("/work/task1/test_dataset.csv"),
        ],
        "测试集",
    )
    out_path = run_task1_full_label_classifier(train_path, test_path, output_name="prediction.csv")
    print(f"saved to {out_path}")
