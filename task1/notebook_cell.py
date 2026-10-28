"""
Jupyter 平台最终提交 Cell
========================

用法：
1. 把本文件内容复制到 Jupyter 一个 cell 中；或者在 cell 中 import/run 本文件。
2. 只需要修改 TRAIN_PATH 和 TEST_PATH。
3. 执行后会生成 /work/task1/prediction.csv。

前提：solution_ranker.py 与本文件同目录，平台会放在 /work/task1/*.py。
"""

from __future__ import annotations

import sys
from pathlib import Path


# =========================
# 只改这里：填写平台给你的真实路径
# =========================
TRAIN_PATH = Path("/work/data/task1/train_dataset.csv")   # TODO: 改成训练集路径
TEST_PATH = Path("/work/data/task1/test_dataset.csv")     # TODO: 改成测试集路径
OUT_PATH = Path("/work/task1/prediction.csv")


# =========================
# 下面不用改
# =========================
TASK1_DIR = Path("/work/task1")
if str(TASK1_DIR) not in sys.path:
    sys.path.insert(0, str(TASK1_DIR))

try:
    from solution_ranker import read_events, train_model, make_submission
except ModuleNotFoundError:
    # 兼容你在本地 notebook 从项目根目录运行的情况
    local_task1 = Path.cwd() / "task1"
    if local_task1.exists() and str(local_task1) not in sys.path:
        sys.path.insert(0, str(local_task1))
    from solution_ranker import read_events, train_model, make_submission


def check_path(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{name}不存在：{path}\n"
            f"请把 cell 顶部的 {name} 改成平台实际给出的文件路径。"
        )


check_path(TRAIN_PATH, "TRAIN_PATH")
check_path(TEST_PATH, "TEST_PATH")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("客户次日行为预测：开始运行")
print(f"训练集：{TRAIN_PATH}")
print(f"测试集：{TEST_PATH}")
print(f"输出文件：{OUT_PATH}")
print("=" * 80)

train_df = read_events(TRAIN_PATH, start="2025-03-01", end="2025-04-02")
test_df = read_events(TEST_PATH, start="2025-03-01", end="2025-04-02")

print(f"训练集记录数：{len(train_df):,}")
print(f"测试集记录数：{len(test_df):,}")
print(f"训练集用户数：{train_df['user_id'].nunique():,}")
print(f"测试集用户数：{test_df['user_id'].nunique():,}")

rank_model = train_model(train_df)
make_submission(rank_model, test_df, OUT_PATH)

print("=" * 80)
print("运行完成")
print(f"提交文件已生成：{OUT_PATH}")
print("=" * 80)
