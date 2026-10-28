from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
EXAM_DIR = Path("/work/data/task1")
LOCAL_TRAIN_PATHS = [PROJECT_DIR / "train1.csv", PROJECT_DIR / "train2.csv", PROJECT_DIR / "train3.csv"]
LOCAL_TEST_PATH = PROJECT_DIR / "clipboard_dump.csv"
EXAM_TRAIN_PATH = EXAM_DIR / "dataset_train.csv"
EXAM_TEST_PATH = EXAM_DIR / "dataset_test.csv"

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "features"
MODEL_DIR = BASE_DIR / "models"
SUBMIT_DIR = BASE_DIR / "submit"

RANDOM_STATE = 42
N_SPLITS = 5
TARGET_COLS = ["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]
TIME_COL = "acs_tm"
USER_COL = "user_id"

# 题目时间窗口：原始数据日期范围较广，正式训练/预测只使用该窗口。
TASK_START = "2025-03-01 00:00:00"
HISTORY_END = "2025-03-31 23:59:59"
TARGET_START = "2025-04-01 00:00:00"
TARGET_END = "2025-04-02 00:00:00"

for _dir in [RAW_DIR, PROCESSED_DIR, FEATURE_DIR, MODEL_DIR, SUBMIT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
