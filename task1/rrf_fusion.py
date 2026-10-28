from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .diagnostic_outputs import write_prediction_distribution, write_prediction_format_report

COLS = ["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]
SUBMISSION_COLUMNS = ["user_id", *COLS, "index"]


def add_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quad_key"] = df["action_typ"].astype(str).str.cat(
        [df["prod_typ"].astype(str), df["prod_sub_typ"].astype(str), df["rsk_lvl"].astype(str)], sep="|"
    )
    return df


def split_quad(q: str) -> tuple[str, str, str, str]:
    parts = q.split("|")
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3]
    return "浏览详情", "理财", "短债", "R2"


def rrf_fuse_files(
    input_paths: list[Path],
    output_path: Path,
    weights: list[float] | None = None,
    rrf_k: float = 20.0,
    top_k: int = 20,
) -> Path:
    if not input_paths:
        raise ValueError("至少需要一个输入预测文件。")
    weights = weights or [1.0] * len(input_paths)
    if len(weights) != len(input_paths):
        raise ValueError("weights 数量必须和 input_paths 一致。")

    model_groups = []
    users: list[str] = []
    for path in input_paths:
        df = add_key(pd.read_csv(path)).sort_values(["user_id", "index"])
        model_groups.append({str(uid): g for uid, g in df.groupby("user_id", sort=False)})
        if not users:
            users = [str(x) for x in df["user_id"].drop_duplicates().tolist()]

    rows = []
    for uid in users:
        scores: defaultdict[str, float] = defaultdict(float)
        for groups, weight in zip(model_groups, weights, strict=False):
            g = groups.get(uid)
            if g is None:
                continue
            for rank, q in enumerate(g["quad_key"].tolist(), 1):
                scores[q] += float(weight) / (rrf_k + rank)
        selected = [q for q, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        for idx, q in enumerate(selected, 1):
            action, prod, sub, rsk = split_quad(q)
            rows.append(
                {
                    "user_id": uid,
                    "action_typ": action,
                    "prod_typ": prod,
                    "prod_sub_typ": sub,
                    "rsk_lvl": rsk,
                    "index": idx,
                }
            )
    pred = pd.DataFrame(rows)[SUBMISSION_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"RRF 融合文件已保存：{output_path}")
    print(f"格式报告：{write_prediction_format_report(pred, output_path.parent, prefix=output_path.stem)}")
    print(f"分布报告：{write_prediction_distribution(pred, output_path.parent, prefix=output_path.stem)}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RRF fuse multiple prediction files.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path("/work/task1/submit/prediction.csv"))
    parser.add_argument("--rrf-k", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rrf_fuse_files(args.inputs, args.out, args.weights, args.rrf_k)


if __name__ == "__main__":
    main()
