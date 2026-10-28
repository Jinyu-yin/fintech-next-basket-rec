from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def make_nonwealth_empty(input_path: Path, output_path: Path) -> Path:
    df = pd.read_csv(input_path)
    mask = df["prod_typ"].astype(str).eq("非财富")
    df.loc[mask, "prod_sub_typ"] = ""
    df.loc[mask, "rsk_lvl"] = ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"rows: {len(df)}")
    print(f"nonwealth rows emptied: {int(mask.sum())}")
    print(df[mask].head(20).to_string(index=False))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Set prod_sub_typ/rsk_lvl empty for nonwealth rows.")
    parser.add_argument("--input", type=Path, default=Path("/work/task1/submit/prediction.csv"))
    parser.add_argument("--out", type=Path, default=Path("/work/task1/submit/prediction_nonwealth_empty.csv"))
    args = parser.parse_args()
    make_nonwealth_empty(args.input, args.out)


if __name__ == "__main__":
    main()
