from __future__ import annotations

import argparse
from pathlib import Path

from .full_label_fusion import fuse_prediction_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse full-label v1/v2 prediction files.")
    parser.add_argument("--v1", type=Path, default=Path("/work/task1/submit/prediction_v1.csv"))
    parser.add_argument("--v2", type=Path, default=Path("/work/task1/submit/prediction_v2.csv"))
    parser.add_argument("--out", type=Path, default=Path("/work/task1/submit/prediction.csv"))
    parser.add_argument("--v2-head", type=int, default=6)
    args = parser.parse_args()
    out = fuse_prediction_files(args.v1, args.v2, args.out, v2_head=args.v2_head)
    print(out)


if __name__ == "__main__":
    main()
