from __future__ import annotations

from pathlib import Path

import pandas as pd

from .make_fusion_candidates import splice_fuse, summarize
from .make_nonwealth_empty import make_nonwealth_empty


def main() -> None:
    d = Path("/work/task1/submit")
    out = d / "fusion_candidates"
    out.mkdir(parents=True, exist_ok=True)
    v1 = d / "prediction_v1_no_prior.csv"
    v2 = d / "prediction_v2_no_prior.csv"
    recent = d / "prediction_recent.csv"
    specs = [
        ("splice_v1_14_recent_4_v2_2.csv", [(v1, 14), (recent, 4), (v2, 2)]),
        ("splice_v1_16_recent_2_v2_2.csv", [(v1, 16), (recent, 2), (v2, 2)]),
        ("splice_v1_12_recent_4_v2_4.csv", [(v1, 12), (recent, 4), (v2, 4)]),
        ("splice_v1_15_recent_3_v2_2.csv", [(v1, 15), (recent, 3), (v2, 2)]),
    ]
    paths = []
    empty_paths = []
    for name, sources in specs:
        path = splice_fuse(sources, out / name, [v1, v2, recent])
        paths.append(path)
        empty_path = out / name.replace(".csv", "_nonwealth_empty.csv")
        empty_paths.append(make_nonwealth_empty(path, empty_path))
    summary = pd.DataFrame([summarize(p) for p in paths])
    summary.to_csv(out / "candidate_summary_low_nonwealth.csv", index=False, encoding="utf-8-sig")
    empty_summary = pd.DataFrame([summarize(p) for p in empty_paths])
    empty_summary.to_csv(out / "candidate_summary_low_nonwealth_empty.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(empty_summary.to_string(index=False))


if __name__ == "__main__":
    main()
