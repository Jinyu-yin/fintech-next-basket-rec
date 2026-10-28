from __future__ import annotations

from pathlib import Path

import pandas as pd

from .make_fusion_candidates import splice_fuse, summarize


def main() -> None:
    d = Path("/work/task1/submit")
    out = d / "fusion_candidates_conservative"
    out.mkdir(parents=True, exist_ok=True)
    v1 = d / "prediction_v1_no_prior.csv"
    v2 = d / "prediction_v2_no_prior.csv"
    recent = d / "prediction_recent_conservative.csv"
    specs = [
        ("splice_v1_17_rc_1_v2_2.csv", [(v1, 17), (recent, 1), (v2, 2)]),
        ("splice_v1_16_rc_2_v2_2.csv", [(v1, 16), (recent, 2), (v2, 2)]),
        ("splice_v1_15_rc_3_v2_2.csv", [(v1, 15), (recent, 3), (v2, 2)]),
        ("splice_v1_14_rc_4_v2_2.csv", [(v1, 14), (recent, 4), (v2, 2)]),
        ("splice_v1_16_rc_1_v2_3.csv", [(v1, 16), (recent, 1), (v2, 3)]),
        ("splice_v1_15_rc_2_v2_3.csv", [(v1, 15), (recent, 2), (v2, 3)]),
        ("splice_v1_14_rc_3_v2_3.csv", [(v1, 14), (recent, 3), (v2, 3)]),
    ]
    paths = []
    for name, sources in specs:
        paths.append(splice_fuse(sources, out / name, [v1, v2, recent]))
    summary = pd.DataFrame([summarize(p) for p in paths])
    summary.to_csv(out / "candidate_summary_conservative.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
