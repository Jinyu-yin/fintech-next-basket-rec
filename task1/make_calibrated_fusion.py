from __future__ import annotations

from pathlib import Path

import pandas as pd

from .diagnostic_outputs import write_prediction_distribution, write_prediction_format_report
from .make_fusion_candidates import summarize

COLS = ["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]
SUBMISSION_COLUMNS = ["user_id", *COLS, "index"]


def add_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    nonwealth = df["prod_typ"].astype(str).eq("非财富")
    df.loc[nonwealth, "prod_sub_typ"] = ""
    df.loc[nonwealth, "rsk_lvl"] = ""
    df["quad_key"] = df["action_typ"].astype(str).str.cat(
        [df["prod_typ"].astype(str), df["prod_sub_typ"].astype(str), df["rsk_lvl"].astype(str)], sep="|"
    )
    return df


def calibrated_splice(
    sources: list[tuple[Path, int]],
    output_path: Path,
    fallback_paths: list[Path],
    nonwealth_cap: int = 2,
    nonwealth_topk_cap: int = 1,
    topk_guard: int = 10,
) -> Path:
    groups = []
    users: list[str] = []
    for path, _quota in sources:
        df = add_key(pd.read_csv(path, keep_default_na=False)).sort_values(["user_id", "index"])
        groups.append({str(uid): g for uid, g in df.groupby("user_id", sort=False)})
        if not users:
            users = [str(x) for x in df["user_id"].drop_duplicates().tolist()]
    fallback_groups = []
    for path in fallback_paths:
        df = add_key(pd.read_csv(path, keep_default_na=False)).sort_values(["user_id", "index"])
        fallback_groups.append({str(uid): g for uid, g in df.groupby("user_id", sort=False)})

    rows = []
    for uid in users:
        selected = []
        seen = set()
        nonwealth_count = 0

        def can_add(row) -> bool:
            nonlocal nonwealth_count
            if row.quad_key in seen:
                return False
            if row.prod_typ == "非财富":
                if nonwealth_count >= nonwealth_cap:
                    return False
                if len(selected) < topk_guard:
                    top_nonwealth = sum(1 for x in selected[:topk_guard] if x.prod_typ == "非财富")
                    if top_nonwealth >= nonwealth_topk_cap:
                        return False
            return True

        def add_row(row) -> None:
            nonlocal nonwealth_count
            selected.append(row)
            seen.add(row.quad_key)
            if row.prod_typ == "非财富":
                nonwealth_count += 1

        for (_path, quota), group in zip(sources, groups, strict=False):
            g = group.get(uid)
            if g is None:
                continue
            taken = 0
            for r in g.itertuples(index=False):
                if taken >= quota or len(selected) >= 20:
                    break
                if can_add(r):
                    add_row(r)
                    taken += 1
        for group in fallback_groups + groups:
            if len(selected) >= 20:
                break
            g = group.get(uid)
            if g is None:
                continue
            for r in g.itertuples(index=False):
                if len(selected) >= 20:
                    break
                if can_add(r):
                    add_row(r)
        for idx, r in enumerate(selected[:20], 1):
            rows.append(
                {
                    "user_id": uid,
                    "action_typ": r.action_typ,
                    "prod_typ": r.prod_typ,
                    "prod_sub_typ": "" if r.prod_typ == "非财富" else r.prod_sub_typ,
                    "rsk_lvl": "" if r.prod_typ == "非财富" else r.rsk_lvl,
                    "index": idx,
                }
            )
    pred = pd.DataFrame(rows)[SUBMISSION_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output_path, index=False, encoding="utf-8-sig")
    write_prediction_format_report(pred, output_path.parent, prefix=output_path.stem)
    write_prediction_distribution(pred, output_path.parent, prefix=output_path.stem)
    print(f"calibrated: {output_path}")
    return output_path


def main() -> None:
    d = Path("/work/task1/submit")
    if not d.exists():
        d = Path("cloud_submit_all_latest")
    out = d / "fusion_candidates_calibrated"
    out.mkdir(parents=True, exist_ok=True)
    v1 = d / "prediction_v1_no_prior.csv"
    v2 = d / "prediction_v2_no_prior.csv"
    recent = d / "prediction_recent.csv"
    specs = [
        ("cal_v1_16_recent_2_v2_2_cap2_top1.csv", [(v1, 16), (recent, 2), (v2, 2)], 2, 1),
        ("cal_v1_15_recent_3_v2_2_cap2_top1.csv", [(v1, 15), (recent, 3), (v2, 2)], 2, 1),
        ("cal_v1_14_recent_4_v2_2_cap2_top1.csv", [(v1, 14), (recent, 4), (v2, 2)], 2, 1),
        ("cal_v1_16_recent_2_v2_2_cap3_top1.csv", [(v1, 16), (recent, 2), (v2, 2)], 3, 1),
        ("cal_v1_15_recent_3_v2_2_cap3_top1.csv", [(v1, 15), (recent, 3), (v2, 2)], 3, 1),
        ("cal_v1_16_recent_1_v2_3_cap2_top1.csv", [(v1, 16), (recent, 1), (v2, 3)], 2, 1),
        ("cal_v1_17_recent_1_v2_2_cap2_top1.csv", [(v1, 17), (recent, 1), (v2, 2)], 2, 1),
    ]
    paths = []
    for name, sources, cap, topcap in specs:
        paths.append(calibrated_splice(sources, out / name, [v1, v2, recent], cap, topcap))
    summary = pd.DataFrame([summarize(p) for p in paths])
    summary.to_csv(out / "candidate_summary_calibrated.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
