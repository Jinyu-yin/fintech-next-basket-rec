from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .rrf_fusion import rrf_fuse_files
from .diagnostic_outputs import write_prediction_distribution, write_prediction_format_report

COLS = ["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]
SUBMISSION_COLUMNS = ["user_id", *COLS, "index"]


def add_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quad_key"] = df["action_typ"].astype(str).str.cat(
        [df["prod_typ"].astype(str), df["prod_sub_typ"].astype(str), df["rsk_lvl"].astype(str)], sep="|"
    )
    return df


def splice_fuse(
    sources: list[tuple[Path, int]],
    output_path: Path,
    fallback_paths: list[Path] | None = None,
) -> Path:
    groups = []
    users: list[str] = []
    for path, _quota in sources:
        df = add_key(pd.read_csv(path)).sort_values(["user_id", "index"])
        groups.append({str(uid): g for uid, g in df.groupby("user_id", sort=False)})
        if not users:
            users = [str(x) for x in df["user_id"].drop_duplicates().tolist()]
    fallback_groups = []
    for path in fallback_paths or []:
        df = add_key(pd.read_csv(path)).sort_values(["user_id", "index"])
        fallback_groups.append({str(uid): g for uid, g in df.groupby("user_id", sort=False)})

    rows = []
    for uid in users:
        selected = []
        seen = set()
        for (_path, quota), group in zip(sources, groups, strict=False):
            g = group.get(uid)
            if g is None:
                continue
            taken = 0
            for r in g.itertuples(index=False):
                if taken >= quota or len(selected) >= 20:
                    break
                if r.quad_key in seen:
                    continue
                selected.append(r)
                seen.add(r.quad_key)
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
                if r.quad_key in seen:
                    continue
                selected.append(r)
                seen.add(r.quad_key)
        for idx, r in enumerate(selected[:20], 1):
            rows.append({
                "user_id": uid,
                "action_typ": r.action_typ,
                "prod_typ": r.prod_typ,
                "prod_sub_typ": r.prod_sub_typ,
                "rsk_lvl": r.rsk_lvl,
                "index": idx,
            })
    pred = pd.DataFrame(rows)[SUBMISSION_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output_path, index=False, encoding="utf-8-sig")
    write_prediction_format_report(pred, output_path.parent, prefix=output_path.stem)
    write_prediction_distribution(pred, output_path.parent, prefix=output_path.stem)
    print(f"spliced: {output_path}")
    return output_path


def summarize(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    action = df["action_typ"].value_counts(normalize=True).to_dict()
    prod = df["prod_typ"].value_counts(normalize=True).to_dict()
    rsk = df["rsk_lvl"].value_counts(normalize=True).to_dict()
    top_quad_share = (
        df["action_typ"].astype(str).str.cat(
            [df["prod_typ"].astype(str), df["prod_sub_typ"].astype(str), df["rsk_lvl"].astype(str)], sep="|"
        ).value_counts(normalize=True).head(1).iloc[0]
    )
    return {
        "file": path.name,
        "rows": len(df),
        "users": df["user_id"].nunique(),
        "nonwealth": prod.get("非财富", 0.0),
        "browse": action.get("浏览详情", 0.0),
        "hold": action.get("浏览持仓", 0.0),
        "buy": action.get("购买", 0.0),
        "redeem": action.get("赎回", 0.0),
        "collect": action.get("收藏", 0.0),
        "offline": action.get("线下接触", 0.0),
        "activity": action.get("参与活动", 0.0),
        "r1": rsk.get("R1", 0.0),
        "r2": rsk.get("R2", 0.0),
        "r4": rsk.get("R4", 0.0),
        "r5": rsk.get("R5", 0.0),
        "top_quad_share": float(top_quad_share),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multiple fast fusion candidates from existing prediction files.")
    parser.add_argument("--submit-dir", type=Path, default=Path("/work/task1/submit"))
    args = parser.parse_args()

    d = args.submit_dir
    v1 = d / "prediction_v1_no_prior.csv"
    v2 = d / "prediction_v2_no_prior.csv"
    recent = d / "prediction_recent.csv"
    out_dir = d / "fusion_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[Path] = []
    rrf_specs = [
        ("rrf_v1_recent_1p0_1p6.csv", [v1, recent], [1.0, 1.6]),
        ("rrf_v1_recent_0p8_2p0.csv", [v1, recent], [0.8, 2.0]),
        ("rrf_v2_recent_0p8_1p8.csv", [v2, recent], [0.8, 1.8]),
        ("rrf_all_recent_heavy.csv", [v1, v2, recent], [0.8, 0.4, 2.0]),
        ("rrf_all_recent_very_heavy.csv", [v1, v2, recent], [0.6, 0.3, 2.4]),
    ]
    for name, paths, weights in rrf_specs:
        candidates.append(rrf_fuse_files(paths, out_dir / name, weights=weights, rrf_k=20.0))

    candidates.append(splice_fuse([(v1, 6), (recent, 10), (v2, 4)], out_dir / "splice_v1_6_recent_10_v2_4.csv", [recent, v1, v2]))
    candidates.append(splice_fuse([(v2, 5), (recent, 11), (v1, 4)], out_dir / "splice_v2_5_recent_11_v1_4.csv", [recent, v1, v2]))
    candidates.append(splice_fuse([(recent, 8), (v1, 8), (v2, 4)], out_dir / "splice_recent_8_v1_8_v2_4.csv", [v1, recent, v2]))
    candidates.append(splice_fuse([(v1, 8), (recent, 12)], out_dir / "splice_v1_8_recent_12.csv", [recent, v1]))

    summary = pd.DataFrame([summarize(p) for p in candidates])
    summary_path = out_dir / "candidate_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.sort_values(["top_quad_share", "nonwealth"]).to_string(index=False))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
