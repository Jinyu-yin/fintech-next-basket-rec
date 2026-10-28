from __future__ import annotations
import numpy as np
import pandas as pd

from .config import USER_COL, TIME_COL, TARGET_START, TARGET_END

# ===================== 官方强制值域校验 =====================
PROD_RSK_VALID = {
    "理财": ["R1", "R2"],
    "基金": ["R3", "R4", "R5"],
    "存款": ["R1", "R2"],
    "保险": ["R1", "R2", "R3"],
    "非财富": ["R2"]
}

ACTION_PROD_VALID = {
    "浏览详情": ["理财", "基金", "存款", "保险"],
    "浏览持仓": ["理财", "基金", "存款", "保险"],
    "收藏": ["理财", "基金", "存款", "保险"],
    "购买": ["理财", "基金", "存款", "保险"],
    "赎回": ["理财", "基金", "存款", "保险"],
    "参与活动": ["非财富"],
    "线下接触": ["非财富"]
}

PROD_SUB_MAP = {
    "理财": ["短债", "周周宝", "月月宝", "季季宝", "固收+", "半年宝", "多月宝", "定期宝", "现金", "非短债固收", "期次", "其他"],
    "基金": ["TMT行业", "周期行业", "灵活配置", "制造行业", "医药生物行业", "港股行业", "消费行业", "金融地产行业", "公用事业行业", "海外行业", "固收+", "固收", "FOF", "货币", "其他"],
    "存款": ["一般性", "结构性"],
    "保险": ["意外险", "医疗保险", "重大疾病保险", "护理保险", "两全险", "定期寿险", "终身寿险", "子女教育年金", "专属商业养老", "商业养老金", "税延养老年金", "养老年金", "财产保障险", "其他保险"],
    "非财富": ["其他"]
}

RSK_MAP = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5}
COLS = ["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]


def _mode(series: pd.Series) -> str:
    series = series.dropna()
    return series.mode().iloc[0] if not series.empty else "浏览详情"


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """填充空值并保证关键字段可用。"""
    df = df.copy()
    df = df.dropna(subset=["prod_typ"]).reset_index(drop=True)

    for prod_typ, default_subs in PROD_SUB_MAP.items():
        mask = (df["prod_typ"] == prod_typ) & df["prod_sub_typ"].isna()
        if mask.any():
            global_mode_series = df[df["prod_typ"] == prod_typ]["prod_sub_typ"].mode()
            global_mode = global_mode_series.iloc[0] if not global_mode_series.empty else default_subs[0]
            df.loc[mask, "prod_sub_typ"] = global_mode

    for prod_typ, default_rsks in PROD_RSK_VALID.items():
        mask = (df["prod_typ"] == prod_typ) & df["rsk_lvl"].isna()
        if mask.any():
            global_mode_series = df[df["prod_typ"] == prod_typ]["rsk_lvl"].mode()
            global_mode = global_mode_series.iloc[0] if not global_mode_series.empty else default_rsks[0]
            df.loc[mask, "rsk_lvl"] = global_mode

    df["action_typ"] = df["action_typ"].fillna("浏览详情")
    return df


def build_next_day_features(df: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    """构建用于预测下一天的用户级特征。"""
    tmp = df.copy()
    tmp[TIME_COL] = pd.to_datetime(tmp[TIME_COL], errors="coerce")
    tmp = _fill_missing_values(tmp)
    tmp = tmp[tmp[TIME_COL] <= end_date].reset_index(drop=True)
    tmp = tmp.sort_values([USER_COL, TIME_COL]).reset_index(drop=True)

    tmp["days_diff"] = (end_date - tmp[TIME_COL]).dt.total_seconds() / 86400
    tmp["quadruple"] = tmp[COLS].astype(str).agg("|".join, axis=1)

    out = []
    for user_id, g in tmp.groupby(USER_COL, sort=False):
        g = g.reset_index(drop=True)
        last_1d = g[g["days_diff"] <= 1]
        last_3d = g[g["days_diff"] <= 3]
        last_7d = g[g["days_diff"] <= 7]

        row = {USER_COL: user_id}
        row["last_act_hours_ago"] = (end_date - g[TIME_COL].iloc[-1]).total_seconds() / 3600
        row["cnt_last_1d"] = len(last_1d)
        row["cnt_last_3d"] = len(last_3d)
        row["cnt_last_7d"] = len(last_7d)
        row["total_events"] = len(g)
        row["last_week_ratio"] = row["cnt_last_7d"] / row["total_events"] if row["total_events"] > 0 else 0

        if len(g) > 1:
            avg_interval_hours = (g[TIME_COL].diff().dt.total_seconds() / 3600).mean()
        else:
            avg_interval_hours = 24
        half_life_days = max(1, min(7, avg_interval_hours / 24))
        g["decay_w"] = np.exp(-g["days_diff"] / half_life_days)

        if len(last_1d) > 0:
            row["last_1d_act"] = last_1d["action_typ"].iloc[-1]
            row["last_1d_prod"] = last_1d["prod_typ"].iloc[-1]
            row["last_1d_sub"] = last_1d["prod_sub_typ"].iloc[-1]
            row["last_1d_rsk"] = last_1d["rsk_lvl"].iloc[-1]
            row["last_1d_act_mode"] = _mode(last_1d["action_typ"])
            row["last_1d_prod_mode"] = _mode(last_1d["prod_typ"])
        else:
            row["last_1d_act"] = g["action_typ"].iloc[-1]
            row["last_1d_prod"] = g["prod_typ"].iloc[-1]
            row["last_1d_sub"] = g["prod_sub_typ"].iloc[-1]
            row["last_1d_rsk"] = g["rsk_lvl"].iloc[-1]
            row["last_1d_act_mode"] = _mode(g["action_typ"])
            row["last_1d_prod_mode"] = _mode(g["prod_typ"])

        row["last_3d_act_mode"] = _mode(last_3d["action_typ"])
        row["last_3d_prod_mode"] = _mode(last_3d["prod_typ"])
        row["last_3d_sub_mode"] = _mode(last_3d["prod_sub_typ"])
        row["last_7d_act_mode"] = _mode(last_7d["action_typ"])
        row["last_7d_prod_mode"] = _mode(last_7d["prod_typ"])
        row["last_7d_sub_mode"] = _mode(last_7d["prod_sub_typ"])

        row["wgt_act"] = g.groupby("action_typ")["decay_w"].sum().idxmax()
        row["wgt_prod"] = g.groupby("prod_typ")["decay_w"].sum().idxmax()
        row["wgt_sub"] = g.groupby("prod_sub_typ")["decay_w"].sum().idxmax()
        row["wgt_rsk"] = g.groupby("rsk_lvl")["decay_w"].sum().idxmax()

        mapped_rsk = g["rsk_lvl"].map(RSK_MAP)
        row["rsk_mean"] = mapped_rsk.mean()
        row["rsk_std"] = mapped_rsk.std(ddof=0) if len(g) > 1 else 0.0
        row["recent_act_switch"] = 1 if len(g) > 1 and g["action_typ"].iloc[-1] != g["action_typ"].iloc[-2] else 0
        row["recent_prod_switch"] = 1 if len(g) > 1 and g["prod_typ"].iloc[-1] != g["prod_typ"].iloc[-2] else 0
        row["recent_sub_switch"] = 1 if len(g) > 1 and g["prod_sub_typ"].iloc[-1] != g["prod_sub_typ"].iloc[-2] else 0
        row["recent_rsk_switch"] = 1 if len(g) > 1 and g["rsk_lvl"].iloc[-1] != g["rsk_lvl"].iloc[-2] else 0
        row["act_unique_7d"] = g["action_typ"].nunique()
        row["prod_unique_7d"] = g["prod_typ"].nunique()
        row["sub_unique_7d"] = g["prod_sub_typ"].nunique()

        quad_count_all = g["quadruple"].value_counts().to_dict()
        quad_count_3d = last_3d["quadruple"].value_counts().to_dict()
        quad_count_7d = last_7d["quadruple"].value_counts().to_dict()
        quad_weighted_count = g.groupby("quadruple")["decay_w"].sum().to_dict()
        act_prod_count = g.groupby(["action_typ", "prod_typ"]).size().to_dict()
        prod_sub_count = g.groupby(["prod_typ", "prod_sub_typ"]).size().to_dict()
        rsk_count = g.groupby(["prod_typ", "rsk_lvl"]).size().to_dict()

        row["quad_count_all"] = quad_count_all
        row["quad_count_3d"] = quad_count_3d
        row["quad_count_7d"] = quad_count_7d
        row["quad_weighted_count"] = quad_weighted_count
        row["act_prod_count"] = act_prod_count
        row["prod_sub_count"] = prod_sub_count
        row["rsk_count"] = rsk_count

        top_quads = []
        top_quads.extend(last_1d["quadruple"].unique().tolist())
        top_quads.extend(last_3d["quadruple"].unique().tolist())
        top_quads.extend(last_7d["quadruple"].unique().tolist())
        top_quads.extend(g.groupby("quadruple")["decay_w"].sum().nlargest(30).index.tolist())
        top_quads.extend(g.groupby(["action_typ", "prod_typ", "prod_sub_typ", "rsk_lvl"]).size().sort_values(ascending=False).head(30).index.map(lambda x: "|".join(x)).tolist())

        seen = set()
        row["top_quads"] = []
        for q in top_quads:
            if q in seen:
                continue
            seen.add(q)
            row["top_quads"].append(tuple(q.split("|")))
            if len(row["top_quads"]) >= 30:
                break

        out.append(row)

    return pd.DataFrame(out).fillna(0)


def build_training_cutoff_features(df: pd.DataFrame, cutoff_date: str, target_date: str) -> pd.DataFrame:
    """构造单日 next-day 监督样本，便于验证或 4/1 增强。"""
    cutoff_ts = pd.Timestamp(cutoff_date)
    target_ts = pd.Timestamp(target_date)
    feat = build_next_day_features(df[df[TIME_COL] <= cutoff_ts], cutoff_ts)
    target_df = df[(df[TIME_COL] >= target_ts) & (df[TIME_COL] < target_ts + pd.Timedelta(days=1))].copy()
    target_df = _fill_missing_values(target_df)
    target_df["quadruple"] = target_df[COLS].astype(str).agg("|".join, axis=1)
    target_map = target_df.groupby(USER_COL)["quadruple"].apply(lambda s: set(s.tolist())).to_dict()
    feat["target_quads"] = feat[USER_COL].map(target_map).apply(lambda x: x if isinstance(x, set) else set())
    return feat


def build_global_top_quads(df: pd.DataFrame, top_n: int = 100) -> list[tuple]:
    """生成全局高频四元组。"""
    df = _fill_missing_values(df)
    df["quadruple"] = df[COLS].astype(str).agg("|".join, axis=1)
    top_quads = df["quadruple"].value_counts().nlargest(top_n).index.tolist()
    return [tuple(q.split("|")) for q in top_quads]


def build_quad_april1_count(train_df: pd.DataFrame) -> dict[str, int]:
    """统计目标日当天每个四元组的出现次数，避免把题目窗口外日期混入热门统计。"""
    target_start = pd.Timestamp(TARGET_START)
    target_end = pd.Timestamp(TARGET_END)
    april1_df = train_df[(train_df[TIME_COL] >= target_start) & (train_df[TIME_COL] < target_end)].copy()
    if april1_df.empty:
        return {}
    april1_df["quadruple"] = april1_df[COLS].astype(str).agg("|".join, axis=1)
    return april1_df["quadruple"].value_counts().to_dict()
