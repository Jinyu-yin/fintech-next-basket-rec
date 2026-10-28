# Fintech Next-Basket Recommendation

> 金融产品推荐赛题：预测用户**下一天**会做的「行为 × 产品 × 子类 × 风险等级」四元组，核心是 next-basket 与 repeat-aware 排序。

![python](https://img.shields.io/badge/python-3.10%2B-3776ab) ![model](https://img.shields.io/badge/model-LGBMRanker-9ca3af) ![task](https://img.shields.io/badge/task-ranking-f59e0b)

---

## 问题定义

给定用户在历史窗口内的行为序列，为每个用户预测**次日**最可能发生的一条行为记录：

```
(user_id, action_typ, prod_typ, prod_sub_typ, rsk_lvl)
```

其中四个字段并非自由组合，而是受业务规则强约束的——这是本题最容易被忽略、也最容易踩坑的地方。

### 标签取值

| 字段 | 取值 |
|---|---|
| `action_typ` | 浏览详情 / 浏览持仓 / 收藏 / 购买 / 赎回 / 参与活动 / 线下接触 |
| `prod_typ` | 理财 / 基金 / 存款 / 保险 / 非财富 |
| `prod_sub_typ` | 按产品大类展开，共 40+ 个细分（短债、周周宝、TMT行业、结构性…） |
| `rsk_lvl` | R1 ~ R5 |

### 合法性约束（关键）

```python
# 产品大类 → 允许的风险等级
PROD_RSK_VALID = {
    "理财": ["R1", "R2"],
    "基金": ["R3", "R4", "R5"],
    "存款": ["R1", "R2"],
    "保险": ["R1", "R2", "R3"],
    "非财富": ["R2"],
}

# 行为 → 允许的产品大类（"参与活动"/"线下接触" 只能是非财富）
ACTION_PROD_VALID = { ... }
```

**所有候选在进入排序前必须先过滤**，否则模型会把概率质量浪费在根本不存在的组合上。

---

## 时间窗口

| 用途 | 区间 |
|---|---|
| 训练/特征历史 | `2025-03-01 00:00:00` ~ `2025-03-31 23:59:59` |
| 预测目标日 | `2025-04-01 00:00:00` |

原始数据日期范围比这更宽，但**只使用该窗口**——这是题目约定，也避免跨期分布漂移。

---

## 方法：混合排序

不是纯粹的端到端模型，而是「规则先验 + 学习排序」的混合打分：

```python
HYBRID_WEIGHTS = {
    "lgbm": 0.25,           # 模型分
    "repeat_1d": 18.0,      # 昨日重复（最强的信号）
    "repeat_3d": 10.0,
    "repeat_7d": 6.0,
    "user_weighted": 4.0,   # 用户加权频次
    "user_all": 1.5,
    "field_pref": 3.0,      # 分字段偏好
    "transition": 5.0,      # 行为转移
    "risk": 2.0,            # 风险等级偏好
    "april1_hot": 2.0,      # 全局热度
}
```

为什么给 `repeat_1d` 这么高的权重？因为金融产品的用户行为有极强的**惯性**：昨天看了"月月宝"，今天大概率还是看它。纯 LGBM 在冷启动用户上会输给这条规则。

### 行为转移矩阵

```python
ACTION_TRANSITIONS = {
    "浏览详情": [("浏览详情", 1.0), ("收藏", 0.75), ("购买", 0.45)],
    "购买":     [("浏览详情", 0.45), ("浏览持仓", 1.0), ("购买", 0.6)],
    "浏览持仓": [("浏览持仓", 1.0), ("赎回", 0.75), ("购买", 0.35)],
    ...
}
```

这条链把「浏览 → 收藏 → 购买 → 持仓 → 赎回」的漏斗显式编码进去，属于业务先验，比让模型自己学更省数据。

### 特征工程

`feature_engineering.py` 构造的核心特征：

| 特征 | 说明 |
|---|---|
| `quad_count_all` | 用户对某四元组的历史总次数 |
| `quad_count_3d` / `quad_count_7d` | 近 3 天 / 7 天次数（捕捉短期意图） |
| `quad_weighted_count` | 时间衰减加权次数 |
| `quad_april1_count` | 4 月 1 日当天热度（跨用户全局） |
| `act_prod_count` | (行为, 产品) 二元组频次 |

### 排序模型

`modeling.py` 用 `LGBMRanker`（LambdaRank）做学习排序：

- 每个用户构造正样本 + `neg_per_user` 个负样本；
- 负样本来自该用户历史上出现过的其它四元组（**in-basket negatives**），比随机负采样更难、更有效；
- 排序目标直接优化 NDCG@1。

---

## 目录结构

```
fintech-next-basket-rec/
└── task1/
    ├── config.py                      # 路径、时间窗口、目标列
    ├── data_utils.py
    ├── feature_engineering.py         # ⭐ 特征 + 合法性约束表
    ├── modeling.py                    # ⭐ LGBMRanker + 混合打分
    ├── pipeline.py                    # 主流水线（含 checkpoint）
    ├── full_label_classifier.py       # 四元组全标签分类/排序
    ├── full_label_fusion.py
    ├── fast_light_ranker.py / fast_light_cross.py
    ├── recent_repeat_model.py         # 近期重复基线
    ├── recent_conservative_model.py
    ├── repeat_first.py                # 重复优先基线
    ├── solution_ranker.py
    ├── make_fusion_candidates.py      # 候选融合
    ├── make_calibrated_fusion.py      # 校准融合
    ├── make_conservative_grid.py      # 保守参数网格
    ├── make_low_nonwealth_candidates.py
    ├── make_nonwealth_empty.py
    ├── rrf_fusion.py                  # Reciprocal Rank Fusion
    ├── fuse_predictions.py
    ├── local_eval.py                  # 本地评估
    ├── diagnostics.py / diagnostic_outputs.py
    ├── reproduce_5996.py              # 复现线上成绩（v1 + 融合）
    ├── run_full_label_variant.py
    ├── run.py                         # 入口
    └── notebook_cell.py               # Jupyter 一键提交 Cell
```

---

## 快速开始

```bash
pip install pandas numpy scikit-learn lightgbm
```

数据需自行准备：`dataset_train.csv` / `dataset_test.csv`，放在 `/work/data/task1/`（或修改 `config.py` 中的 `EXAM_DIR`）。

```bash
# 标准入口
python run.py

# 复现线上成绩（多模型 + 融合）
python -m task1.reproduce_5996 \
  --train-path data/dataset_train.csv \
  --test-path  data/dataset_test.csv \
  --submit-dir submit
```

Jupyter 平台只需复制 `notebook_cell.py` 内容到一个 cell，改两行路径即可。

输出：`submit/prediction.csv`，列为

```
user_id, action_typ, prod_typ, prod_sub_typ, rsk_lvl, index
```

---

## 多方案融合

单模型不稳定，因此仓库里保留了多条互补路径，最后做融合：

| 方案 | 思路 |
|---|---|
| **全标签分类器** | 把四元组当整体标签，做分类 + 频率先验 |
| **学习排序** | LGBMRanker + in-basket 负采样 |
| **重复优先** | 直接预测"昨日同款"，作为强基线 |
| **近期重复** | 只在近 N 天窗口内找重复 |
| **保守模型** | 限制候选空间，牺牲召回换精度 |
| **RRF 融合** | 按排名倒数加权融合多路结果 |
| **校准融合** | 在验证集上学习各路结果的融合权重 |

`make_*_candidates.py` 负责生成各路候选，`fuse_predictions.py` / `rrf_fusion.py` / `make_calibrated_fusion.py` 负责合并。

---

## 已知限制

- **无数据**：仓库不含任何赛题数据，`data/` 目录为空，需自行放入。
- **路径硬编码**：`config.py` 中 `/work/...` 为竞赛平台路径，本地运行请改 `EXAM_DIR` 与 `PROJECT_DIR`。
- **checkpoint 机制**：`pipeline.py` 会把中间状态写入 `models/*.pkl`，调试时若改了特征逻辑记得清空，否则会读到旧缓存。
- **混合权重是手调的**：`HYBRID_WEIGHTS` 来自验证集经验，不是学出来的，换数据集需要重新校准。
- 未做线上 A/B 或时延约束验证，仅为离线指标优化。
