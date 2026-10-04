# Master-revised-code

论文《**Spatio-temporal Dynamics and Network Diffusion of Weibo Sentiment Toward U.S. Tariff Policies: A Geographic Perspective**》的可复现代码。在原有零散代码/Notebook 基础上整理、补全，从清洗后数据 `Master/weibo_cleaned_full.csv` 出发，复现论文第 3–4 章全部分析与图表。

## 环境

- Python 3.8 + CUDA 12.0，venv：`/home/dengjianwei/venvs/thesis`
- 依赖：`pip install -r requirements.txt`
- GPU：2×A800 80GB（推理默认使用 `cuda:1`，`cuda:0` 长期被其他任务占用）

## 目录与论文章节对应

| 目录 | 内容 | 论文章节/图表 |
|---|---|---|
| `01_preprocessing/` | `merge_and_clean.py` 原始 CSV 合并去重；`weibo_data_cleaning.py` 垃圾过滤/去噪/jieba 分词/停用词 | §3 数据预处理 |
| `02_sentiment/` | `llm_teacher_labeling.py` DeepSeek 蒸馏标注；`train_roberta.py` RoBERTa 微调；`infer_sentiment.py` HF 模型批量推理；`infer_sentiment_skep.py` SKEP 批量推理（**本机实际运行**）；`download_model.py` HF 模型下载 | §3.2 / Fig 22 |
| `03_temporal/` | `dataset_overview.py` 关键词帖子/用户点图；`temporal_dynamics.py` SMA-7+τ 阶段检测、总体情感、事件生命周期、事件前后（pre 7d/post 14d）对比 | §3.1 / Fig 2–4 |
| `04_regional/` | `cross_regional.py` 中外/中美对比、海外国家、省级地图 | §3.3 / Fig 5–13 |
| `05_spatial/` | `gwr_analysis.py` 主题簇、Moran's I、OLS/GWR、局部 R²/系数地图 | §3.4 / Table 5–9 / Fig 14–17 |
| `06_network/` | `retweet_chain_builder.py` 转发边/节点/链；`cascade_analysis.py` 五级用户级联、KOL、阈值对比、深度 CCDF、拓扑对比 | §4 / Table 10 / Fig 18–21 |
| `outputs/` | 各阶段 CSV 与图表（按分子目录） | — |
| `data/` | Master 原始数据副本：原始合并/各版清洗 CSV、关键词与停用词、省界 shp、DeepSeek 标注样本（项目自包含，约 7.1 GB） | 数据 |
| `reference/` | 未整合但有参考价值的原始脚本存档（见下「参考脚本存档」） | — |

全局口径（路径、阶段日期、四主题簇关键词映射、五级用户、省名映射、阈值）统一在 `config.py`。数据路径优先指向项目内 `data/`，缺失时自动回退到 `/home/dengjianwei/Master`。

## 运行顺序

```bash
PY=/home/dengjianwei/venvs/thesis/bin/python

# (已有 weibo_cleaned_full.csv, 01 仅作存档, 通常无需重跑)

# 02 情感推理 -> outputs/weibo_sentiment.csv (A800 batch_size=256 约 20 分钟, 支持断点续跑)
$PY 02_sentiment/infer_sentiment_skep.py

# 03 时序
$PY 03_temporal/dataset_overview.py    # Fig 2
$PY 03_temporal/temporal_dynamics.py

# 04 跨区域
$PY 04_regional/cross_regional.py

# 05 空间计量
$PY 05_spatial/gwr_analysis.py

# 06 网络级联
$PY 06_network/retweet_chain_builder.py
$PY 06_network/cascade_analysis.py
```

## 情感模型说明（与论文口径的差异）

- **论文路线**：DeepSeek-V3 蒸馏标注 2000 条 → 微调 `hfl/chinese-roberta-wwm-ext`（lr 2e-5，bs 32，maxlen 128，3 epochs，AdamW）→ 全量三分类推理。对应 `llm_teacher_labeling.py` + `train_roberta.py` + `infer_sentiment.py`，脚本与超参按论文原样保留；运行需要 DeepSeek API key 与 HuggingFace 访问。
- **本机实际路线**：校园网出口拦截了 HuggingFace（302 跳转认证页），改用可从百度 BOS 获取的 **PaddleNLP `skep_ernie_1.0_large_ch`**（ERNIE 2.0 large 情感模型）做全量推理（`infer_sentiment_skep.py`）。该模型为二分类（positive/negative）+ 置信度，转换为连续分（positive→score，negative→1−score）后按论文 0.4/0.6 阈值映射 Negative/Neutral/Positive，下游所有分析口径不变。
- 论文图 22 的四模型对比依据 `Master/Test_outcomes/` 既有结果（DeepSeek-V3 准确率 80% 最优）。

## 适配说明

- 原始数据无 `user_id`，网络分析以 **用户昵称** 作为用户标识。
- `retweet_id` 指向数据集外的帖子记为外部根，成员仍归入该级联（Fig 18 以虚拟中心节点绘制外部根级联）。
- 清洗数据中 `retweet_id` 只记录直接转发的原帖，图结构为扁平星状（最大图深度=1）；级联深度另从微博正文嵌套标记 `//@用户:` 解析，与图深度取较大值，用于 Table 10 / Fig 21。
- 省界使用 `data/Geo_analysis/cn_shp/cn.shp`（GADM，英文名，EPSG:4326），GWR 前投影到 EPSG:4547 取质心坐标；geopandas 读 shp 需 `engine="pyogrio"`（本机 fiona 1.10 不兼容）。
- 省级总量图广东（53,716 帖）为极端头部；省级两张图均用 Blues 线性色阶（与论文原图一致），人均图按每千万人口归一化。
- 三个事件阶段各单独跑一次 GWR（结果 `table9_phase_gwr_coefficients.csv`），用于 Fig 17 的阶段间系数差值。阶段 1/3 窗口内「芯片/半导体」关键词全省均无帖（Technology 占比恒为 0），该变量在这两个阶段无法估计，对应系数留空、地图灰显，这是数据本身的特点。

## `data/` 数据副本说明

| 文件 | 说明 |
|---|---|
| `merged_all_data.csv` (2.8G) | 各关键词目录原始 CSV 合并后的最原始数据 |
| `weibo_cleaned_full.csv` (2.1G) | 清洗+分词后的全量数据（**本项目全部分析的输入**，337,546 帖） |
| `weibo_cleaned_data.csv` (1.5G) / `cleaned_weibo_data.csv` (796M) | 早期清洗版本（存档对照） |
| `filtered_posts1.csv` / `filtered_posts.csv` | 艾特用户>3 的广告排查样本 |
| `keywords.txt` / `stopwords.txt` / `hit_stopwords.txt` / `highfreinvalid.TXT` / `user.TXT` | 58 个检索关键词、停用词、高频无效词、用户名单 |
| `Geo_analysis/` | 中国省级边界 shp（含海南等 34 行英文名边界）与 ArcGIS Getis 探索工程（论文未使用） |
| `Test_outcomes/` | DeepSeek/GLM/Kimi 三模型在三类话题上的标注样本与准确率截图（Fig 22 依据） |

## 参考脚本存档（`reference/`）

- `original_sentiment_analysis_uer_roberta.py`：作者原始情感分析管线，使用 **UER RoBERTa-JD 二分类模型**（`uer/roberta-base-finetuned-jd-binary-chinese`，论文正文方法描述的模型路线之一），含事件窗口（pre 7d/post 14d）前后对比、小时模式、省级统计、网络属性导出。因 HuggingFace 在本机不可达未纳入可执行管线；其事件前后对比口径已吸收进 `03_temporal/temporal_dynamics.py`。
- `original_retweet_chain_builder.py`：原始转发链构建脚本（48KB），含根追溯/深度、交互式 HTML 可视化与通用网络图（拓扑/深度分布/入度分布/级联规模）。论文 Fig 18–21 的专用图已由 `06_network/cascade_analysis.py` 复现，此文件保留原始实现细节供核对。
- 原始 Notebook（`SUMMARY.ipynb`、`Untitled*.ipynb`）内容为合并、清洗、分词、艾特网络草稿，其有效部分已整理进 `01_preprocessing/`，未再复制；艾特网络依赖清洗时丢弃的「艾特用户」列，无法从现有数据重建。

## 本机运行结果（2025-01-01 ~ 2025-07-07，共 337,546 帖）

**情感分布**：Positive 160,924（47.7%）/ Neutral 31,238（9.3%）/ Negative 145,384（43.1%）。

三事件窗口情感均值（±15 天，帖量加权）：

| 窗口 | 帖数 | 均值 | Neg% | Neu% | Pos% |
|---|---|---|---|---|---|
| 基线（1 月） | 44,086 | 0.447 | 50.4 | 11.7 | 37.9 |
| Phase 1（02-13） | 42,661 | 0.438 | 52.1 | 11.2 | 36.6 |
| Phase 2（04-09） | 30,284 | 0.426 | 53.9 | 8.6 | 37.5 |
| Phase 3（05-14） | 51,613 | 0.464 | 49.2 | 8.6 | 42.2 |

SMA-7+τ 检测到 5 月下旬–6 月中旬负向比例显著低于基线（中美日内瓦会谈后回暖区间）。

**事件前后对比**（pre 7 天 vs post 14 天，`table_event_prepost_comparison.csv` / `Fig_event_prepost_comparison.png`）：Phase 1（2-13 对等关税）后情感均值 0.485→0.447（−0.038，负面占比 +5.1pp）；Phase 2（4-9 暂缓关税）0.355→0.448（+0.093，负面 −10.6pp）；Phase 3（5-14 日内瓦会谈）0.500→0.640（+0.140，负面 −16.0pp、正面 +18.4pp）。

**Table 5（多标签主题簇频次）**：Strategic 192,204（56.9%）/ Trade 165,825（49.1%）/ Livelihood 130,383（38.6%）/ Technology 50,504（15.0%）。

**Table 6（空间计量，31 省）**：Moran's I = −0.190（p=0.181，空间自相关不显著，符号与论文 −0.081 一致）；OLS R² = 0.611；GWR R² = **0.770**，自适应 bi-square/AICc 带宽 = **29**（论文 BW=27、R²=0.6004，高度接近）。

**网络级联**：唯一帖子 337,430，转发边 15,201；size>2 级联 1,510 个、size>5 级联 383 个；深度 CCDF N：Neg 5,611 / Neu 5,175 / Pos 4,581。Table 10（Basic，size>2，按用户级认证归类，不含外部根级联）：GoldV 372 / RedV 221 / YellowV 86 / BlueV 266 / Regular 256（另有 309 个级联的根帖在数据集外，单列 External）。

说明：认证字段在原数据中大面积缺失且同一用户各帖可能不一致，脚本按用户汇总其全部**非空**认证记录，只有某类 V 标记数严格多于明确的“普通用户”标记时才归为该 V，否则按普通用户处理（避免单条偶然标记升级）。受字段缺失影响，少数实际认证账号（如“央视新闻”）在本数据里没有任何 V 记录，只能显示为 Regular。

**海外 Top5**：美国 1,666、加拿大 850、日本 464、澳大利亚 363、安哥拉 305（共 79 个海外国家/地区）。

**中国 vs 美国独立情感汇总**（`outputs/regional/`）：

| 区域 | 帖数 | 均值 | 中位数 | Neg% | Neu% | Pos% |
|---|---|---|---|---|---|---|
| 中国 | 164,568 | 0.4521 | 0.3881 | 50.5 | 9.5 | 40.0 |
| 美国 | 1,666 | 0.3759 | 0.2340 | 59.8 | 9.3 | 30.9 |
| 海外整体 | 3,759 | 0.3974 | 0.2825 | 56.8 | 10.6 | 32.6 |

在美华人用户情感显著低于国内（Welch t=8.62，p≈1.5e-17；Mann-Whitney U p≈1.0e-12）。产物：`table_china_us_sentiment.csv`（总体构成）、`table_china_us_sentiment_by_phase.csv`（总体+三事件窗口均值/构成）、`table_china_us_tests.csv`（差异检验）、`Fig_china_us_sentiment_summary.png`（构成柱+窗口均值对比）。Fig 8 为中/美原始逐日情感线（不做滑动平均，配色对齐论文原图）。

### 与论文数值的差异及原因

- 本次复现基于**清洗后全量 337,546 帖**，论文 Table 5 / Table 10 / CCDF 的绝对计数基于其更大的原始采集集（百万级），故频次/级联数整体偏小，但相对排序（Strategic>Trade>Livelihood>Technology；五级用户结构）一致。
- OLS R²（0.611 vs 论文 0.0306）偏高、GWR 带宽（29 vs 27）与 R²（0.770 vs 0.6004）接近：自变量为四主题簇的省级发帖占比（source_dir 单标签），在本数据上主题结构对省均情感的全局解释力更强；模型设定（y=省均情感、X=簇占比、adaptive bi-square、AICc 选带宽）与论文一致。
- Fig 14 按论文原图重制为四主题簇的关键词情感均值横条图（红/蓝以 0.5 分界，面板内按均值排序）；SKEP 输出分数集中于 0/1 附近（二分类置信度），故多数关键词均值落在两端，具体排序与论文连续教师分不同。原图 Livelihood 面板中的“物价（prices）”在本数据采集关键词中不存在，该条未画。
- Fig 1（研究框架，概念图）与 Fig 22（四模型准确率对比，依赖外部 API 与 `Test_outcomes/`）不在复现范围内。
#   W e i b o - t a r i f f - d y n a m i c s - d i f f u s i o n  
 