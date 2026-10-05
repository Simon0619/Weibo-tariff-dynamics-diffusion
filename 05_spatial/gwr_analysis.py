#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 空间分析部分:
# 先按省份和四个关键词簇汇总 (表5), 再算 Moran's I、OLS、GWR (表6-9),
# 被解释变量是各省平均情感分, 解释变量是四个簇的帖子占比。
# GWR 用自适应 bi-square 核, 带宽按 AICc 自动选 (mgwr 包)。
# 图: Fig14 各簇情感箱线图, Fig15 局部 R2 地图, Fig16 整体系数 2x2,
# Fig17 三个事件阶段之间的系数差值 + 整体图。
# 另外三个阶段窗口各跑一次 GWR, 结果存在 table9_phase_gwr_coefficients.csv。
# 直接 python gwr_analysis.py 就行, 读的是 config 里配好的路径。
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

# mgwr/spglm 2.2.1 仍用 np.float 等 NumPy 1.24 已删除的别名, 打补丁
for _alias, _real in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _real)

C.setup_plot()
OUT = C.OUTPUT_DIR / "spatial"

KW2CLUSTER = {kw: cl for cl, kws in C.KEYWORD_CLUSTERS.items() for kw in kws}

# 图14 四个面板里要画的关键词, 完全照论文原图选 (注意 "大豆" 在 Trade 和 Technology 都出现)
FIG14_PANELS = {
    "Strategic": ["中美达成协议", "欧盟", "协议", "美元", "合作", "特朗普", "中美贸易",
                  "美国", "拜登", "301调查", "双边关系", "WTO", "RCEP", "国际合作",
                  "政治舆论", "保护主义", "多边贸易"],
    "Livelihood": ["黄金", "汇率", "汇率波动", "物价", "通货膨胀", "GDP", "经济", "A股",
                   "股市波动", "全球经济", "外贸", "全球股市", "民意", "投资者信心", "抗议"],
    "Trade": ["反倾销", "豁免", "关税", "进口", "大豆", "关税政策", "跨境电商", "出口",
              "对华关税", "贸易逆差", "壁垒", "供应链", "全球贸易战", "贸易壁垒",
              "贸易战", "产业转移", "自由贸易区"],
    "Technology": ["资本外流", "光伏", "半导体", "大豆", "稀土", "钢铁", "国产替代",
                   "进口替代", "汽车", "新能源", "芯片", "电动车", "中国制造"],
}
FIG14_EN = {
    "中美达成协议": "US-China agreement", "欧盟": "EU", "协议": "agreement", "美元": "US Dollar",
    "合作": "cooperation", "特朗普": "Trump", "中美贸易": "US-China trade", "美国": "United States",
    "拜登": "Biden", "301调查": "Section 301 investigation", "双边关系": "bilateral relations",
    "WTO": "WTO", "RCEP": "RCEP", "国际合作": "international cooperation",
    "政治舆论": "political opinion", "保护主义": "protectionism", "多边贸易": "multilateral trade",
    "黄金": "gold", "汇率": "exchange rate", "汇率波动": "exchange rate volatility",
    "物价": "prices", "通货膨胀": "inflation", "GDP": "GDP", "经济": "economy",
    "A股": "A-share market", "股市波动": "stock market volatility", "全球经济": "global economy",
    "外贸": "foreign trade", "全球股市": "global stock market", "民意": "public opinion",
    "投资者信心": "investor confidence", "抗议": "protests", "反倾销": "anti-dumping",
    "豁免": "exemptions", "关税": "tariffs", "进口": "imports", "大豆": "soybeans",
    "关税政策": "tariff policy", "跨境电商": "cross-border e-commerce", "出口": "exports",
    "对华关税": "tariffs on China", "贸易逆差": "trade deficit", "壁垒": "barriers",
    "供应链": "supply chain", "全球贸易战": "global trade war", "贸易壁垒": "trade barriers",
    "贸易战": "trade war", "产业转移": "industry relocation", "自由贸易区": "free trade zone",
    "资本外流": "capital outflow", "光伏": "photovoltaics", "半导体": "semiconductors",
    "稀土": "rare earths", "钢铁": "steel", "国产替代": "domestic substitution",
    "进口替代": "import substitution", "汽车": "automobiles", "新能源": "new energy",
    "芯片": "chips", "电动车": "electric vehicles", "中国制造": "Made in China",
}


def fig14_keyword_means():
    """图14: 扫一遍数据算每个关键词的平均情感分, 缺数据的关键词直接跳过。"""
    want = set(sum(FIG14_PANELS.values(), []))
    cnt_sum = {}
    for chunk in pd.read_csv(C.SENTIMENT_OUTPUT, usecols=[C.COL_KEYWORD, C.COL_SENT],
                             chunksize=200000, low_memory=False):
        chunk = chunk[chunk[C.COL_KEYWORD].isin(want)]
        chunk[C.COL_SENT] = pd.to_numeric(chunk[C.COL_SENT], errors="coerce")
        g = chunk.groupby(C.COL_KEYWORD)[C.COL_SENT].agg(["count", "sum"])
        for kw, row in g.iterrows():
            n0, s0 = cnt_sum.get(kw, (0, 0.0))
            cnt_sum[kw] = (n0 + int(row["count"]), s0 + float(row["sum"]))
    means = {kw: s / n for kw, (n, s) in cnt_sum.items() if n > 0}
    missing = sorted(want - set(means))
    if missing:
        print("图14 里这几个关键词数据集中没有, 跳过: " + "、".join(missing))
    return means


def draw_fig14(means):
    # 2x2 横条图, 和论文原图一样: 正向蓝、负向红, 按均值从大到小排, 0.5 处画虚线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Keyword Sentiment Mean by Cluster", fontsize=17)
    red, blue = "#F09A9A", "#93BCE5"
    for ax, cl in zip(axes.flat, ["Strategic", "Livelihood", "Trade", "Technology"]):
        pairs = [(kw, means[kw]) for kw in FIG14_PANELS[cl] if kw in means]
        pairs.sort(key=lambda x: x[1])   # barh 最下面最短, 所以升序
        kws = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        colors = [blue if v >= 0.5 else red for v in vals]
        labels = [f"{kw} ({FIG14_EN[kw]})" for kw in kws]
        y = np.arange(len(kws))
        ax.barh(y, vals, color=colors, edgecolor="white", height=0.82)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0.5, color="0.25", ls="--", lw=1.1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Sentiment Mean")
        ax.set_title(cl, fontsize=13)
        ax.grid(axis="x", alpha=0.3)
        ax.legend(handles=[Patch(facecolor=red, label="Negative"),
                           Patch(facecolor=blue, label="Positive")],
                  loc="center right", fontsize=9, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "Fig14_keyword_cluster_sentiment.png", dpi=C.FIG_DPI)
    plt.close(fig)


def aggregate(start=None, end=None):
    """分块聚合: (省, 簇) -> 发帖数与情感。不给时间范围时默认整个分析窗口。"""
    start = start or C.ANALYSIS_START
    end = end or C.ANALYSIS_END
    agg = {}
    for chunk in pd.read_csv(C.SENTIMENT_OUTPUT,
                             usecols=[C.COL_TIME, C.COL_IP, C.COL_KEYWORD, C.COL_SENT],
                             chunksize=100000, low_memory=False):
        t = pd.to_datetime(chunk[C.COL_TIME], errors="coerce")
        m = (t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end) + pd.Timedelta(days=1))
        chunk = chunk.loc[m]
        chunk["_prov"] = chunk[C.COL_IP].map(C.normalize_province)
        chunk = chunk[chunk["_prov"].notna()]
        chunk["_cl"] = chunk[C.COL_KEYWORD].map(KW2CLUSTER)
        chunk["_cl"] = chunk["_cl"].fillna("Other")
        for (p, cl), g in chunk.groupby(["_prov", "_cl"]):
            a = agg.setdefault(p, {})
            if cl == "Other":
                cl = None
            a[cl] = a.get(cl, {"n": 0, "sum": 0.0})
            a[cl]["n"] += len(g)
            a[cl]["sum"] += g[C.COL_SENT].sum()
    return agg


def build_province_table(agg):
    """省级: y=平均情感, X=四簇占比"""
    rows = []
    for prov, d in agg.items():
        tot_all = sum(v["n"] for v in d.values())
        row = {"province": prov, "n_posts": tot_all}
        sent_sum = sum(v["sum"] for v in d.values())
        row["mean_sent"] = sent_sum / tot_all if tot_all else np.nan
        for cl in C.CLUSTER_ORDER:
            row[f"share_{cl}"] = d.get(cl, {"n": 0})["n"] / tot_all if tot_all else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def table5_text():
    """Table 5: 四簇频次 (多标签口径——帖子正文出现该簇任一关键词即计入一次,
    一帖可同时计入多簇; 簇频次总和可大于帖子数, 与论文 Table 5 口径一致)"""
    import re
    patterns = {cl: re.compile("|".join(map(re.escape, kws)))
                for cl, kws in C.KEYWORD_CLUSTERS.items()}
    tot = {cl: 0 for cl in C.CLUSTER_ORDER}
    n_all = 0
    for chunk in pd.read_csv(C.SENTIMENT_OUTPUT, usecols=[C.COL_TEXT],
                             chunksize=100000, low_memory=False):
        txt = chunk[C.COL_TEXT].fillna("").astype(str)
        n_all += len(txt)
        for cl, pat in patterns.items():
            tot[cl] += int(txt.str.contains(pat, regex=True).sum())
    t5 = pd.DataFrame({"cluster": list(tot), "n_posts": list(tot.values())})
    t5["share_of_all_posts"] = t5["n_posts"] / n_all
    print(f"\n[Table 5] 四主题簇频次 (多标签, 总帖数 {n_all:,})")
    print(t5.to_string(index=False))
    t5.to_csv(OUT / "table5_cluster_frequency.csv", index=False)
    return t5


def morans_i(g, y):
    from esda import Moran
    from libpysal.weights import Queen
    w = Queen.from_dataframe(g)
    w.transform = "r"
    mi = Moran(y, w)
    return mi, w


def run_ols(X, y):
    import statsmodels.api as sm
    Xc = sm.add_constant(X)
    m = sm.OLS(y, Xc).fit()
    return m


def run_gwr(coords, X, y):
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW
    y = np.asarray(y).reshape(-1, 1)
    sel = Sel_BW(coords, y, X, fixed=False, kernel="bisquare")
    # 数据少的时间窗里, 带宽取得太小会让局部回归矩阵奇异,
    # 遇到这种情况就把搜索下限往上抬一点再试, 直到能算出来
    bw_min = max(X.shape[1] + 2, 2)
    while True:
        try:
            # mgwr 默认自适应下限 40+2p 会超过 31 个省, 显式给出最近邻数搜索范围
            bw = sel.search(criterion="AICc", bw_min=bw_min, bw_max=len(y))
            m = GWR(coords, y, X, bw=bw, fixed=False, kernel="bisquare").fit()
            break
        except np.linalg.LinAlgError:
            bw_min += 1
            if bw_min >= len(y):
                raise
            print(f"  带宽下限 {bw_min - 1} 时矩阵奇异, 改用 {bw_min} 重试...")
    print(f"最优带宽 (adaptive nearest neighbors, AICc): {bw}")
    return m, bw


def plot_map(g, column, fname, title, cmap="RdBu_r", center=None):
    fig, ax = plt.subplots(figsize=(12, 9))
    kw = {}
    if center is not None:
        kw = {"vmin": -abs(g[column]).max(), "vmax": abs(g[column]).max()}
    g.plot(ax=ax, column=column, cmap=cmap, legend=True,
           missing_kwds={"color": "#D3D3D3", "label": "Missing"},
           edgecolor="white", linewidth=0.5, legend_kwds={"shrink": 0.6}, **kw)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=C.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def fit_window(g_all, prov):
    """对一个时间窗口的省级表跑 GWR, 返回带系数列的地图 frame 和模型结果。
    某些短窗口里个别主题簇全省都没帖子 (占比恒为 0), 这种列放进去矩阵会奇异,
    所以这一窗口就不带它玩, 对应系数留空, 画图时灰显。"""
    gm = g_all.merge(prov, on="province", how="left")
    gm = gm[gm["name"].isin(C.MAINLAND_PROVINCES)].copy()
    gm = gm.dropna(subset=["mean_sent"]).reset_index(drop=True)
    share_cols = [f"share_{cl}" for cl in C.CLUSTER_ORDER]
    X_all = gm[share_cols].values.astype(float)
    # 短窗口里可能某个簇全省都是 0 帖 (占比恒为 0), 而四个占比本身又近似加起来等于 1,
    # 和截距放一起会共线, 矩阵奇异。这里用 QR 列主元挑出彼此线性独立的列,
    # 被剔掉的簇这一阶段就估不出系数, 后面画成灰色。
    X_std = np.zeros_like(X_all)
    sd = X_all.std(axis=0)
    ok = sd > 1e-12
    X_std[:, ok] = X_all[:, ok] / sd[ok]
    from scipy.linalg import qr as _qr
    _, _R, _piv = _qr(X_std, pivoting=True)
    tol = X_std.shape[0] * np.finfo(float).eps * (abs(_R[0, 0]) if _R.size else 1.0)
    rank = int(np.sum(np.abs(np.diag(_R)) > tol))
    keep_idx = sorted(int(j) for j in _piv[:rank])
    dropped = [C.CLUSTER_ORDER[j] for j in range(X_all.shape[1]) if j not in keep_idx]
    if dropped:
        print(f"  该窗口 {dropped} 簇共线或缺帖子, 这一阶段 GWR 里剔除")
    X = X_all[:, keep_idx]
    y = gm["mean_sent"].values
    g_proj = gm.to_crs("EPSG:4547")
    coords = np.array([(pt.x, pt.y) for pt in g_proj.geometry.centroid])
    gwr, bw = run_gwr(coords, X, y)
    gm["coef_Intercept"] = gwr.params[:, 0]
    for cl in C.CLUSTER_ORDER:
        gm[f"coef_{cl}"] = np.nan
    for k, j in enumerate(keep_idx, start=1):
        gm[f"coef_{C.CLUSTER_ORDER[j]}"] = gwr.params[:, k]
    gm["local_R2"] = gwr.localR2
    return gm, gwr, bw


def draw_coef_map(g_all, ax, col, sub_title):
    """在 ax 上画一张红蓝发散系数图 (0 居中), 自带 colorbar; 缺数据省份灰色。
    整列都估不出来时 (比如该阶段某个簇根本没帖子) 就只画灰底 + N/A。"""
    vmax = float(np.nanmax(np.abs(g_all[col])))
    if not np.isfinite(vmax) or vmax == 0:
        g_all.plot(ax=ax, color="#D3D3D3", edgecolor="white", linewidth=0.4)
        ax.set_title(sub_title, fontsize=10, fontstyle="italic")
        ax.axis("off")
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center",
                va="center", fontsize=16, color="#808080")
        return
    g_all.plot(ax=ax, column=col, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
               edgecolor="white", linewidth=0.4,
               missing_kwds={"color": "#D3D3D3"})
    ax.set_title(sub_title, fontsize=10, fontstyle="italic")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                               norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    fig = ax.figure
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)


def panel_2x2(g_all, subfig, coef_cols, group_title):
    """在一个 subfigure 里画 2x2 四张系数图。"""
    axes = np.array([[subfig.add_subplot(2, 2, 1), subfig.add_subplot(2, 2, 2)],
                     [subfig.add_subplot(2, 2, 3), subfig.add_subplot(2, 2, 4)]])
    for ax, col in zip(axes.flat, coef_cols):
        draw_coef_map(g_all, ax, col, col.replace("coef_", "").replace("diff_", ""))
    subfig.suptitle(group_title, fontsize=11)


def main():
    import geopandas as gpd
    OUT.mkdir(parents=True, exist_ok=True)
    print("聚合省级 x 簇 统计...")
    agg = aggregate()
    t5 = table5_text()
    prov = build_province_table(agg)
    prov.to_csv(OUT / "province_cluster_stats.csv", index=False)

    g = gpd.read_file(C.CN_SHP, engine="pyogrio")
    inv = {v: k for k, v in C.PROVINCE_EN.items()}
    g["province"] = g["name"].map(inv)
    g = g.merge(prov, on="province", how="left")
    g_main = g[g["name"].isin(C.MAINLAND_PROVINCES)].copy()
    g_main = g_main.dropna(subset=["mean_sent"]).reset_index(drop=True)
    print(f"GWR 样本省份: {len(g_main)}")

    # ---- Moran's I ----
    y = g_main["mean_sent"].values
    try:
        mi, w = morans_i(g_main, y)
        print(f"\n[Table 6] Overall Moran's I = {mi.I:.4f} (p={mi.p_norm:.4f})")
    except Exception as e:
        print(f"Moran's I 失败 (用 KNN 替代): {e}")
        from esda import Moran
        from libpysal.weights import KNN
        w = KNN.from_dataframe(g_main, k=4); w.transform = "r"
        mi = Moran(y, w)
        print(f"\n[Table 6] Overall Moran's I = {mi.I:.4f} (p={mi.p_norm:.4f})")

    # ---- OLS ----
    X = g_main[[f"share_{cl}" for cl in C.CLUSTER_ORDER]].values
    ols = run_ols(X, y)
    print(f"[Table 6] OLS R2 = {ols.rsquared:.4f}")
    print("\n[Table 7] OLS 系数")
    print(ols.summary2().tables[1].round(4).to_string())

    # ---- GWR ----
    g_main, gwr, bw = fit_window(g[["name", "province", "geometry"]], prov)
    print(f"[Table 6] GWR R2 = {gwr.R2:.4f}, 带宽 = {bw}")

    varnames = ["Intercept"] + C.CLUSTER_ORDER
    res = g_main[["province", "name"] + [f"coef_{v}" for v in varnames] + ["local_R2"]]
    res.to_csv(OUT / "table9_province_gwr_coefficients.csv", index=False)

    print("\n[Table 8] GWR 系数描述统计")
    t8 = res[[f"coef_{v}" for v in varnames] + ["local_R2"]].describe().loc[["min", "mean", "max"]]
    print(t8.round(4).to_string())
    t8.to_csv(OUT / "table8_gwr_coef_summary.csv")

    t6 = pd.DataFrame({
        "metric": ["Moran's I", "Moran's I p", "OLS R2", "GWR R2", "GWR bandwidth"],
        "value": [round(mi.I, 4), round(float(mi.p_norm), 4), round(ols.rsquared, 4),
                  round(gwr.R2, 4), bw],
    })
    t6.to_csv(OUT / "table6_overall_metrics.csv", index=False)

    # ---- 三个事件阶段分别跑 GWR (Fig 17 要画阶段间系数差值) ----
    g_base = g[["name", "province", "geometry"]]
    phase_gm = {}
    phase_rows = []
    for ph in C.PHASES:
        print(f"\n阶段 {ph['name']} ({ph['start']} ~ {ph['end']}) 聚合 + GWR...")
        prov_p = build_province_table(aggregate(ph["start"], ph["end"]))
        gm_p, gwr_p, bw_p = fit_window(g_base, prov_p)
        print(f"{ph['name']}: GWR R2 = {gwr_p.R2:.4f}, 带宽 = {bw_p}, 省份 = {len(gm_p)}")
        phase_gm[ph["name"]] = gm_p
        tmp = gm_p[["province"] + [f"coef_{v}" for v in C.CLUSTER_ORDER] + ["local_R2"]].copy()
        tmp.insert(0, "phase", ph["name"])
        phase_rows.append(tmp)
    pd.concat(phase_rows, ignore_index=True).to_csv(
        OUT / "table9_phase_gwr_coefficients.csv", index=False)

    # 画图底图: 全部省界 + 左连接系数 (港澳台等没数据的省自动灰显)
    coef_cols = [f"coef_{v}" for v in C.CLUSTER_ORDER]
    g_overall = g_base.merge(
        g_main[["province", "coef_Intercept"] + coef_cols + ["local_R2"]],
        on="province", how="left")
    g_p1 = g_base.merge(phase_gm["Phase 1"][["province"] + coef_cols], on="province", how="left")
    g_p2 = g_base.merge(phase_gm["Phase 2"][["province"] + coef_cols], on="province", how="left")
    g_p3 = g_base.merge(phase_gm["Phase 3"][["province"] + coef_cols], on="province", how="left")
    g_d12 = g_base.copy()   # Peak - Start
    g_d23 = g_base.copy()   # De-escalation - Peak
    for col in coef_cols:
        m12 = g_p2[["province", col]].merge(g_p1[["province", col]], on="province",
                                            suffixes=("_p2", "_p1"))
        m12[col] = m12[f"{col}_p2"] - m12[f"{col}_p1"]
        g_d12 = g_d12.merge(m12[["province", col]], on="province", how="left")
        m23 = g_p3[["province", col]].merge(g_p2[["province", col]], on="province",
                                            suffixes=("_p3", "_p2"))
        m23[col] = m23[f"{col}_p3"] - m23[f"{col}_p2"]
        g_d23 = g_d23.merge(m23[["province", col]], on="province", how="left")

    # ---- Fig 14 四个主题簇的关键词情感均值横条图 ----
    print("\n绘图...")
    draw_fig14(fig14_keyword_means())

    # ---- Fig 15 局部 R2 地图 (和论文原图一样用红蓝发散色带, 0 居中) ----
    r2max = float(np.nanmax(np.abs(g_overall["local_R2"])))
    fig, ax = plt.subplots(figsize=(10, 9))
    g_overall.plot(ax=ax, column="local_R2", cmap="RdBu_r", vmin=-r2max, vmax=r2max,
                   edgecolor="black", linewidth=0.4,
                   missing_kwds={"color": "#D3D3D3"})
    ax.set_title("Overall Local R²", fontsize=16)
    ax.axis("off")
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r",
                                       norm=plt.Normalize(vmin=-r2max, vmax=r2max)),
                 ax=ax, shrink=0.85, pad=0.03)
    fig.tight_layout()
    fig.savefig(OUT / "Fig15_gwr_localR2_overall.png", dpi=C.FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 单个变量的系数地图 (附产物, 方便细看)
    for v in varnames:
        plot_map(g_overall, f"coef_{v}", f"gwr_coef_{v}_overall.png",
                 f"GWR Coefficient: {v}", cmap="RdBu_r", center=0)

    # 论文图里的摆放顺序: 上面 Strategic/Livelihood, 下面 Trade/Technology
    plot_cols = [f"coef_{v}" for v in ["Strategic", "Livelihood", "Trade", "Technology"]]

    # ---- Fig 16 整体四系数 2x2 合成图 ----
    fig = plt.figure(figsize=(13, 11))
    sfig = fig.subfigures(1, 1)
    panel_2x2(g_overall, sfig, plot_cols, "")
    sfig.suptitle("Overall GWR Coefficients", fontsize=15, y=0.99)
    fig.savefig(OUT / "Fig16_gwr_coef_overall.png", dpi=C.FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 17 三组 2x2: 阶段差值 x2 + 整体系数 ----
    fig = plt.figure(figsize=(27, 9))
    subs = fig.subfigures(1, 3, wspace=0.18)
    panel_2x2(g_d12, subs[0], plot_cols,
              "Conflict Peak - Conflict Start GWR Coefficient Difference")
    panel_2x2(g_d23, subs[1], plot_cols,
              "De-escalation - Conflict Peak GWR Coefficient Difference")
    panel_2x2(g_overall, subs[2], plot_cols, "Conflict GWR Coefficients")
    fig.savefig(OUT / "Fig17_gwr_coef_composite_horizontal.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig17_gwr_coef_composite_horizontal.png",
                dpi=C.FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print("\n全部输出 ->", OUT)


if __name__ == "__main__":
    main()
