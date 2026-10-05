#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 跨区域对比的图都在这个文件里 (Fig 5-13):
# Fig 5/6 中国和海外每天的发帖量, 6 是各自除以自己最大值的归一化版
# Fig 7   中国和海外每天的平均情感分
# Fig 8/9 换成中国 vs 美国, 同样的两张
# 另外单独存了中美情感构成、事件窗口和差异检验的表和一张汇总图
# Fig 10/11 海外各国总发帖量、每百万人口发帖量 (世界地图填色)
# Fig 12/13 国内各省总发帖量、每千万人口发帖量 (中国地图填色)
# 直接 python cross_regional.py 跑
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

C.setup_plot()
OUT = C.OUTPUT_DIR / "regional"

# 省级人口 (2020 普查, 百万) —— 用于 Fig 13 人均归一化
PROVINCE_POP_M = {
    "北京": 21.89, "天津": 13.87, "河北": 74.61, "山西": 34.92, "内蒙古": 24.05,
    "辽宁": 42.59, "吉林": 24.07, "黑龙江": 31.85, "上海": 24.87, "江苏": 84.75,
    "浙江": 64.57, "安徽": 61.03, "福建": 41.54, "江西": 45.19, "山东": 101.53,
    "河南": 99.37, "湖北": 57.75, "湖南": 66.44, "广东": 126.01, "广西": 50.13,
    "海南": 10.08, "重庆": 32.05, "四川": 83.67, "贵州": 38.56, "云南": 47.21,
    "西藏": 3.65, "陕西": 39.53, "甘肃": 25.02, "青海": 5.92, "宁夏": 7.20, "新疆": 25.85,
}
# 海外国家人口 (百万, 约值) —— 用于 Fig 11
COUNTRY_POP_M = {
    "美国": 331.9, "日本": 125.7, "韩国": 51.7, "英国": 67.3, "加拿大": 38.2,
    "澳大利亚": 25.7, "德国": 83.2, "法国": 67.8, "新加坡": 5.9, "马来西亚": 32.4,
    "泰国": 69.8, "越南": 97.3, "印度尼西亚": 273.5, "菲律宾": 109.6, "印度": 1380.0,
    "俄罗斯": 144.1, "巴西": 212.6, "意大利": 59.6, "西班牙": 47.4, "荷兰": 17.4,
    "瑞典": 10.4, "瑞士": 8.6, "新西兰": 5.1, "阿联酋": 9.9, "沙特阿拉伯": 34.8,
    "土耳其": 84.3, "墨西哥": 128.9, "阿根廷": 45.4, "南非": 59.3, "埃及": 102.3,
    "尼日利亚": 206.1, "巴基斯坦": 220.9, "孟加拉国": 164.7, "柬埔寨": 16.7,
    "老挝": 7.3, "缅甸": 54.4, "哈萨克斯坦": 18.8, "乌兹别克斯坦": 34.2,
    "安哥拉": 32.9, "比利时": 11.6, "乌克兰": 44.1, "摩洛哥": 36.9,
    "白俄罗斯": 9.4, "埃塞俄比亚": 114.9, "莫桑比克": 31.3, "马耳他": 0.5,
    "希腊": 10.7, "波兰": 37.9, "厄瓜多尔": 17.6, "马尔代夫": 0.5, "秘鲁": 32.9,
    "波斯尼亚和黑塞哥维那": 3.3, "马拉维": 19.1, "科威特": 4.3, "罗马尼亚": 19.2,
    "以色列": 9.2, "冰岛": 0.4, "纳米比亚": 2.5, "奥地利": 8.9,
    "刚果民主共和国": 89.6, "肯尼亚": 53.8, "葡萄牙": 10.3, "丹麦": 5.8,
    "巴林": 1.7, "坦桑尼亚": 59.7, "波多黎各": 3.3, "科特迪瓦": 26.4,
    "保加利亚": 6.9, "阿富汗": 38.9, "利比亚": 6.9, "阿曼": 5.1,
    "阿尔巴尼亚": 2.9, "阿塞拜疆": 10.1, "津巴布韦": 14.9, "古巴": 11.3,
    "蒙古": 3.3, "匈牙利": 9.8, "卢旺达": 12.9, "尼泊尔": 29.1, "挪威": 5.4,
}
US_NAMES = {"美国"}

# 国家中文名 -> 英文名 (条形图标签, 与论文英文图表一致)
COUNTRY_EN = {
    "美国": "United States", "加拿大": "Canada", "日本": "Japan", "澳大利亚": "Australia",
    "安哥拉": "Angola", "新加坡": "Singapore", "马来西亚": "Malaysia", "比利时": "Belgium",
    "英国": "United Kingdom", "泰国": "Thailand", "法国": "France", "德国": "Germany",
    "韩国": "South Korea", "乌克兰": "Ukraine", "荷兰": "Netherlands", "越南": "Vietnam",
    "阿联酋": "UAE", "新西兰": "New Zealand", "摩洛哥": "Morocco", "西班牙": "Spain",
    "俄罗斯": "Russia", "意大利": "Italy", "瑞士": "Switzerland", "白俄罗斯": "Belarus",
    "埃塞俄比亚": "Ethiopia", "印度尼西亚": "Indonesia", "柬埔寨": "Cambodia",
    "南非": "South Africa", "菲律宾": "Philippines", "土耳其": "Turkey", "墨西哥": "Mexico",
    "挪威": "Norway", "缅甸": "Myanmar", "尼日利亚": "Nigeria", "瑞典": "Sweden",
    "莫桑比克": "Mozambique", "埃及": "Egypt", "马耳他": "Malta", "希腊": "Greece",
    "波兰": "Poland", "厄瓜多尔": "Ecuador", "沙特阿拉伯": "Saudi Arabia",
    "马尔代夫": "Maldives", "秘鲁": "Peru", "波斯尼亚和黑塞哥维那": "Bosnia & Herzegovina",
    "马拉维": "Malawi", "科威特": "Kuwait", "罗马尼亚": "Romania", "以色列": "Israel",
    "阿根廷": "Argentina", "哈萨克斯坦": "Kazakhstan", "冰岛": "Iceland",
    "纳米比亚": "Namibia", "奥地利": "Austria", "刚果民主共和国": "DR Congo",
    "巴西": "Brazil", "肯尼亚": "Kenya", "葡萄牙": "Portugal", "丹麦": "Denmark",
    "巴林": "Bahrain", "坦桑尼亚": "Tanzania", "波多黎各": "Puerto Rico",
    "科特迪瓦": "Côte d'Ivoire", "保加利亚": "Bulgaria", "阿富汗": "Afghanistan",
    "利比亚": "Libya", "印度": "India", "乌兹别克斯坦": "Uzbekistan", "阿曼": "Oman",
    "阿尔巴尼亚": "Albania", "阿塞拜疆": "Azerbaijan", "老挝": "Laos",
    "津巴布韦": "Zimbabwe", "古巴": "Cuba", "蒙古": "Mongolia", "匈牙利": "Hungary",
    "卢旺达": "Rwanda", "巴基斯坦": "Pakistan", "尼泊尔": "Nepal",
}


def aggregate():
    """分块聚合: 日 x {china,overseas,US} 量与情感; 国家/省份总量"""
    daily = {}
    country_tot = {}
    province_tot = {}
    for chunk in pd.read_csv(C.SENTIMENT_OUTPUT,
                             usecols=[C.COL_TIME, C.COL_IP, C.COL_SENT],
                             chunksize=100000, low_memory=False):
        t = pd.to_datetime(chunk[C.COL_TIME], errors="coerce")
        m = (t >= pd.Timestamp(C.ANALYSIS_START)) & \
            (t <= pd.Timestamp(C.ANALYSIS_END) + pd.Timedelta(days=1))
        cls = chunk.loc[m, C.COL_IP].map(C.classify_ip)
        chunk = chunk.loc[m].assign(_d=t[m].dt.date,
                                    _r=[c[0] for c in cls],
                                    _n=[c[1] for c in cls])
        for (d, r), g in chunk.groupby(["_d", "_r"]):
            if r is None:
                continue
            a = daily.setdefault((d, r), {"n": 0, "sum": 0.0})
            a["n"] += len(g); a["sum"] += g[C.COL_SENT].sum()
        for (d, n), g in chunk.groupby(["_d", "_n"]):
            if n in US_NAMES:
                a = daily.setdefault((d, "us"), {"n": 0, "sum": 0.0})
                a["n"] += len(g); a["sum"] += g[C.COL_SENT].sum()
        for (r, n), cnt in chunk.groupby(["_r", "_n"]).size().items():
            if r == "china":
                province_tot[n] = province_tot.get(n, 0) + cnt
            elif r == "overseas":
                country_tot[n] = country_tot.get(n, 0) + cnt
    return daily, country_tot, province_tot


def daily_df(daily, key):
    rows = [(d, a["n"], a["sum"] / a["n"] if a["n"] else np.nan)
            for (d, r), a in daily.items() if r == key]
    df = pd.DataFrame(rows, columns=["date", "n", "sent"]).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


def fig_counts(cn, ov, fname):
    """Fig 5: 原始日发帖量, 中国浅蓝 / 海外浅橙 (对齐论文原图)。"""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(cn["date"], cn["n"], color="#A6C8ED", lw=1.8, label="China Daily Posts")
    ax.plot(ov["date"], ov["n"], color="#F7C076", lw=1.8, label="Overseas Daily Posts")
    ax.set_ylabel("Post Count")
    ax.set_xlabel("Date")
    ax.set_title("China vs Overseas Daily Posts (2025-01 to 2025-07)")
    ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=C.FIG_DPI); plt.close(fig)


def fig_normalized_counts(cn, other, fname, other_label, title):
    """Fig 6/9: Max=1 归一化 (各自除以自身峰值), 中国蓝 / 对方橙。"""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    cn_y = cn["n"] / max(cn["n"].max(), 1)
    ot_y = other["n"] / max(other["n"].max(), 1)
    ax.plot(cn["date"], cn_y, color="#1f77b4", lw=2.0, label="China Normalized (Max=1)")
    ax.plot(other["date"], ot_y, color="#ff7f0e", lw=2.0,
            label=f"{other_label} Normalized (Max=1)")
    ax.set_ylabel("Normalized Posts")
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=C.FIG_DPI); plt.close(fig)


def fig_sentiment(cn, other, fname, other_label, title):
    """Fig 7/8: 逐日平均情感原始日线 (不做滑动平均), 中国蓝 / 对方橙。"""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(cn["date"], cn["sent"], color="#1f77b4", lw=1.8, label="China Daily Sentiment")
    ax.plot(other["date"], other["sent"], color="#ff7f0e", lw=1.8,
            label=f"{other_label} Daily Sentiment")
    ax.set_ylabel("Average Sentiment")
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=C.FIG_DPI); plt.close(fig)


def collect_region_sentiment():
    """独立收集 China / United States / Overseas 三类帖子的 (日期, 情感分, 类别)。"""
    frames = []
    cols = [C.COL_TIME, C.COL_IP, C.COL_SENT, C.COL_SENT_CAT]
    for chunk in pd.read_csv(C.SENTIMENT_OUTPUT, usecols=cols,
                             chunksize=100000, low_memory=False):
        t = pd.to_datetime(chunk[C.COL_TIME], errors="coerce")
        m = (t >= pd.Timestamp(C.ANALYSIS_START)) & \
            (t <= pd.Timestamp(C.ANALYSIS_END) + pd.Timedelta(days=1))
        chunk, t = chunk.loc[m], t.loc[m]
        cls = chunk[C.COL_IP].map(C.classify_ip)
        region = pd.Series(np.where(cls.map(lambda z: z[0]) == "china", "China",
                           np.where(cls.map(lambda z: z[1]) == "美国", "United States",
                           np.where(cls.map(lambda z: z[0]) == "overseas", "Overseas", np.nan))),
                           index=chunk.index)
        sub = pd.DataFrame({"date": t.dt.normalize().values,
                            "region": region.values,
                            "score": pd.to_numeric(chunk[C.COL_SENT], errors="coerce").values,
                            "category": chunk[C.COL_SENT_CAT].values}, index=chunk.index)
        frames.append(sub.dropna(subset=["region"]))
    return pd.concat(frames, ignore_index=True)


def region_sentiment_summary(df):
    """中国与美国 (含海外整体) 的独立情感汇总: 总体统计 + 事件窗口统计 + 差异检验 + 图。"""
    from scipy import stats as st

    regions = ["China", "United States", "Overseas"]
    cats = ["Negative", "Neutral", "Positive"]

    # ---- 总体描述统计 ----
    rows = []
    for r in regions:
        g = df[df["region"] == r]
        s = g["score"].dropna()
        row = {"region": r, "n_posts": len(g),
               "mean": round(s.mean(), 4), "median": round(s.median(), 4),
               "std": round(s.std(), 4)}
        cnt = g["category"].value_counts()
        for c in cats:
            row[f"{c}_n"] = int(cnt.get(c, 0))
            row[f"{c}_share"] = round(cnt.get(c, 0) / max(len(g), 1), 4)
        rows.append(row)
    overall = pd.DataFrame(rows)
    overall.to_csv(OUT / "table_china_us_sentiment.csv", index=False)
    print("\n[中国/美国 独立情感汇总]")
    print(overall.to_string(index=False))

    # ---- 三事件窗口 (中心 ±15 天) 均值与类别占比 ----
    wins = [("Overall", C.ANALYSIS_START, "2025-07-07")]
    for ph in C.PHASES:
        c = pd.Timestamp(ph["center"])
        wins.append((ph["name"], (c - pd.Timedelta(days=C.PHASE_WINDOW_DAYS)).strftime("%Y-%m-%d"),
                     (c + pd.Timedelta(days=C.PHASE_WINDOW_DAYS)).strftime("%Y-%m-%d")))
    pr_rows = []
    for wname, a, b in wins:
        w = df[(df["date"] >= pd.Timestamp(a)) & (df["date"] <= pd.Timestamp(b))]
        for r in ["China", "United States"]:
            g = w[w["region"] == r]
            if not len(g):
                continue
            cnt = g["category"].value_counts()
            pr_rows.append({"window": wname, "region": r, "n_posts": len(g),
                            "mean_sentiment": round(g["score"].mean(), 4),
                            "neg_share": round(cnt.get("Negative", 0) / len(g), 4),
                            "neu_share": round(cnt.get("Neutral", 0) / len(g), 4),
                            "pos_share": round(cnt.get("Positive", 0) / len(g), 4)})
    by_phase = pd.DataFrame(pr_rows)
    by_phase.to_csv(OUT / "table_china_us_sentiment_by_phase.csv", index=False)

    # ---- 中国 vs 美国 差异检验 (Welch t + Mann-Whitney U) ----
    x = df.loc[df["region"] == "China", "score"].dropna()
    y = df.loc[df["region"] == "United States", "score"].dropna()
    tstat, tp = st.ttest_ind(x, y, equal_var=False)
    ustat, up = st.mannwhitneyu(x, y, alternative="two-sided")
    tests = pd.DataFrame([{"test": "Welch t-test", "statistic": round(tstat, 3),
                           "p_value": tp, "China_mean": round(x.mean(), 4),
                           "US_mean": round(y.mean(), 4), "China_n": len(x), "US_n": len(y)},
                          {"test": "Mann-Whitney U", "statistic": round(ustat, 1),
                           "p_value": up, "China_mean": round(x.mean(), 4),
                           "US_mean": round(y.mean(), 4), "China_n": len(x), "US_n": len(y)}])
    tests.to_csv(OUT / "table_china_us_tests.csv", index=False)
    print("\n[中国 vs 美国 情感差异检验]")
    print(tests.to_string(index=False))

    # ---- 图: 左=类别构成, 右=各窗口均值 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    xpos = np.arange(len(regions)); w = 0.25
    palette = {"Negative": C.COLOR_NEG, "Neutral": C.COLOR_NEU, "Positive": C.COLOR_POS}
    for i, c in enumerate(cats):
        ax.bar(xpos + (i - 1) * w, overall[f"{c}_share"], w, label=c, color=palette[c])
    ax.set_xticks(xpos); ax.set_xticklabels(regions)
    ax.set_ylabel("Share of posts"); ax.set_ylim(0, 1)
    ax.set_title("Sentiment Composition: China vs US vs Overseas")
    ax.legend()

    ax = axes[1]
    pivot = by_phase.pivot(index="window", columns="region", values="mean_sentiment")
    order = [w[0] for w in wins if w[0] in pivot.index]
    pivot = pivot.loc[order]
    xp = np.arange(len(pivot))
    ax.bar(xp - 0.2, pivot["China"], 0.4, color="#1f77b4", label="China")
    ax.bar(xp + 0.2, pivot["United States"], 0.4, color="#ff7f0e", label="United States")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(xp); ax.set_xticklabels(order, rotation=15)
    ax.set_ylabel("Mean sentiment"); ax.set_ylim(0, 1)
    ax.set_title("Mean Sentiment by Event Window")
    ax.legend()
    fig.suptitle("Sentiment Analysis: China vs United States", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "Fig_china_us_sentiment_summary.png", dpi=C.FIG_DPI)
    plt.close(fig)
    print("[Fig] Fig_china_us_sentiment_summary.png")


# naturalearth_lowres 里的国名和我们用的英文国名有几个不一样, 单独对一下
NE_NAME_FIX = {
    "United States": "United States of America",
    "UAE": "United Arab Emirates",
    "Bosnia & Herzegovina": "Bosnia and Herz.",
    "DR Congo": "Dem. Rep. Congo",
}
WORLD_SHP = (Path("/home/dengjianwei/venvs/thesis/lib/python3.8/site-packages/"
                  "geopandas/datasets/naturalearth_lowres/naturalearth_lowres.shp"))
CHINA_POP_M = 1411.0   # 第七次人口普查, 百万


def fig_country_world(country_tot, china_total, per_million, fname, cbar_label,
                      vmin, vmax, ticks):
    """Fig 10/11: 世界地图填色 (Blues, 对数色标), 中国也算在内。"""
    import matplotlib.colors as mcolors
    import geopandas as gpd

    vals = {}
    for zh, cnt in country_tot.items():
        if per_million and zh not in COUNTRY_POP_M:
            continue
        en = COUNTRY_EN.get(zh, zh)
        en = NE_NAME_FIX.get(en, en)
        vals[en] = cnt / COUNTRY_POP_M[zh] if per_million else float(cnt)
    vals["China"] = china_total / CHINA_POP_M if per_million else float(china_total)

    world = gpd.read_file(WORLD_SHP, engine="pyogrio")
    world["value"] = world["name"].map(vals)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)

    fig, ax = plt.subplots(figsize=(13, 6.8))
    world.plot(ax=ax, color="#f2f2f2", edgecolor="#999999", linewidth=0.25)
    world.plot(ax=ax, column="value", cmap="Blues", norm=norm, edgecolor="#999999",
               linewidth=0.25, missing_kwds={"color": "#f2f2f2"})
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.axis("off")
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="Blues", norm=norm),
                        ax=ax, shrink=0.72, pad=0.02, ticks=ticks)
    cbar.set_label(cbar_label, fontsize=12)
    cbar.ax.set_yticklabels([str(t) for t in ticks])
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=C.FIG_DPI, bbox_inches="tight"); plt.close(fig)


def fig_province_map(province_tot, normalized, fname, cbar_label, vmax=None):
    """Fig 12/13: 省级 Blues 线性色阶地图。normalized=True 时按每千万人口归一化。"""
    import matplotlib.colors as mcolors
    import geopandas as gpd
    g = gpd.read_file(C.CN_SHP, engine="pyogrio")
    vals = {}
    for prov, cnt in province_tot.items():
        en = C.PROVINCE_EN.get(prov)
        if not en:
            continue
        if normalized and prov in PROVINCE_POP_M:
            v = cnt / PROVINCE_POP_M[prov] * 10.0   # 每千万人口
        else:
            v = float(cnt)
        vals[en] = v
    g["value"] = g["name"].map(vals)
    if vmax is None:
        vmax = float(pd.Series(vals).max())
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(11, 9))
    g.plot(ax=ax, column="value", cmap="Blues", norm=norm, edgecolor="#cccccc",
           linewidth=0.4, missing_kwds={"color": "#f2f2f2"})
    ax.axis("off")
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="Blues", norm=norm),
                        ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=C.FIG_DPI, bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("聚合 (分块读取)...")
    daily, country_tot, province_tot = aggregate()
    print(f"海外国家数: {len(country_tot)}, 省份数: {len(province_tot)}")
    pd.Series(country_tot).sort_values(ascending=False).to_csv(OUT / "country_total_posts.csv")
    pd.Series(province_tot).sort_values(ascending=False).to_csv(OUT / "province_total_posts.csv")

    cn, ov, us = daily_df(daily, "china"), daily_df(daily, "overseas"), daily_df(daily, "us")
    china_total = int(sum(province_tot.values()))
    print(f"中国总帖数: {china_total:,}")
    print("\n[Fig 5-13]")
    fig_counts(cn, ov, "Fig5_china_overseas_daily_counts.png")
    fig_normalized_counts(cn, ov, "Fig6_china_overseas_daily_counts_normalized.png",
                          "Overseas", "China vs Overseas Normalized Posts (2025-01 to 2025-07)")
    fig_sentiment(cn, ov, "Fig7_china_overseas_daily_sentiment.png", "Overseas",
                  "China vs Overseas Daily Sentiment (2025-01 to 2025-07)")
    if len(us):
        # Fig 8/9: 中国 vs 美国, 同样是原始日线 / Max=1 归一化
        fig_sentiment(cn, us, "Fig8_china_us_daily_sentiment.png", "US",
                      "China vs US Daily Sentiment (2025-01 to 2025-07)")
        fig_normalized_counts(cn, us, "Fig9_china_us_daily_counts_normalized.png",
                              "US", "China vs US Normalized Posts (2025-01 to 2025-07)")

    # 中国与美国的独立情感汇总分析 (总体构成/事件窗口/差异检验)
    region_df = collect_region_sentiment()
    region_sentiment_summary(region_df)

    # Fig 10/11: 世界地图 (总量 / 每百万人口), 色标和论文原图一致
    fig_country_world(country_tot, china_total, False,
                      "Fig10_overseas_country_totalposts.png",
                      "Amount of posts per country", 1, 200000,
                      [1, 10, 100, 1000, 10000, 100000, 200000])
    fig_country_world(country_tot, china_total, True,
                      "Fig11_overseas_country_posts_per_million.png",
                      "Posts per million people", 0.1, 100, [0.1, 1, 10, 100])
    # Fig 12/13: 省级地图, Blues 线性色阶
    fig_province_map(province_tot, False, "Fig12_china_province_totalposts.png",
                     "Amount of posts (province scale)")
    fig_province_map(province_tot, True, "Fig13_china_province_normalized_posts.png",
                     "Posts per 10 million people")
    print("全部输出 ->", OUT)


if __name__ == "__main__":
    main()
