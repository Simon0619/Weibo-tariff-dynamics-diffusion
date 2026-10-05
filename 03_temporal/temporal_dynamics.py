#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 时序部分的两张主图:
# Fig 3 上面是每天的平均情感分和发帖量, 黄底标出三个事件阶段;
#       下面三个小图是各事件前后大家一般在一天里的什么时间发帖
# Fig 4 三个事件各自前后 15 天的情感构成堆叠柱状图, 黑线是负面占比的 3 日滑动平均,
#       红色虚线是基线 + 0.20 的告警阈值
# 另外顺手算了下阶段自动检测和事件前后对比表, 结果都存到 temporal 文件夹
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

OUT = C.OUTPUT_DIR / "temporal"
USECOLS = [C.COL_TIME, C.COL_SENT, C.COL_SENT_CAT]


def load_daily():
    """分块聚合到 日 x 情感类别"""
    agg = {}
    hourly = {}
    win = (pd.Timestamp(C.ANALYSIS_START), pd.Timestamp(C.ANALYSIS_END))
    for chunk in pd.read_csv(C.SENTIMENT_OUTPUT, usecols=USECOLS, chunksize=100000, low_memory=False):
        t = pd.to_datetime(chunk[C.COL_TIME], errors="coerce")
        m = (t >= win[0]) & (t <= win[1] + pd.Timedelta(days=1))
        chunk = chunk.loc[m].assign(_d=t[m].dt.date, _h=t[m].dt.hour).dropna(subset=["_d"])
        for (d, cat), n in chunk.groupby(["_d", C.COL_SENT_CAT]).size().items():
            a = agg.setdefault(d, {"Positive": 0, "Neutral": 0, "Negative": 0, "sum": 0.0})
            a[cat] += n
        for d, s in chunk.groupby("_d")[C.COL_SENT].sum().items():
            agg[d]["sum"] += s
        for (d, h), n in chunk.groupby(["_d", "_h"]).size().items():
            hourly.setdefault(d, np.zeros(24, dtype=int))
            hourly[d][int(h)] += n
    rows = []
    for d, a in agg.items():
        tot = a["Positive"] + a["Neutral"] + a["Negative"]
        rows.append({"date": pd.Timestamp(d), "total": tot,
                     "Positive": a["Positive"], "Neutral": a["Neutral"], "Negative": a["Negative"],
                     "mean_sent": a["sum"] / tot if tot else np.nan})
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    daily["hourly"] = daily["date"].map(lambda x: hourly.get(x.date(), np.zeros(24, dtype=int)))
    return daily


def detect_phases(daily):
    """SMA-7 + tau 偏移检测; 基线 = 前 BASELINE_DAYS 天的负面比例均值"""
    daily = daily.copy()
    daily["neg_ratio"] = daily["Negative"] / daily["total"]
    base = daily["neg_ratio"].iloc[:C.BASELINE_DAYS].mean()
    daily["dev"] = daily["neg_ratio"] - base
    daily["sma7"] = daily["dev"].rolling(C.SMA_WINDOW, center=True).mean()
    print(f"基线负面比例 (前{C.BASELINE_DAYS}天): {base:.4f}, tau={C.TAU}")
    print("SMA-7 偏移超阈值区间:")
    over = daily[daily["sma7"].abs() > C.TAU]
    print(over[["date", "neg_ratio", "sma7"]].to_string(index=False) if len(over) else "  (无)")
    return daily


def fig3_overall(daily):
    """Fig 3: 主图(情感线+量柱+事件窗) + 底部3个小时分布子图"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 0.05, 1.2], hspace=0.35)
    ax = fig.add_subplot(gs[0, :])

    ax2 = ax.twinx()
    ax2.bar(daily["date"], daily["total"], color="#B0BEC5", alpha=0.5, width=1.0, label="Daily volume")
    ax2.set_ylabel("Daily post volume", color="#607D8B")
    ax2.tick_params(axis="y", labelcolor="#607D8B")

    ax.plot(daily["date"], daily["mean_sent"], color="#1565C0", lw=1.8, label="Daily mean sentiment")
    sma = daily["mean_sent"].rolling(7, center=True).mean()
    ax.plot(daily["date"], sma, color="#D32F2F", lw=2.2, ls="--", label="7-day SMA")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_ylabel("Mean sentiment score")
    ax.set_ylim(0.2, 0.8)

    for i, ph in enumerate(C.PHASES):
        s, e, c = pd.Timestamp(ph["start"]), pd.Timestamp(ph["end"]), pd.Timestamp(ph["center"])
        ax.axvspan(s, e, color="#FF8F00", alpha=0.15)
        ax.axvline(c, color="#FF8F00", ls="--", lw=1.2)
        ax.text(c, ax.get_ylim()[1] * 0.98, ph["name"], ha="center", va="top", fontsize=10,
                color="#E65100", fontweight="bold")

    ax.set_title("Overall Sentiment Dynamics and Event Windows (2025)", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

    for i, ph in enumerate(C.PHASES):
        axi = fig.add_subplot(gs[2, i])
        s, e, c = pd.Timestamp(ph["start"]), pd.Timestamp(ph["end"]), pd.Timestamp(ph["center"])
        win = daily[(daily["date"] >= c - pd.Timedelta(days=C.PHASE_WINDOW_DAYS)) &
                    (daily["date"] <= c + pd.Timedelta(days=C.PHASE_WINDOW_DAYS))]
        if len(win):
            hmat = np.stack(win["hourly"].values)
            hmean = hmat.mean(axis=0)
            axi.bar(range(24), hmean, color="#42A5F5", alpha=0.85)
            axi.set_title(f"{ph['name']} hourly posts\n({ph['center']}±{C.PHASE_WINDOW_DAYS}d)", fontsize=9)
            axi.set_xticks(range(0, 24, 6))
            axi.tick_params(labelsize=8)
        axi.set_xlabel("Hour", fontsize=8)

    fig.savefig(OUT / "Fig3_overall_sentiment.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig3_overall_sentiment.png", dpi=C.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 3 -> {OUT}")


def fig4_lifecycle(daily):
    """Fig 4: 三阶段生命周期 (堆叠比例柱 + 3日MA + 20% 阈值线)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    base_neg = (daily["Negative"] / daily["total"]).iloc[:C.BASELINE_DAYS].mean()

    for i, ph in enumerate(C.PHASES):
        ax = axes[i]
        c = pd.Timestamp(ph["center"])
        win = daily[(daily["date"] >= c - pd.Timedelta(days=C.PHASE_WINDOW_DAYS)) &
                    (daily["date"] <= c + pd.Timedelta(days=C.PHASE_WINDOW_DAYS))].copy()
        if not len(win):
            continue
        win["x"] = (win["date"] - c).dt.days
        pos = win["Positive"] / win["total"]
        neu = win["Neutral"] / win["total"]
        neg = win["Negative"] / win["total"]

        ax.bar(win["x"], pos, color=C.COLOR_POS, label="Positive", width=0.9)
        ax.bar(win["x"], neu, bottom=pos, color=C.COLOR_NEU, label="Neutral", width=0.9)
        ax.bar(win["x"], neg, bottom=pos + neu, color=C.COLOR_NEG, label="Negative", width=0.9)

        neg_ma3 = neg.rolling(3, center=True).mean()
        ax.plot(win["x"], neg_ma3, color="black", lw=2, label="Negative 3-day MA")
        ax.axhline(base_neg + C.TAU, color="red", ls="--", lw=1.2,
                   label=f"Baseline+{C.TAU:.2f} ({base_neg + C.TAU:.2f})")
        ax.axhline(base_neg, color="gray", ls=":", lw=1, label=f"Baseline ({base_neg:.2f})")
        ax.axvline(0, color="#FF8F00", ls="--", lw=1.2)
        ax.set_title(f"{ph['name']}  (center {ph['center']})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Days relative to event center")
        if i == 0:
            ax.set_ylabel("Daily sentiment proportion")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, 1.02)

    fig.suptitle("Event Lifecycle: Sentiment Composition Around Tariff Events (±15 days)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "Fig4_event_lifecycle.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig4_event_lifecycle.png", dpi=C.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 4 -> {OUT}")


def event_prepost_comparison(daily, pre_days=7, post_days=14):
    # 对每个事件比较中心日之前 pre_days 天和之后 post_days 天的情感变化
    rows = []
    for ph in C.PHASES:
        c = pd.Timestamp(ph["center"])
        pre = daily[(daily["date"] >= c - pd.Timedelta(days=pre_days)) & (daily["date"] < c)]
        post = daily[(daily["date"] >= c) & (daily["date"] <= c + pd.Timedelta(days=post_days))]

        def stat(w):
            n = w["total"].sum()
            return {"n": int(n),
                    "mean_sent": float((w["mean_sent"] * w["total"]).sum() / n) if n else np.nan,
                    "pos_share": float(w["Positive"].sum() / n) if n else np.nan,
                    "neg_share": float(w["Negative"].sum() / n) if n else np.nan}
        a, b = stat(pre), stat(post)
        rows.append({"event": ph["name"], "center": ph["center"],
                     "pre_n": a["n"], "post_n": b["n"],
                     "pre_mean": round(a["mean_sent"], 4), "post_mean": round(b["mean_sent"], 4),
                     "mean_change": round(b["mean_sent"] - a["mean_sent"], 4),
                     "pre_neg_share": round(a["neg_share"], 4),
                     "post_neg_share": round(b["neg_share"], 4),
                     "neg_share_change": round(b["neg_share"] - a["neg_share"], 4),
                     "pre_pos_share": round(a["pos_share"], 4),
                     "post_pos_share": round(b["pos_share"], 4),
                     "pos_share_change": round(b["pos_share"] - a["pos_share"], 4)})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "table_event_prepost_comparison.csv", index=False)
    print(f"\n[事件前后对比 pre {pre_days}d / post {post_days}d]")
    print(tab.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(tab)); w = 0.35
    axes[0].bar(x - w/2, tab["pre_mean"], w, label="Pre-event", color="#2196F3")
    axes[0].bar(x + w/2, tab["post_mean"], w, label="Post-event", color="#F44336")
    axes[0].axhline(0.5, color="gray", ls="--", alpha=0.5)
    axes[0].set_title("Mean sentiment"); axes[0].set_ylim(0, 1)
    axes[1].bar(x - w/2, tab["pre_neg_share"], w, label="Pre-event", color="#2196F3")
    axes[1].bar(x + w/2, tab["post_neg_share"], w, label="Post-event", color="#F44336")
    axes[1].set_title("Negative share"); axes[1].set_ylim(0, 1)
    axes[2].bar(x - w/2, tab["pre_pos_share"], w, label="Pre-event", color="#2196F3")
    axes[2].bar(x + w/2, tab["post_pos_share"], w, label="Post-event", color="#F44336")
    axes[2].set_title("Positive share"); axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(tab["event"], rotation=12); ax.legend()
    fig.suptitle(f"Sentiment Change: {pre_days} Days Before vs {post_days} Days After Each Event",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "Fig_event_prepost_comparison.png", dpi=C.FIG_DPI)
    plt.close(fig)
    print("[Fig] Fig_event_prepost_comparison.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("聚合日度统计 (分块读取)...")
    daily = load_daily()
    daily.drop(columns=["hourly"]).to_csv(OUT / "daily_stats.csv", index=False)
    print(f"日度数据: {len(daily)} 天  {daily['date'].min().date()} ~ {daily['date'].max().date()}")

    print("\n[阶段检测]")
    detect_phases(daily)

    print("\n[Fig 3]")
    fig3_overall(daily)
    print("\n[Fig 4]")
    fig4_lifecycle(daily)
    event_prepost_comparison(daily)
    print("\n全部输出 ->", OUT)


if __name__ == "__main__":
    main()
