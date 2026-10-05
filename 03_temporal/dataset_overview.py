# -*- coding: utf-8 -*-
# Fig 2: 统计每个检索关键词爬到了多少条微博、多少个不同用户, 画成两列散点图。
# 按帖子数从多到少排, 和论文里那张 weibo_stats 点图对应。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import config as C

OUT = C.OUTPUT_DIR / "temporal"

KW_EN = {
    "关税": "Tariff", "国际合作": "Intl Cooperation", "特朗普": "Trump", "芯片": "Chips",
    "股市波动": "Stock Volatility", "跨境电商": "Cross-border E-comm",
    "投资者信心": "Investor Confidence", "半导体": "Semiconductor", "大豆": "Soybean",
    "A股": "A-Shares", "民意": "Public Opinion", "贸易壁垒": "Trade Barriers",
    "政策反馈": "Policy Feedback", "协议": "Agreement", "政治舆论": "Political Opinion",
    "汽车": "Automobile", "美国": "USA", "全球经济": "Global Economy", "出口": "Export",
    "合作": "Cooperation", "电动车": "EVs", "美元": "USD", "贸易逆差": "Trade Deficit",
    "经济": "Economy", "壁垒": "Barriers", "稀土": "Rare Earth",
    "国际资本流动": "Intl Capital Flow", "光伏": "Photovoltaic", "供应链": "Supply Chain",
    "贸易战": "Trade War", "黄金": "Gold", "拜登": "Biden",
    "产业转移": "Industrial Transfer", "钢铁": "Steel", "关税政策": "Tariff Policy",
    "301调查": "301 Investigation", "全球股市": "Global Stocks", "抗议": "Protest",
    "外贸": "Foreign Trade", "进口": "Import", "中国制造": "Made in China", "欧盟": "EU",
    "保护主义": "Protectionism", "反制": "Countermeasures",
    "中美达成协议": "US-China Agreement", "汇率": "Exchange Rate",
    "通货膨胀": "Inflation", "进口替代": "Import Substitution", "GDP": "GDP",
    "豁免": "Exemption", "贸易政策": "Trade Policy", "新能源": "New Energy",
    "反倾销": "Anti-dumping", "全球贸易战": "Global Trade War", "中美贸易": "US-China Trade",
    "双边关系": "Bilateral Relations", "多边贸易": "Multilateral Trade", "WTO": "WTO",
    "RCEP": "RCEP", "国产替代": "Domestic Substitution", "资本外流": "Capital Outflow",
    "反制措施": "Countermeasures", "汇率波动": "FX Volatility",
    "自由贸易区": "Free Trade Zone", "经济制裁": "Economic Sanctions",
    "对华关税": "Tariffs on China",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.SENTIMENT_OUTPUT,
                     usecols=[C.COL_KEYWORD, C.COL_USER], dtype=str)
    stat = (df.groupby(C.COL_KEYWORD)
              .agg(tweets=(C.COL_USER, "size"),
                   users=(C.COL_USER, "nunique"))
              .sort_values("tweets", ascending=False))
    stat.to_csv(OUT / "table_keyword_stats.csv", encoding="utf-8-sig")

    labels = [f"{kw} ({KW_EN.get(kw, kw)})" for kw in stat.index]
    y = range(len(stat))[::-1]
    plt = C.setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(11, max(8, 0.30 * len(stat))), sharey=True)
    axes[0].scatter(stat["tweets"], list(y), s=90, color="#8E44AD", zorder=3)
    axes[1].scatter(stat["users"], list(y), s=90, color="#1ABC9C", zorder=3)
    for ax, title in zip(axes, ["Tweets", "Users"]):
        ax.set_title(title, fontsize=13)
        ax.grid(axis="x", ls="--", alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(labels, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "Fig2_keyword_stats_dotplot.png", dpi=C.FIG_DPI,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 2] 关键词数={len(stat)}, 总帖={stat['tweets'].sum():,}, "
          f"去重用户={df[C.COL_USER].nunique():,}")
    print(f"输出 -> {OUT / 'Fig2_keyword_stats_dotplot.png'}")


if __name__ == "__main__":
    main()
