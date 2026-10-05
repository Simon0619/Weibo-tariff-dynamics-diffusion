#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 转发网络 / 级联分析 (表10, Fig 18-21)
# 这份数据没有 user_id, 就拿用户昵称当用户; retweet_id 指向它转发的那条帖子。
# 级联就是一棵转发树: 最源头的原创帖是根, 顺着 retweet_id 一层层往下;
# 如果源头帖子不在数据集里, 就当一个虚拟的外部根, 转发它的人照样算进这棵树。
# Table 10 五种认证类型用户的级联数、成员数、深度 (分 size>2 和 >5 两档)
# Fig 18 理性 (中性根) 和感性 (负向根) 级联的树状对比 + 三个结构指标箱线图
# Fig 19 被转发最多的 Top10 用户条形图
# Fig 20 五种用户两档阈值的级联数/平均规模/中位规模三联柱
# Fig 21 三种情感转发深度的 CCDF 双对数图
# 直接 python cascade_analysis.py 跑
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

C.setup_plot()
OUT = C.OUTPUT_DIR / "network"

TIER_ZH2EN = C.USER_TIER_MAP
TIER_ORDER = C.TIER_ORDER


def norm_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return None
    # id 列可能被 pandas 读成浮点 (如 5.178e15), 去 .0
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def load_graph():
    """一遍扫描构建: posts 属性 + parent 映射。
    attrs[pid] = (user, tier, sent, cat, text_chain_depth)
    text_chain_depth: 微博嵌套转发由正文 '//@用户:' 标记串联,
    出现次数即该帖所在转发链的文本深度 (retweet_id 仅指向原帖, 图深度恒为1)。"""
    attrs = {}
    parent = {}
    # 每个人各条帖子上出现过的认证类型计数 (空值不计), 用来在用户层面定他的身份
    user_votes = defaultdict(lambda: np.zeros(len(TIER_ORDER), dtype=int))
    n_rows = 0
    text_col = "text_raw"
    reader = pd.read_csv(C.SENTIMENT_OUTPUT,
                         usecols=[C.COL_ID, C.COL_USER, C.COL_RETWEET, C.COL_AUTH,
                                  C.COL_SENT, C.COL_SENT_CAT, text_col],
                         chunksize=100000, low_memory=False)
    for chunk in tqdm(reader, desc="读取/建图"):
        ids = chunk[C.COL_ID].map(norm_id)
        rts = chunk[C.COL_RETWEET].map(norm_id)
        users = chunk[C.COL_USER].fillna("").astype(str)
        raw_tiers = chunk[C.COL_AUTH]
        tiers = raw_tiers.fillna("普通用户").map(TIER_ZH2EN).fillna("Regular")
        sents = pd.to_numeric(chunk[C.COL_SENT], errors="coerce").fillna(0.5)
        cats = chunk[C.COL_SENT_CAT].fillna("Neutral")
        tdep = chunk[text_col].fillna("").astype(str).str.count("//@").clip(upper=20)
        for pid, rt, u, t, raw_t, s, c, d in zip(ids, rts, users, tiers, raw_tiers,
                                                 sents, cats, tdep):
            n_rows += 1
            if pid is None:
                continue
            if pid not in attrs:
                attrs[pid] = (u, t, float(s), c, int(d))
            if u and isinstance(raw_t, str) and raw_t.strip():
                ti = TIER_ORDER.index(t) if t in TIER_ORDER else TIER_ORDER.index("Regular")
                user_votes[u][ti] += 1
            if rt is not None:
                parent[pid] = rt
    print(f"总行数 {n_rows:,}, 唯一帖子 {len(attrs):,}, 转发边 {len(parent):,}")
    return attrs, parent, user_votes


def find_roots(attrs, parent):
    """根追溯 + 路径压缩。返回 root_of[pid], 含外部根"""
    root_of = {}

    def root_of_node(x):
        path = []
        cur = x
        while True:
            if cur in root_of:
                r = root_of[cur]
                break
            p = parent.get(cur)
            if p is None:
                r = cur  # 原创帖(数据集内)
                break
            if p not in attrs and p not in parent:
                r = p  # 外部根
                break
            path.append(cur)
            cur = p
        for node in path:
            root_of[node] = r
        root_of[x] = r
        return r

    for pid in tqdm(list(attrs), desc="根追溯"):
        root_of_node(pid)
    return root_of


def decide_user_tiers(user_votes):
    # 认证类型很多帖子是空的, 同一个人各条帖子标的还可能不一样。
    # 规则: 只数非空的认证记录, 某种 V 比 "普通用户" 出现得多才算该 V,
    # 否则一律算普通用户 (避免一条偶然的 V 标记就把人升级)
    best = {}
    reg_i = TIER_ORDER.index("Regular")
    for u, v in user_votes.items():
        v_idx = [i for i, t in enumerate(TIER_ORDER) if t != "Regular" and v[i] > 0]
        if v_idx and max(v[i] for i in v_idx) > v[reg_i]:
            best[u] = TIER_ORDER[max(v_idx, key=lambda i: v[i])]
        else:
            best[u] = "Regular"
    return best


def compute_cascades(attrs, parent, root_of, user_tier):
    """级联统计: size/depth/breadth/根层级/根情感"""
    members = defaultdict(list)
    for pid, r in root_of.items():
        members[r].append(pid)

    # 深度: 沿 parent 到根的跳数 (迭代 + 记忆化)
    depth_cache = {}

    def depth(x):
        if x in depth_cache:
            return depth_cache[x]
        path, cur, onpath = [], x, set()
        while True:
            if cur in depth_cache:
                d = depth_cache[cur]
                for node in reversed(path):
                    d += 1
                    depth_cache[node] = d
                return depth_cache[x]
            p = parent.get(cur)
            if p is None or p not in root_of or cur in onpath:
                d = 0
                for node in reversed(path):
                    d += 1
                    depth_cache[node] = d
                depth_cache.setdefault(cur, 0)
                return depth_cache[x]
            onpath.add(cur)
            path.append(cur)
            cur = p

    rows = []
    for root, mem in tqdm(members.items(), desc="级联指标"):
        a = attrs.get(root)
        # 根用户的认证类型也按用户层面的最佳类型来, 避免根帖刚好没标认证
        tier = user_tier.get(a[0], a[1]) if a else "External"
        cat = a[3] if a else (attrs[mem[0]][3] if mem else "Neutral")
        levels = defaultdict(int)
        maxd = 0
        for m in mem:
            # 图深度 (retweet_id 跳数) 与正文嵌套链 '//@' 深度取较大值
            td = attrs[m][4] if m in attrs else 0
            d = max(depth(m), td)
            levels[d] += 1
            if d > maxd:
                maxd = d
        breadth = max(levels.values())
        rows.append({"root": root, "tier": tier, "root_cat": cat,
                     "size": len(mem), "depth": maxd, "breadth": breadth})
    return pd.DataFrame(rows), depth_cache


# 图里五级用户的写法 (表格保留 GoldV 短名, 画图用带空格的)
TIER_DISPLAY = {"GoldV": "Gold V", "RedV": "Red V", "YellowV": "Yellow V",
                "BlueV": "Blue V", "Regular": "Regular"}
# KOL 图里用户身份的说明文字
TIER_KOL_LABEL = {"GoldV": "Gold V: Big V", "RedV": "Red V: Verified",
                  "YellowV": "Yellow V: Private", "BlueV": "Blue V: Official",
                  "Regular": "Regular"}


def table10(cas):
    # 论文里 cascade breadth 就是级联总节点数 |V|, 即这里的 size
    out = []
    for thr_name, min_size in [("Basic (>2)", C.CASCADE_MIN_SIZE_LOOSE),
                               ("Impact (>5)", C.CASCADE_MIN_SIZE_STRICT)]:
        sub = cas[cas["size"] > min_size]
        for tier in TIER_ORDER:
            g = sub[sub["tier"] == tier]
            out.append({
                "threshold": thr_name, "tier": tier, "count": len(g),
                "mean_breadth": round(g["size"].mean(), 2) if len(g) else np.nan,
                "median_breadth": float(g["size"].median()) if len(g) else np.nan,
                "max_breadth": int(g["size"].max()) if len(g) else 0,
                "mean_depth": round(g["depth"].mean(), 2) if len(g) else np.nan,
                "max_depth": int(g["depth"].max()) if len(g) else 0,
            })
    t10 = pd.DataFrame(out)
    t10.to_csv(OUT / "table10_cascade_metrics.csv", index=False)
    print("\n[Table 10]")
    print(t10.to_string(index=False))
    return t10


def fig20_thresholds(cas):
    """Fig 20: 三联柱 = Cascade Count / Mean Size / Median Size (对齐论文原图)。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    sub2 = cas[cas["size"] > C.CASCADE_MIN_SIZE_LOOSE]
    sub5 = cas[cas["size"] > C.CASCADE_MIN_SIZE_STRICT]
    x = np.arange(len(TIER_ORDER))
    panels = [("count", "Cascade Count"),
              ("mean_size", "Mean Size"),
              ("median_size", "Median Size")]
    for ax, (m, title) in zip(axes, panels):
        if m == "count":
            v2 = [len(sub2[sub2["tier"] == t]) for t in TIER_ORDER]
            v5 = [len(sub5[sub5["tier"] == t]) for t in TIER_ORDER]
        elif m == "mean_size":
            v2 = [sub2.loc[sub2["tier"] == t, "size"].mean() for t in TIER_ORDER]
            v5 = [sub5.loc[sub5["tier"] == t, "size"].mean() for t in TIER_ORDER]
        else:
            v2 = [sub2.loc[sub2["tier"] == t, "size"].median() for t in TIER_ORDER]
            v5 = [sub5.loc[sub5["tier"] == t, "size"].median() for t in TIER_ORDER]
        ax.bar(x - 0.2, v2, 0.4, color="#5B7CB2", label="size > 2")
        ax.bar(x + 0.2, v5, 0.4, color="#D98C5F", label="size > 5")
        ax.set_xticks(x)
        ax.set_xticklabels([TIER_DISPLAY[t] for t in TIER_ORDER], rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "Fig20_compare_thresholds_5types.png", dpi=C.FIG_DPI)
    plt.close(fig)


def fig19_kol(attrs, parent, user_tier):
    """Fig 19: 全部用户放在一起排前 10 (按收到的直接转发数), 金色横条 (对齐论文原图)。"""
    # 直接按用户名汇总收到的转发数, 不要按认证类型拆开 (同一个人会有多个标签)
    recv = defaultdict(int)
    for pid, rt in parent.items():
        a = attrs.get(rt)
        if a is not None and a[0]:
            recv[a[0]] += 1
    users = sorted(recv.items(), key=lambda z: z[1], reverse=True)
    top = users[:10][::-1]   # barh 从下往上画, 反转一下让第一名在最上面

    fig, ax = plt.subplots(figsize=(12, 9))
    labels = [f"{u}\n({TIER_KOL_LABEL[user_tier.get(u, 'Regular')]})" for u, _ in top]
    vals = [n for _, n in top]
    ax.barh(labels, vals, color="#D4AF37")
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, str(v), va="center", fontsize=11)
    ax.set_xlabel("Number of Retweets/Mentions (In-Degree)", fontsize=13)
    ax.set_title("Top 10 Most Reshared Users", fontsize=15)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT / "Fig19_KOL_ranking_barchart.png", dpi=C.FIG_DPI)
    plt.close(fig)
    pd.DataFrame([{"user": u, "tier": user_tier.get(u, "Regular"),
                   "retweets_received": n} for u, n in users]).to_csv(
        OUT / "user_retweet_received.csv", index=False)


def fig18_topology(attrs, parent, root_of, cas):
    """Fig 18: 上面两棵转发树示例 (理性广播 / 感性缠绕), 下面 WCC/LWCC/AvgDepth 箱线图。
    节点颜色按帖子自身情感: 负向红, 中性/正向蓝。"""
    children = defaultdict(list)
    for pid, rt in parent.items():
        children[rt].append(pid)

    def draw_tree(ax, root, title, cap, seed):
        mem = [m for m, r in root_of.items() if r == root and m in attrs]
        if root in attrs:
            mem = [root] + mem
        if len(mem) > cap:   # 太大就只留根 + 前 cap-1 个成员, 避免糊成一团
            mem = [root] + mem[1:cap]
        keep = set(mem)
        G = nx.DiGraph()
        G.add_nodes_from(keep)
        for m in keep:
            p = parent.get(m)
            if p in keep:
                G.add_edge(p, m)
        pos = nx.spring_layout(G, k=1.4 / np.sqrt(max(len(G), 1)), iterations=100, seed=seed)
        node_colors, node_sizes = [], []
        for n in G.nodes():
            cat = attrs[n][3] if n in attrs else "Neutral"
            node_colors.append(C.COLOR_NEG if cat == "Negative" else C.COLOR_POS)
            node_sizes.append(120 if n == root else 25)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, arrows=False, width=0.8)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                               node_color=node_colors, alpha=0.85,
                               edgecolors="white", linewidths=0.4)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    # 理性: 选一个中性根、规模适中(10~30)的级联; 感性: 负向根里最大的 (最多画 60 个点)
    neu = cas[(cas["root"].isin(attrs)) & (cas["root_cat"] == "Neutral")]
    neg = cas[(cas["root"].isin(attrs)) & (cas["root_cat"] == "Negative")].sort_values(
        "size", ascending=False)
    cand = neu[(neu["size"] >= 10) & (neu["size"] <= 30)]
    neu_root = (cand.iloc[0]["root"] if len(cand) else neu.sort_values(
        "size", ascending=False).iloc[0]["root"])
    neg_root = neg.iloc[0]["root"]

    # 箱线图口径: 以"根用户"为单位 (size>2 的级联), WCC=该用户级联总成员数,
    # LWCC=其中最大单棵级联大小, AvgDepth=这些级联的平均深度
    big = cas[(cas["size"] > C.CASCADE_MIN_SIZE_LOOSE) & cas["root"].isin(attrs)].copy()
    big["root_user"] = [attrs[r][0] for r in big["root"]]
    groups = {"Mainstream (Rational)": [], "Misleading (Emotional)": []}
    for (cat, u), gg in big.groupby(["root_cat", "root_user"]):
        if cat == "Neutral":
            key = "Mainstream (Rational)"
        elif cat == "Negative":
            key = "Misleading (Emotional)"
        else:
            continue
        groups[key].append((int(gg["size"].sum()), int(gg["size"].max()),
                            float(gg["depth"].mean())))

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0])
    ax0 = fig.add_subplot(gs[0, :2])
    ax1 = fig.add_subplot(gs[0, 2])
    draw_tree(ax0, neu_root, "Rational Broadcast (Tree Structure)", 30, seed=7)
    draw_tree(ax1, neg_root, "Emotional Entanglement (Tree Structure)", 60, seed=11)

    metric_names = [("WCC", 0), ("LWCC", 1), ("AvgDepth", 2)]
    for j, (mname, idx) in enumerate(metric_names):
        ax = fig.add_subplot(gs[1, j])
        data = [[row[idx] for row in groups[k]] for k in groups]
        bp = ax.boxplot(data, labels=list(groups), patch_artist=True, widths=0.55)
        for patch, color in zip(bp["boxes"], ["#3b97d3", "#e15748"]):
            patch.set_facecolor(color); patch.set_alpha(0.9)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_title(mname, fontsize=12, fontweight="bold")
        ax.set_ylabel(mname)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "Fig18_rational_emotional.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig18_rational_emotional.png", dpi=C.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def fig21_ccdf(attrs, parent, depth_cache, root_of):
    """转发帖深度 CCDF, 按帖子自身情感类别分组 (log-log 阶梯)。
    深度 = max(retweet_id 图跳数, 正文 '//@' 嵌套链深度)。"""
    # 所有帖子都计入 (深度 0 也算), 这样 x=1 处的纵值才是 P(深度>=1), 和论文原图一致
    by_cat = {"Negative": [], "Neutral": [], "Positive": []}
    for pid, a in attrs.items():
        gd = depth_cache.get(pid, 0)
        d = max(gd, a[4])  # 图深度 与 正文嵌套链深度
        by_cat[a[3]].append(d)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = {"Negative": C.COLOR_NEG, "Neutral": C.COLOR_NEU, "Positive": C.COLOR_POS}
    for cat in ["Negative", "Neutral", "Positive"]:
        vals = np.sort(np.array(by_cat[cat], dtype=float))
        if not len(vals):
            continue
        # 深度 0 的点不画出来 (不然左边缘会有一根竖线), 但分母还是全部帖子,
        # 所以第一段的高度正好就是 P(深度>=1), 和论文图一致
        xs = vals[vals >= 1]
        ccdf = 1 - np.searchsorted(vals, xs, side="left") / len(vals)
        ax.step(xs, ccdf, where="post", color=colors[cat], lw=2.0,
                label=f"{cat} (N={len(vals):,})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=1e-6, top=2)
    ax.set_xlabel("Text Depth (log scale)", fontsize=12)
    ax.set_ylabel("P(Depth >= x) (log scale)", fontsize=12)
    ax.set_title("CCDF of Cascade Depth by Sentiment", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "Fig21_RQ3_depth_ccdf.png", dpi=C.FIG_DPI)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    attrs, parent, user_votes = load_graph()
    root_of = find_roots(attrs, parent)
    user_tier = decide_user_tiers(user_votes)
    cas, depth_cache = compute_cascades(attrs, parent, root_of, user_tier)
    cas.to_csv(OUT / "cascades_all.csv", index=False)
    print(f"级联总数: {len(cas):,}; size>2: {(cas['size']>2).sum():,}; "
          f"size>5: {(cas['size']>5).sum():,}")

    table10(cas)
    print("\n[Fig 20]"); fig20_thresholds(cas)
    print("[Fig 19]"); fig19_kol(attrs, parent, user_tier)
    print("[Fig 18]"); fig18_topology(attrs, parent, root_of, cas)
    print("[Fig 21]"); fig21_ccdf(attrs, parent, depth_cache, root_of)
    print("\n全部输出 ->", OUT)


if __name__ == "__main__":
    main()
