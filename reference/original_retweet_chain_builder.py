#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retweet_chain_builder.py
从 merge_all_data.csv 基于 retweet_id 构建转发链/网络数据，并导出用于网络分析与可视化的结果。

功能概述：
1) 分块读取大文件（chunksize，默认 200000），做必要的清洗与标准化：
   - 统一 id/retweet_id 字段为字符串，strip()，'nan'/'None'/'' 视为缺失
   - id 去重（保留首次出现）
2) 按参数 keep_only_valid_edges 控制边的保留规则：
   - True：只保留 retweet_id 能在数据集找到对应 id 的边
   - False：允许指向数据集外部的源微博（外部根）
3) 导出三类结果：
   A. 边表 retweet_edges.csv（src_post_id, dst_post_id, dst_user_id, time, 可选 ip/会员等级/会员类型/sentiment_score）
   B. 节点表 retweet_nodes.csv（post_id, user_id, time, ip, 会员等级, 会员类型, sentiment_score, is_external_root）
   C. 链路摘要 retweet_chain_summary.csv（post_id, root_post_id, depth, has_cycle, path）
4) 控制台输出链结构统计（总数、边数/节点数、根节点数、深度分布、Top20入度、环检测）

仅使用 pandas / numpy / tqdm，不依赖图数据库或 networkx。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------
# 配置与常量
# ---------------------------
REQUIRED_COLUMNS = {"id", "user_id", "retweet_id"}
OPTIONAL_COLUMNS = {
    "发布时间",
    "ip",
    "会员等级",
    "会员类型",
    "sentiment_score",
}


# ---------------------------
# 工具函数
# ---------------------------
def _normalize_str_id(x: Any) -> Optional[str]:
    """
    标准化 ID/retweet_id：
    - 转字符串并 strip
    - 'nan'/'None'/'' -> None
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return None
    return s


def _ensure_columns_exist(df_columns: List[str]) -> None:
    """
    检查最少需要的列是否存在；提示可选列缺失但不报错。
    """
    cols = set(df_columns)
    missing = REQUIRED_COLUMNS - cols
    if missing:
        raise ValueError(f"输入数据缺少必要列: {missing}")
    optional_missing = OPTIONAL_COLUMNS - cols
    if optional_missing:
        print(f"[WARN] 可选列缺失（不影响运行）: {optional_missing}")


# ---------------------------
# 数据加载（两遍扫描）
# ---------------------------
def load_data(
    input_path: Path,
    chunksize: int = 200000,
) -> Tuple[Set[str], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    分块读取 CSV，两遍逻辑（合并在一次循环内完成）：
    - 收集 id_set（首见去重），保存帖子属性 posts_attrs[id] = {...}
    - 直接生成原始边列表 edges_raw（后续再依据 keep_only_valid_edges 过滤）
    返回：
    - id_set：数据集中存在的微博 id（去重后）
    - posts_attrs：id -> 属性字典（仅首次出现）
    - edges_raw：所有存在 retweet_id 的边记录（未过滤）
    """
    id_set: Set[str] = set()
    posts_attrs: Dict[str, Dict[str, Any]] = {}
    edges_raw: List[Dict[str, Any]] = []

    # 先探测列头
    head = pd.read_csv(input_path, nrows=1)
    _ensure_columns_exist(list(head.columns))

    # 分块迭代
    for chunk in tqdm(
        pd.read_csv(input_path, chunksize=chunksize),
        desc="Loading & Cleaning",
    ):
        # 标准化 ID/retweet_id
        chunk["id"] = chunk["id"].apply(_normalize_str_id)
        chunk["retweet_id"] = chunk["retweet_id"].apply(_normalize_str_id)

        # 去除 id 缺失的行
        chunk = chunk[chunk["id"].notna()]

        # 跨块去重：保留首次出现的 id
        # 这里通过逐行扫描确保“首次”写入 posts_attrs
        for _, row in chunk.iterrows():
            pid = row["id"]
            if pid in id_set:
                # 已收录，跳过节点属性写入；但边仍可能需要记录
                pass
            else:
                id_set.add(pid)
                # 采集帖子属性（仅首次）
                attr = {
                    "post_id": pid,
                    "user_id": _normalize_str_id(row.get("user_id")),
                    "time": row.get("发布时间"),
                    "ip": row.get("ip"),
                    "membership_level": row.get("会员等级"),
                    "membership_type": row.get("会员类型"),
                    "sentiment_score": row.get("sentiment_score"),
                }
                posts_attrs[pid] = attr

            # 记录边（如 retweet_id 存在）
            src = row["retweet_id"]
            if src is not None:
                edge = {
                    "src_post_id": src,                # 源微博
                    "dst_post_id": pid,                # 转发微博
                    "dst_user_id": _normalize_str_id(row.get("user_id")),
                    "time": row.get("发布时间"),
                    "ip": row.get("ip"),
                    "membership_level": row.get("会员等级"),
                    "membership_type": row.get("会员类型"),
                    "sentiment_score": row.get("sentiment_score"),
                }
                edges_raw.append(edge)

    return id_set, posts_attrs, edges_raw


# ---------------------------
# 构建边与节点
# ---------------------------
def build_edges_nodes(
    id_set: Set[str],
    posts_attrs: Dict[str, Dict[str, Any]],
    edges_raw: List[Dict[str, Any]],
    keep_only_valid_edges: bool = True,
    include_external_roots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    """
    根据参数过滤边，并构造节点表。
    返回：
    - edges_df: 边表
    - nodes_df: 节点表
    - external_roots: 数据集中未出现但被指向的 retweet_id 集合
    """
    # 过滤边
    if keep_only_valid_edges:
        filtered_edges = [
            e for e in edges_raw if e["src_post_id"] in id_set
        ]
    else:
        filtered_edges = edges_raw

    edges_df = pd.DataFrame(filtered_edges)

    # 构造帖子节点（全部帖子）
    nodes_df = pd.DataFrame(list(posts_attrs.values()))
    nodes_df["is_external_root"] = 0

    # 外部根节点
    external_roots = set()
    if include_external_roots:
        # 外部根：edges 中出现但不在 id_set 的 src_post_id
        external_roots = {
            src for src in edges_df["src_post_id"].dropna().unique()
            if src not in id_set
        }
        if external_roots:
            ext_nodes = pd.DataFrame({
                "post_id": list(external_roots),
                "user_id": [None] * len(external_roots),
                "time": [None] * len(external_roots),
                "ip": [None] * len(external_roots),
                "membership_level": [None] * len(external_roots),
                "membership_type": [None] * len(external_roots),
                "sentiment_score": [None] * len(external_roots),
                "is_external_root": [1] * len(external_roots),
            })
            nodes_df = pd.concat([nodes_df, ext_nodes], ignore_index=True)

    # 列顺序整理
    edges_cols = [
        "src_post_id",
        "dst_post_id",
        "dst_user_id",
        "time",
        "ip",
        "membership_level",
        "membership_type",
        "sentiment_score",
    ]
    nodes_cols = [
        "post_id",
        "user_id",
        "time",
        "ip",
        "membership_level",
        "membership_type",
        "sentiment_score",
        "is_external_root",
    ]
    edges_df = edges_df.reindex(columns=edges_cols)
    nodes_df = nodes_df.reindex(columns=nodes_cols)
    return edges_df, nodes_df, external_roots


# ---------------------------
# 根追溯与深度计算
# ---------------------------
def find_roots_and_depths(
    id_set: Set[str],
    id_to_retweet: Dict[str, Optional[str]],
    max_depth: int = 30,
) -> pd.DataFrame:
    """
    对每个“转发微博”（retweet_id 非空）的帖子，追溯到最终根：
    - 一直沿 retweet_id 追溯，直到 retweet_id 为空（返回自身）或指向不存在的 id（作为外部根）
    - 检测环（cycle）：若链条中出现重复 id，则 has_cycle=True，并停止追溯
    - 返回 DataFrame: post_id, root_post_id, depth, has_cycle, path
    """
    records: List[Dict[str, Any]] = []

    # 只对有 retweet_id 的帖子做链路摘要
    candidate_posts = [pid for pid, rid in id_to_retweet.items() if rid is not None]

    for pid in tqdm(candidate_posts, desc="Tracing roots"):
        path: List[str] = [pid]
        visited: Set[str] = set([pid])
        cur = pid
        depth = 0
        has_cycle = False
        root_post_id: Optional[str] = None

        # 迭代追溯
        while depth < max_depth:
            rid = id_to_retweet.get(cur)

            if rid is None:
                # 当前帖子不是转发（或已到链末端）：根是当前帖子
                root_post_id = cur
                break

            # 指向外部根（数据集中不存在）
            if rid not in id_set:
                root_post_id = rid  # 记录为外部根 id（字符串）
                depth += 1
                path.append(rid)
                break

            # 正常继续追溯
            if rid in visited:
                # 检测到环
                has_cycle = True
                root_post_id = rid
                path.append(rid)
                depth += 1
                break

            path.append(rid)
            visited.add(rid)
            depth += 1
            cur = rid

        # 限制 path 长度
        if len(path) > max_depth:
            path = path[:max_depth]

        records.append({
            "post_id": pid,
            "root_post_id": root_post_id,
            "depth": depth,
            "has_cycle": int(has_cycle),
            "path": "->".join(path),
        })

    return pd.DataFrame(records)


# ---------------------------
# 导出与统计
# ---------------------------
def export_results(
    out_dir: Path,
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    chain_df: pd.DataFrame,
) -> None:
    """
    导出三类 CSV，并打印统计信息。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_path = out_dir / "retweet_edges.csv"
    nodes_path = out_dir / "retweet_nodes.csv"
    chain_path = out_dir / "retweet_chain_summary.csv"

    edges_df.to_csv(edges_path, index=False, encoding="utf-8-sig")
    nodes_df.to_csv(nodes_path, index=False, encoding="utf-8-sig")
    chain_df.to_csv(chain_path, index=False, encoding="utf-8-sig")

    # 统计信息
    total_posts = len(nodes_df["post_id"].unique())
    retweet_posts = len(chain_df["post_id"].unique())
    edge_count = len(edges_df)
    node_count = len(nodes_df)
    root_count = len(chain_df["root_post_id"].dropna().unique())
    max_depth = chain_df["depth"].max() if len(chain_df) else 0
    p50_depth = np.percentile(chain_df["depth"], 50) if len(chain_df) else 0
    p90_depth = np.percentile(chain_df["depth"], 90) if len(chain_df) else 0
    cycles = chain_df["has_cycle"].sum() if "has_cycle" in chain_df.columns else 0

    # Top20 入度（被转发次数）
    indegree = (
        edges_df.groupby("src_post_id")
        .size()
        .reset_index(name="indegree")
        .sort_values("indegree", ascending=False)
    )
    top20 = indegree.head(20)

    print("\n" + "=" * 70)
    print("Retweet Chain Summary & Network Stats")
    print("=" * 70)
    print(f"Total posts: {total_posts}")
    print(f"Retweet posts (retweet_id not null): {retweet_posts}")
    print(f"Edges: {edge_count}")
    print(f"Nodes: {node_count}")
    print(f"Root count: {root_count}")
    print(f"Depth max: {max_depth} | P50: {p50_depth:.2f} | P90: {p90_depth:.2f}")
    print(f"Cycles detected: {cycles}")
    print("\nTop 20 sources by indegree:")
    if len(top20):
        print(top20.to_string(index=False))
    else:
        print("(empty)")
    print("=" * 70 + "\n")

    print(f"✓ edges saved: {edges_path}")
    print(f"✓ nodes saved: {nodes_path}")
    print(f"✓ chain summary saved: {chain_path}")


def export_viz_html(
    out_dir: Path,
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    max_nodes: int = 3000,
    max_edges: int = 8000,
    seed: int = 42,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    edges = edges_df[["src_post_id", "dst_post_id"]].dropna().copy()
    edges.columns = ["source", "target"]

    nodes = nodes_df.copy()
    nodes = nodes.dropna(subset=["post_id"]).copy()
    nodes["post_id"] = nodes["post_id"].astype(str)

    if len(edges) == 0 or len(nodes) == 0:
        out_path = out_dir / "retweet_graph.html"
        out_path.write_text(
            "<!doctype html><meta charset='utf-8'><title>Retweet Graph</title><p>No graph data.</p>",
            encoding="utf-8",
        )
        return out_path

    degree = pd.concat(
        [
            edges.groupby("source").size().rename("outdegree"),
            edges.groupby("target").size().rename("indegree"),
        ],
        axis=1,
    ).fillna(0)
    degree["degree"] = degree["outdegree"] + degree["indegree"]

    if len(nodes) > max_nodes:
        top_ids = degree.sort_values("degree", ascending=False).head(max_nodes).index.astype(str)
        nodes = nodes[nodes["post_id"].isin(set(top_ids))].copy()
        keep = set(nodes["post_id"].astype(str).tolist())
        edges = edges[edges["source"].astype(str).isin(keep) & edges["target"].astype(str).isin(keep)].copy()

    if len(edges) > max_edges:
        edges = edges.sample(n=max_edges, random_state=seed).copy()

    nodes = nodes.drop_duplicates(subset=["post_id"], keep="first").copy()

    node_records: List[Dict[str, Any]] = []
    for _, r in nodes.iterrows():
        node_records.append(
            {
                "id": str(r.get("post_id")),
                "user_id": None if pd.isna(r.get("user_id")) else str(r.get("user_id")),
                "time": None if pd.isna(r.get("time")) else str(r.get("time")),
                "ip": None if pd.isna(r.get("ip")) else str(r.get("ip")),
                "membership_level": None if pd.isna(r.get("membership_level")) else str(r.get("membership_level")),
                "membership_type": None if pd.isna(r.get("membership_type")) else str(r.get("membership_type")),
                "sentiment_score": None if pd.isna(r.get("sentiment_score")) else float(r.get("sentiment_score")),
                "is_external_root": int(r.get("is_external_root") or 0),
            }
        )

    link_records: List[Dict[str, Any]] = []
    for _, e in edges.iterrows():
        link_records.append({"source": str(e["source"]), "target": str(e["target"])})

    graph_data = {
        "nodes": node_records,
        "links": link_records,
        "meta": {
            "node_count": len(node_records),
            "edge_count": len(link_records),
            "max_nodes": max_nodes,
            "max_edges": max_edges,
            "sample_seed": seed,
        },
    }

    data_json = json.dumps(graph_data, ensure_ascii=False)
    out_path = out_dir / "retweet_graph.html"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Retweet Graph</title>
  <style>
    html, body {{ height: 100%; margin: 0; font-family: system-ui, -apple-system, "Segoe UI", Arial, sans-serif; }}
    #toolbar {{ position: fixed; top: 12px; left: 12px; right: 12px; display: flex; gap: 12px; align-items: center; z-index: 10; background: rgba(255,255,255,0.9); border: 1px solid #ddd; border-radius: 10px; padding: 10px 12px; }}
    #meta {{ color: #333; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    #search {{ flex: 1; min-width: 200px; padding: 8px 10px; border: 1px solid #ccc; border-radius: 8px; }}
    #hint {{ color: #666; font-size: 13px; white-space: nowrap; }}
    #canvas {{ position: fixed; inset: 0; }}
    .link {{ stroke: #9aa0a6; stroke-opacity: 0.65; }}
    .node {{ stroke: #fff; stroke-width: 1px; cursor: grab; }}
    .node:active {{ cursor: grabbing; }}
    #tooltip {{ position: fixed; pointer-events: none; background: rgba(0,0,0,0.82); color: #fff; padding: 8px 10px; border-radius: 8px; font-size: 12px; line-height: 1.35; max-width: 420px; display: none; }}
  </style>
</head>
<body>
  <div id="toolbar">
    <div id="meta"></div>
    <input id="search" placeholder="输入 post_id 定位节点，例如：123456" />
    <div id="hint">滚轮缩放，拖拽平移，拖拽节点</div>
  </div>
  <div id="canvas"></div>
  <div id="tooltip"></div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <script>
    const graph = {data_json};
    const metaEl = document.getElementById("meta");
    metaEl.textContent = `nodes: ${{graph.meta.node_count}} | edges: ${{graph.meta.edge_count}} | sampled(max_nodes=${{graph.meta.max_nodes}}, max_edges=${{graph.meta.max_edges}})`;

    const width = window.innerWidth;
    const height = window.innerHeight;
    const tooltip = document.getElementById("tooltip");

    const svg = d3.select("#canvas").append("svg")
      .attr("width", width)
      .attr("height", height);

    const g = svg.append("g");

    svg.call(d3.zoom().scaleExtent([0.05, 8]).on("zoom", (event) => {{
      g.attr("transform", event.transform);
    }}));

    svg.append("defs").append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 18)
      .attr("refY", 0)
      .attr("markerWidth", 7)
      .attr("markerHeight", 7)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#9aa0a6");

    const nodeById = new Map(graph.nodes.map(n => [n.id, n]));
    const nodes = graph.nodes.map(n => Object.assign({{}}, n));
    const links = graph.links.map(l => Object.assign({{}}, l));

    const indegree = new Map();
    const outdegree = new Map();
    for (const l of links) {{
      indegree.set(l.target, (indegree.get(l.target) || 0) + 1);
      outdegree.set(l.source, (outdegree.get(l.source) || 0) + 1);
    }}
    for (const n of nodes) {{
      const d = (indegree.get(n.id) || 0) + (outdegree.get(n.id) || 0);
      n._degree = d;
      n._r = 3 + Math.min(16, Math.sqrt(d + 1));
    }}

    const link = g.append("g")
      .attr("stroke-linecap", "round")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("class", "link")
      .attr("stroke-width", 1.0)
      .attr("marker-end", "url(#arrow)");

    const node = g.append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("class", "node")
      .attr("r", d => d._r)
      .attr("fill", d => d.is_external_root ? "#d93025" : "#1a73e8")
      .on("mousemove", (event, d) => {{
        const parts = [
          `post_id: ${{d.id}}`,
          d.user_id ? `user_id: ${{d.user_id}}` : null,
          d.time ? `time: ${{d.time}}` : null,
          d.ip ? `ip: ${{d.ip}}` : null,
          d.membership_level ? `会员等级: ${{d.membership_level}}` : null,
          d.membership_type ? `会员类型: ${{d.membership_type}}` : null,
          (d.sentiment_score === null || d.sentiment_score === undefined) ? null : `sentiment_score: ${{d.sentiment_score}}`,
          `degree: ${{d._degree}}`,
          d.is_external_root ? "external_root: 1" : "external_root: 0",
        ].filter(Boolean);
        tooltip.style.display = "block";
        tooltip.textContent = parts.join("\\n");
        tooltip.style.left = (event.clientX + 12) + "px";
        tooltip.style.top = (event.clientY + 12) + "px";
      }})
      .on("mouseleave", () => {{
        tooltip.style.display = "none";
      }});

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(24).strength(0.8))
      .force("charge", d3.forceManyBody().strength(-60))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide().radius(d => d._r + 1.5));

    simulation.on("tick", () => {{
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
    }});

    node.call(d3.drag()
      .on("start", (event, d) => {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }})
      .on("drag", (event, d) => {{
        d.fx = event.x;
        d.fy = event.y;
      }})
      .on("end", (event, d) => {{
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }}));

    const search = document.getElementById("search");
    function highlight(id) {{
      node.attr("stroke", d => d.id === id ? "#fbbc04" : "#fff")
          .attr("stroke-width", d => d.id === id ? 3 : 1);
    }}
    search.addEventListener("input", () => {{
      const id = search.value.trim();
      if (!id) {{
        highlight(null);
        return;
      }}
      if (nodeById.has(id)) {{
        highlight(id);
      }} else {{
        highlight(null);
      }}
    }});

    window.addEventListener("resize", () => {{
      svg.attr("width", window.innerWidth).attr("height", window.innerHeight);
      simulation.force("center", d3.forceCenter(window.innerWidth / 2, window.innerHeight / 2));
      simulation.alpha(0.3).restart();
    }});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------
# 论文级静态可视化
# ---------------------------
def export_publication_figures(
    out_dir: Path,
    edges_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    chain_df: pd.DataFrame,
    max_nodes: int = 2000,
    max_edges: int = 5000,
    seed: int = 42,
    dpi: int = 300,
    figsize_network: tuple = (10, 10),
    figsize_stats: tuple = (12, 4),
) -> List[Path]:
    """
    导出适合论文发表的高质量静态可视化图片。
    
    生成图片：
    1. retweet_network.png - 转发网络拓扑图
    2. retweet_depth_distribution.png - 转发链深度分布
    3. retweet_indegree_distribution.png - 入度分布（对数坐标）
    4. retweet_cascade_size.png - 级联规模分布
    5. retweet_temporal.png - 转发时间序列（如有时间数据）
    
    参数:
        out_dir: 输出目录
        edges_df: 边表
        nodes_df: 节点表  
        chain_df: 链路摘要表
        max_nodes: 网络图最大节点数
        max_edges: 网络图最大边数
        seed: 随机种子
        dpi: 图片分辨率
        figsize_network: 网络图尺寸
        figsize_stats: 统计图尺寸
    
    返回:
        List[Path]: 生成的图片路径列表
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # 设置论文级字体和样式
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'SimSun', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.unicode_minus': False,
        'figure.dpi': 100,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })
    
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    
    # =========================================================================
    # 图1：转发网络拓扑图 (Network Topology)
    # =========================================================================
    try:
        import networkx as nx
        
        print("Generating network topology figure...")
        
        edges = edges_df[["src_post_id", "dst_post_id"]].dropna().copy()
        edges.columns = ["source", "target"]
        
        if len(edges) > 0:
            # 构建 NetworkX 图
            G = nx.DiGraph()
            for _, row in edges.iterrows():
                G.add_edge(str(row["source"]), str(row["target"]))
            
            # 采样大图
            if G.number_of_nodes() > max_nodes:
                # 按度数选择重要节点
                degree_dict = dict(G.degree())
                top_nodes = sorted(degree_dict.keys(), key=lambda x: degree_dict[x], reverse=True)[:max_nodes]
                G = G.subgraph(top_nodes).copy()
            
            if G.number_of_edges() > max_edges:
                # 随机采样边
                np.random.seed(seed)
                sampled_edges = list(G.edges())
                np.random.shuffle(sampled_edges)
                sampled_edges = sampled_edges[:max_edges]
                G = G.edge_subgraph(sampled_edges).copy()
            
            # 计算节点属性
            in_degree = dict(G.in_degree())
            out_degree = dict(G.out_degree())
            
            # 识别根节点（入度为0或高出度）
            external_ids = set(nodes_df[nodes_df["is_external_root"] == 1]["post_id"].astype(str).tolist())
            
            # 节点大小基于度数
            node_sizes = [3 + np.sqrt(in_degree.get(n, 0) + out_degree.get(n, 0)) * 3 for n in G.nodes()]
            
            # 节点颜色配置（更明显区分）
            # External Root: 紫色 - 与其他颜色完全不同
            # Positive (≥0.6): 绿色
            # Neutral (0.4-0.6): 蓝灰色
            # Negative (≤0.4): 橙红色
            COLOR_EXTERNAL_ROOT = '#9C27B0'  # 紫色 (Purple)
            COLOR_POSITIVE = '#4CAF50'       # 绿色 (Green)
            COLOR_NEUTRAL = '#78909C'        # 蓝灰色 (Blue Grey)
            COLOR_NEGATIVE = '#FF5722'       # 橙红色 (Deep Orange)
            
            sentiment_map = dict(zip(
                nodes_df["post_id"].astype(str), 
                nodes_df["sentiment_score"].fillna(0.5)
            ))
            
            node_colors = []
            for n in G.nodes():
                if n in external_ids:
                    node_colors.append(COLOR_EXTERNAL_ROOT)  # 紫色：外部根
                else:
                    score = sentiment_map.get(n, 0.5)
                    if pd.isna(score):
                        score = 0.5
                    # 根据情感分值分配颜色
                    if score >= 0.6:
                        node_colors.append(COLOR_POSITIVE)   # 绿色：正面
                    elif score <= 0.4:
                        node_colors.append(COLOR_NEGATIVE)   # 橙红色：负面
                    else:
                        node_colors.append(COLOR_NEUTRAL)    # 蓝灰色：中性
            
            # 绘制网络
            fig, ax = plt.subplots(figsize=figsize_network)
            
            # 使用 spring layout
            pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), iterations=50, seed=seed)
            
            # 绘制边（半透明）
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color='#B0BEC5',
                alpha=0.3,
                arrows=True,
                arrowsize=8,
                arrowstyle='-|>',
                connectionstyle='arc3,rad=0.1',
                width=0.5,
            )
            
            # 绘制节点
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_size=node_sizes,
                node_color=node_colors,
                alpha=0.85,
                linewidths=0.5,
                edgecolors='white',
            )
            
            # 图例（更新颜色）
            legend_elements = [
                mpatches.Patch(color=COLOR_EXTERNAL_ROOT, label='External Root'),
                mpatches.Patch(color=COLOR_POSITIVE, label='Positive (≥0.6)'),
                mpatches.Patch(color=COLOR_NEUTRAL, label='Neutral (0.4-0.6)'),
                mpatches.Patch(color=COLOR_NEGATIVE, label='Negative (≤0.4)'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
            
            ax.set_title(f'Retweet Network Topology\n(N={G.number_of_nodes():,}, E={G.number_of_edges():,})', 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
            fig_path = out_dir / "retweet_network.png"
            fig.savefig(fig_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            saved_paths.append(fig_path)
            print(f"  ✓ Saved: {fig_path}")
            
            # 同时保存 PDF 用于论文
            pdf_path = out_dir / "retweet_network.pdf"
            fig2, ax2 = plt.subplots(figsize=figsize_network)
            nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='#B0BEC5', alpha=0.3, arrows=True, arrowsize=8, width=0.5)
            nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=node_sizes, node_color=node_colors, alpha=0.85, linewidths=0.5, edgecolors='white')
            ax2.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
            ax2.set_title(f'Retweet Network Topology (N={G.number_of_nodes():,}, E={G.number_of_edges():,})', fontsize=14, fontweight='bold')
            ax2.axis('off')
            fig2.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
            plt.close(fig2)
            saved_paths.append(pdf_path)
            print(f"  ✓ Saved: {pdf_path}")
    
    except ImportError:
        print("  [WARN] networkx not installed, skipping network topology figure")
        print("  Install with: pip install networkx")
    except Exception as e:
        print(f"  [WARN] Failed to generate network figure: {e}")
    
    # =========================================================================
    # 图2：转发链深度分布 (Cascade Depth Distribution)
    # =========================================================================
    if len(chain_df) > 0 and "depth" in chain_df.columns:
        print("Generating depth distribution figure...")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize_stats)
        
        # 左图：深度直方图
        ax1 = axes[0]
        depths = chain_df["depth"].dropna()
        max_depth = int(depths.max())
        
        ax1.hist(depths, bins=range(0, min(max_depth + 2, 25)), 
                color='#1976D2', alpha=0.8, edgecolor='white', linewidth=0.8)
        ax1.set_xlabel('Cascade Depth')
        ax1.set_ylabel('Frequency')
        ax1.set_title('(a) Depth Distribution', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加统计标注
        stats_text = f'Mean: {depths.mean():.2f}\nMedian: {depths.median():.0f}\nMax: {depths.max():.0f}'
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 右图：累积分布 (CDF)
        ax2 = axes[1]
        sorted_depths = np.sort(depths)
        cdf = np.arange(1, len(sorted_depths) + 1) / len(sorted_depths)
        ax2.plot(sorted_depths, cdf, color='#1976D2', linewidth=2)
        ax2.fill_between(sorted_depths, cdf, alpha=0.2, color='#1976D2')
        ax2.set_xlabel('Cascade Depth')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('(b) Cumulative Distribution (CDF)', fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        # 标注关键分位数
        for p, label in [(0.5, 'P50'), (0.9, 'P90'), (0.99, 'P99')]:
            idx = int(len(sorted_depths) * p)
            if idx < len(sorted_depths):
                val = sorted_depths[idx]
                ax2.axhline(p, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
                ax2.axvline(val, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
                ax2.annotate(f'{label}={val:.0f}', xy=(val, p), xytext=(val + 0.5, p - 0.05),
                            fontsize=8, color='#555')
        
        plt.tight_layout()
        fig_path = out_dir / "retweet_depth_distribution.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        saved_paths.append(fig_path)
        print(f"  ✓ Saved: {fig_path}")
    
    # =========================================================================
    # 图3：入度分布 (In-degree Distribution) - 对数-对数坐标
    # =========================================================================
    if len(edges_df) > 0:
        print("Generating in-degree distribution figure...")
        
        indegree = edges_df.groupby("src_post_id").size().reset_index(name="indegree")
        degree_counts = indegree["indegree"].value_counts().sort_index()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize_stats)
        
        # 左图：直方图
        ax1 = axes[0]
        ax1.hist(indegree["indegree"], bins=50, color='#388E3C', alpha=0.8, edgecolor='white')
        ax1.set_xlabel('In-degree (Number of Retweets)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('(a) In-degree Distribution', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 右图：对数-对数坐标（检验幂律分布）
        ax2 = axes[1]
        x = degree_counts.index.values
        y = degree_counts.values
        ax2.scatter(x, y, s=20, color='#388E3C', alpha=0.7, edgecolors='white', linewidth=0.5)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('In-degree (log scale)')
        ax2.set_ylabel('Frequency (log scale)')
        ax2.set_title('(b) Log-Log Plot', fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        
        # 拟合幂律线（可选）
        try:
            log_x = np.log10(x[x > 0])
            log_y = np.log10(y[x > 0])
            if len(log_x) > 2:
                coeffs = np.polyfit(log_x, log_y, 1)
                fit_line = 10 ** (coeffs[0] * log_x + coeffs[1])
                ax2.plot(10 ** log_x, fit_line, 'r--', linewidth=1.5, 
                        label=f'Power-law fit (γ≈{-coeffs[0]:.2f})')
                ax2.legend(loc='upper right', fontsize=9)
        except:
            pass
        
        plt.tight_layout()
        fig_path = out_dir / "retweet_indegree_distribution.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        saved_paths.append(fig_path)
        print(f"  ✓ Saved: {fig_path}")
    
    # =========================================================================
    # 图4：级联规模分布 (Cascade Size Distribution)
    # =========================================================================
    if len(chain_df) > 0 and "root_post_id" in chain_df.columns:
        print("Generating cascade size distribution figure...")
        
        # 计算每个根的级联规模
        cascade_sizes = chain_df.groupby("root_post_id").size().reset_index(name="cascade_size")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize_stats)
        
        # 左图：直方图
        ax1 = axes[0]
        ax1.hist(cascade_sizes["cascade_size"], bins=50, color='#7B1FA2', alpha=0.8, edgecolor='white')
        ax1.set_xlabel('Cascade Size')
        ax1.set_ylabel('Frequency')
        ax1.set_title('(a) Cascade Size Distribution', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加统计
        stats_text = f'Total Cascades: {len(cascade_sizes):,}\nMean Size: {cascade_sizes["cascade_size"].mean():.1f}\nMax Size: {cascade_sizes["cascade_size"].max():,}'
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 右图：对数-对数坐标
        ax2 = axes[1]
        size_counts = cascade_sizes["cascade_size"].value_counts().sort_index()
        x = size_counts.index.values
        y = size_counts.values
        ax2.scatter(x, y, s=20, color='#7B1FA2', alpha=0.7, edgecolors='white', linewidth=0.5)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Cascade Size (log scale)')
        ax2.set_ylabel('Frequency (log scale)')
        ax2.set_title('(b) Log-Log Plot', fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        fig_path = out_dir / "retweet_cascade_size.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        saved_paths.append(fig_path)
        print(f"  ✓ Saved: {fig_path}")
    
    # =========================================================================
    # 图5：时间序列分析 (Temporal Pattern)
    # =========================================================================
    if "time" in edges_df.columns:
        print("Generating temporal pattern figure...")
        
        try:
            edges_with_time = edges_df.copy()
            edges_with_time["datetime"] = pd.to_datetime(edges_with_time["time"], errors="coerce")
            edges_with_time = edges_with_time.dropna(subset=["datetime"])
            
            if len(edges_with_time) > 0:
                # 按日/小时聚合
                daily_counts = edges_with_time.set_index("datetime").resample("D").size()
                hourly_pattern = edges_with_time["datetime"].dt.hour.value_counts().sort_index()
                
                fig, axes = plt.subplots(1, 2, figsize=figsize_stats)
                
                # 左图：每日转发量
                ax1 = axes[0]
                ax1.plot(daily_counts.index, daily_counts.values, color='#0288D1', linewidth=1.2)
                ax1.fill_between(daily_counts.index, daily_counts.values, alpha=0.3, color='#0288D1')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Number of Retweets')
                ax1.set_title('(a) Daily Retweet Volume', fontweight='bold')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(axis='y', alpha=0.3)
                
                # 右图：小时分布
                ax2 = axes[1]
                ax2.bar(hourly_pattern.index, hourly_pattern.values, color='#0288D1', alpha=0.8, edgecolor='white')
                ax2.set_xlabel('Hour of Day')
                ax2.set_ylabel('Number of Retweets')
                ax2.set_title('(b) Hourly Distribution', fontweight='bold')
                ax2.set_xticks(range(0, 24, 2))
                ax2.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                fig_path = out_dir / "retweet_temporal.png"
                fig.savefig(fig_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                saved_paths.append(fig_path)
                print(f"  ✓ Saved: {fig_path}")
                
        except Exception as e:
            print(f"  [WARN] Failed to generate temporal figure: {e}")
    
    # =========================================================================
    # 图6：情感与网络结构关系 (Sentiment-Network Relationship)
    # =========================================================================
    if "sentiment_score" in nodes_df.columns:
        print("Generating sentiment-network relationship figure...")
        
        try:
            # 计算每个节点的度数
            in_deg = edges_df.groupby("dst_post_id").size().reset_index(name="in_degree")
            in_deg.columns = ["post_id", "in_degree"]
            out_deg = edges_df.groupby("src_post_id").size().reset_index(name="out_degree")
            out_deg.columns = ["post_id", "out_degree"]
            
            node_analysis = nodes_df[["post_id", "sentiment_score"]].copy()
            node_analysis["post_id"] = node_analysis["post_id"].astype(str)
            in_deg["post_id"] = in_deg["post_id"].astype(str)
            out_deg["post_id"] = out_deg["post_id"].astype(str)
            
            node_analysis = node_analysis.merge(in_deg, on="post_id", how="left")
            node_analysis = node_analysis.merge(out_deg, on="post_id", how="left")
            node_analysis = node_analysis.fillna(0)
            node_analysis["total_degree"] = node_analysis["in_degree"] + node_analysis["out_degree"]
            
            # 过滤有效数据
            valid_data = node_analysis[
                (node_analysis["sentiment_score"].notna()) & 
                (node_analysis["total_degree"] > 0)
            ]
            
            if len(valid_data) > 0:
                fig, axes = plt.subplots(1, 2, figsize=figsize_stats)
                
                # 左图：情感 vs 入度（散点图）
                ax1 = axes[0]
                scatter = ax1.scatter(
                    valid_data["in_degree"], 
                    valid_data["sentiment_score"],
                    c=valid_data["sentiment_score"],
                    cmap='RdYlGn',
                    alpha=0.5,
                    s=15,
                    edgecolors='white',
                    linewidth=0.3,
                )
                ax1.set_xlabel('In-degree')
                ax1.set_ylabel('Sentiment Score')
                ax1.set_title('(a) Sentiment vs In-degree', fontweight='bold')
                ax1.set_xscale('log')
                ax1.grid(alpha=0.3)
                plt.colorbar(scatter, ax=ax1, label='Sentiment')
                
                # 右图：按情感分组的度数分布
                ax2 = axes[1]
                bins = [0, 0.4, 0.6, 1.0]
                labels = ['Negative\n(≤0.4)', 'Neutral\n(0.4-0.6)', 'Positive\n(≥0.6)']
                valid_data['sentiment_group'] = pd.cut(valid_data['sentiment_score'], bins=bins, labels=labels)
                
                group_stats = valid_data.groupby('sentiment_group')['in_degree'].mean()
                colors = ['#F44336', '#FFEB3B', '#4CAF50']
                bars = ax2.bar(range(len(group_stats)), group_stats.values, color=colors, alpha=0.8, edgecolor='white')
                ax2.set_xticks(range(len(group_stats)))
                ax2.set_xticklabels(group_stats.index)
                ax2.set_ylabel('Mean In-degree')
                ax2.set_title('(b) Mean In-degree by Sentiment', fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                
                # 添加数值标签
                for bar, val in zip(bars, group_stats.values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                            f'{val:.1f}', ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                fig_path = out_dir / "retweet_sentiment_network.png"
                fig.savefig(fig_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                saved_paths.append(fig_path)
                print(f"  ✓ Saved: {fig_path}")
                
        except Exception as e:
            print(f"  [WARN] Failed to generate sentiment-network figure: {e}")
    
    print(f"\n✓ Publication figures saved to: {out_dir}")
    print(f"  Total figures: {len(saved_paths)}")
    
    return saved_paths


# ---------------------------
# 主流程
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build retweet chain/network from merge_all_data.csv"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="merge_all_data.csv",
        help="Input CSV file path (default: merge_all_data.csv)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./retweet_output",
        help="Output directory (default: ./retweet_output)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Chunk size for reading CSV (default: 200000)",
    )
    parser.add_argument(
        "--keep_only_valid_edges",
        action="store_true",
        help="Keep only edges whose retweet_id exists in dataset",
    )
    parser.add_argument(
        "--include_external_roots",
        action="store_true",
        help="Include external retweet_id as nodes",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=30,
        help="Max tracing depth for chain summary (default: 30)",
    )
    parser.add_argument(
        "--viz_html",
        action="store_true",
        help="Export interactive HTML visualization (retweet_graph.html)",
    )
    parser.add_argument(
        "--viz_max_nodes",
        type=int,
        default=3000,
        help="Max nodes in visualization (default: 3000)",
    )
    parser.add_argument(
        "--viz_max_edges",
        type=int,
        default=8000,
        help="Max edges in visualization (default: 8000)",
    )
    parser.add_argument(
        "--viz_publication",
        action="store_true",
        help="Export publication-quality static figures (PNG/PDF) for academic papers",
    )
    parser.add_argument(
        "--viz_dpi",
        type=int,
        default=300,
        help="DPI for publication figures (default: 300)",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    # 1) 加载数据（分块、清洗、去重、收集边与属性）
    id_set, posts_attrs, edges_raw = load_data(
        input_path=input_path,
        chunksize=args.chunksize,
    )

    # 构造 id -> retweet_id 映射（仅首次出现的帖子）
    # 为了兼容：若在首次出现时 retweet_id 缺失，后续出现有值，这里仍以首次为准（满足"保留首次出现"的语义）
    id_to_retweet: Dict[str, Optional[str]] = {}
    # 二次扫描（简化：从 posts_attrs 构造映射；edges_raw 中的 dst_post_id 若不在映射，可补充）
    for pid in id_set:
        # 初值 None；后续如 edges_raw 中出现该 dst_post_id 的 retweet 指向，可补充一次（保持首次）
        id_to_retweet[pid] = None
    for e in edges_raw:
        dst = e["dst_post_id"]
        src = e["src_post_id"]
        if dst in id_to_retweet and id_to_retweet[dst] is None:
            id_to_retweet[dst] = src

    # 2) 构建边与节点
    edges_df, nodes_df, external_roots = build_edges_nodes(
        id_set=id_set,
        posts_attrs=posts_attrs,
        edges_raw=edges_raw,
        keep_only_valid_edges=args.keep_only_valid_edges,
        include_external_roots=args.include_external_roots,
    )

    # 3) 根追溯与深度（仅针对转发帖子）
    chain_df = find_roots_and_depths(
        id_set=id_set,
        id_to_retweet=id_to_retweet,
        max_depth=args.max_depth,
    )

    # 4) 导出与统计
    export_results(
        out_dir=out_dir,
        edges_df=edges_df,
        nodes_df=nodes_df,
        chain_df=chain_df,
    )

    # 5) 交互式HTML可视化
    if args.viz_html:
        out_path = export_viz_html(
            out_dir=out_dir,
            edges_df=edges_df,
            nodes_df=nodes_df,
            max_nodes=args.viz_max_nodes,
            max_edges=args.viz_max_edges,
        )
        print(f"✓ Interactive HTML saved: {out_path}")

    # 6) 论文级静态可视化
    if args.viz_publication:
        print("\n" + "=" * 60)
        print("Generating publication-quality figures...")
        print("=" * 60)
        fig_paths = export_publication_figures(
            out_dir=out_dir,
            edges_df=edges_df,
            nodes_df=nodes_df,
            chain_df=chain_df,
            max_nodes=args.viz_max_nodes,
            max_edges=args.viz_max_edges,
            dpi=args.viz_dpi,
        )
        print("=" * 60)


if __name__ == "__main__":
    main()

