#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 把转发关系整理成边表和节点表, 再顺着 retweet_id 往回追每条帖子属于哪条转发链、深度多少。
# 这份数据没有 user_id, 一律拿用户昵称当作用户。
# 有些帖子转发的是数据集外面的老微博, 那种源头就记成外部根 (is_external_root=1)。
# 输出三个 csv: 边表、节点表、每条转发帖的链路摘要
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

OUT = C.OUTPUT_DIR / "network"


def norm_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return None
    # id 很长, pandas 读进来可能变成带 .0 的科学计数法字符串, 这里去掉
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def load_data(chunksize=100000):
    id_set = set()
    posts_attrs = {}
    edges_raw = []

    for chunk in tqdm(pd.read_csv(C.SENTIMENT_OUTPUT, chunksize=chunksize, low_memory=False),
                      desc="Loading"):
        ids = chunk[C.COL_ID].map(norm_id)
        rts = chunk[C.COL_RETWEET].map(norm_id)
        users = chunk[C.COL_USER].fillna("").astype(str)
        tiers = chunk[C.COL_AUTH].fillna("普通用户").map(C.USER_TIER_MAP).fillna("Regular")
        times = chunk[C.COL_TIME]
        ips = chunk[C.COL_IP]
        sents = pd.to_numeric(chunk.get(C.COL_SENT), errors="coerce")
        for i in range(len(chunk)):
            pid = ids.iloc[i]
            if pid is None:
                continue
            if pid not in id_set:
                id_set.add(pid)
                posts_attrs[pid] = {
                    "post_id": pid,
                    "user": users.iloc[i],
                    "time": times.iloc[i],
                    "ip": ips.iloc[i],
                    "tier": tiers.iloc[i],
                    "sentiment": None if sents is None else float(sents.iloc[i])
                    if pd.notna(sents.iloc[i]) else np.nan,
                }
            rt = rts.iloc[i]
            if rt is not None:
                edges_raw.append({
                    "src_post_id": rt,
                    "dst_post_id": pid,
                    "dst_user": users.iloc[i],
                    "time": times.iloc[i],
                    "ip": ips.iloc[i],
                    "tier": tiers.iloc[i],
                    "sentiment": float(sents.iloc[i]) if sents is not None and pd.notna(sents.iloc[i]) else np.nan,
                })
    return id_set, posts_attrs, edges_raw


def build(id_set, posts_attrs, edges_raw):
    # 只留下源帖也在数据集里的边, 指向外部的源帖单独补成外部节点
    edges = [e for e in edges_raw if e["src_post_id"] in id_set]
    edges_df = pd.DataFrame(edges)
    nodes_df = pd.DataFrame(list(posts_attrs.values()))
    nodes_df["is_external_root"] = 0

    external = {e["src_post_id"] for e in edges_raw if e["src_post_id"] not in id_set}
    if external:
        # 排个序, 不然每次 set 的遍历顺序不一样, 输出文件顺序也跟着变
        ext = pd.DataFrame({"post_id": sorted(external, key=str), "user": "", "time": pd.NA,
                            "ip": pd.NA, "tier": "External", "sentiment": np.nan,
                            "is_external_root": 1})
        nodes_df = pd.concat([nodes_df, ext], ignore_index=True)
    return edges_df, nodes_df, external


def trace_chains(id_set, edges_raw, max_depth=30):
    # 每个转发帖往上游一层层找, 找到原创帖或者数据集外的帖子就是根
    id_to_rt = {}
    for e in edges_raw:
        id_to_rt.setdefault(e["dst_post_id"], e["src_post_id"])
    records = []
    for pid in tqdm(id_to_rt, desc="Tracing"):
        path, visited, cur, d = [pid], {pid}, pid, 0
        has_cycle, root = False, None
        while d < max_depth:
            rt = id_to_rt.get(cur)
            if rt is None:
                root = cur
                break
            if rt not in id_set:
                root = rt; d += 1; path.append(rt)
                break
            if rt in visited:
                has_cycle, root = True, rt
                path.append(rt); d += 1
                break
            path.append(rt); visited.add(rt); d += 1; cur = rt
        records.append({"post_id": pid, "root_post_id": root, "depth": d,
                        "has_cycle": int(has_cycle), "path": "->".join(path[:max_depth])})
    return pd.DataFrame(records)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    id_set, posts_attrs, edges_raw = load_data()
    edges_df, nodes_df, external = build(id_set, posts_attrs, edges_raw)
    chain_df = trace_chains(id_set, edges_raw)

    edges_df.to_csv(OUT / "retweet_edges.csv", index=False, encoding="utf-8-sig")
    nodes_df.to_csv(OUT / "retweet_nodes.csv", index=False, encoding="utf-8-sig")
    chain_df.to_csv(OUT / "retweet_chain_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60 + "\nRetweet Chain Summary\n" + "=" * 60)
    print(f"帖子节点: {len(nodes_df):,}  有效边(集内): {len(edges_df):,}  外部根: {len(external):,}")
    if len(chain_df):
        print(f"深度  max={chain_df['depth'].max()}  "
              f"P50={np.percentile(chain_df['depth'],50):.1f}  "
              f"P90={np.percentile(chain_df['depth'],90):.1f}  "
              f"环: {chain_df['has_cycle'].sum()}")
    print(f"输出 -> {OUT}")


if __name__ == "__main__":
    main()
