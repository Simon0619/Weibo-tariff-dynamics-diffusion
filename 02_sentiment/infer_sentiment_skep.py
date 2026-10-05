#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 用百度的 SKEP 大模型给全部帖子打情感分, 这是最后实际用的那个模型。
# 模型只输出 positive/negative 加一个置信度, 所以要换成 [0,1] 的连续分:
#   positive -> 分数就是置信度本身
#   negative -> 分数 = 1 - 置信度
#   没识别出来的 -> 0.5 算中性
# 然后再按 config 里的 0.4 / 0.6 阈值切成负/中/正三类。
# 跑得很慢 (几十万条要好几个小时), 所以加了断点续跑, 中断了再开会接着写。
# 运行: python infer_sentiment_skep.py (默认 gpu:1)
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def infer_batch(texts, task):
    """SKEP Taskflow 可能跳过空/超长文本导致返回数 != 输入数,
    用返回结果里的 text 字段对齐, 缺失记中性。"""
    from collections import defaultdict, deque
    try:
        batch_results = task(list(texts))
    except Exception:
        batch_results = []
        for t in texts:
            try:
                batch_results.append(task(t)[0])
            except Exception:
                batch_results.append({"text": t, "label": "neutral", "score": 0.5})
    buckets = defaultdict(deque)
    for r in batch_results:
        buckets[r.get("text", "")].append(
            (r.get("label", "neutral").lower(), float(r.get("score", 0.5))))
    out, n_miss = [], 0
    for t in texts:
        if buckets[t]:
            out.append(buckets[t].popleft())
        else:
            out.append(("neutral", 0.5)); n_miss += 1
    return out, n_miss


def to_sentiment_score(label, score):
    if label == "positive":
        return score
    if label == "negative":
        return 1.0 - score
    return 0.5


def score_to_category(score):
    if score >= C.POS_THRESHOLD:
        return "Positive"
    if score <= C.NEG_THRESHOLD:
        return "Negative"
    return "Neutral"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(C.DATA_FULL))
    ap.add_argument("--output", default=str(C.SENTIMENT_OUTPUT))
    ap.add_argument("--chunk-size", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="gpu:1")
    args = ap.parse_args()

    from paddlenlp.taskflow import Taskflow
    device_id = 0 if args.device == "cpu" else int(args.device.replace("gpu:", ""))
    print("加载 SKEP 模型...")
    task = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch",
                    device_id=device_id, batch_size=args.batch_size, max_seq_len=128)
    print("模型就绪")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑: 已写出的数据行数
    skip_rows = 0
    first = True
    cat_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    if out_path.exists():
        with open(out_path, "rb") as f:
            skip_rows = sum(1 for _ in f) - 1  # 减表头
        skip_rows = max(skip_rows, 0)
        if skip_rows > 0:
            print(f"断点续跑: 已存在 {skip_rows:,} 条结果, 将跳过这些输入行")
            # 统计已完成部分的类别分布
            done = pd.read_csv(out_path, usecols=[C.COL_SENT_CAT], low_memory=False)
            for k in cat_counts:
                cat_counts[k] = int((done[C.COL_SENT_CAT] == k).sum())
            first = False

    reader = pd.read_csv(args.input, chunksize=args.chunk_size, low_memory=False)
    total, t0 = skip_rows, time.time()
    consumed = 0

    for chunk in reader:
        if consumed + len(chunk) <= skip_rows:
            consumed += len(chunk)
            continue
        if consumed < skip_rows:
            chunk = chunk.iloc[skip_rows - consumed:].copy()
            consumed = skip_rows
        texts = chunk[C.COL_TEXT].fillna("").astype(str).str.slice(0, 256).tolist()
        labels_scores, misses = [], 0
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            res, m = infer_batch(batch, task)
            labels_scores.extend(res)
            misses += m
        if misses:
            print(f"  [提示] 本块 {misses} 条被模型跳过, 记为中性", flush=True)

        scores = [to_sentiment_score(lab, sc) for lab, sc in labels_scores]
        chunk[C.COL_SENT] = np.round(scores, 4)
        chunk[C.COL_SENT_CAT] = chunk[C.COL_SENT].map(score_to_category)
        for k in cat_counts:
            cat_counts[k] += int((chunk[C.COL_SENT_CAT] == k).sum())

        chunk.to_csv(out_path, mode="w" if first else "a", header=first,
                     index=False, encoding="utf-8-sig")
        first = False
        total += len(chunk)
        speed = total / (time.time() - t0)
        print(f"  已处理 {total:,} 条  ({speed:.0f} 条/s)  累计分布 {cat_counts}", flush=True)

    print(f"\n完成! 共 {total:,} 条 -> {out_path}")
    print(f"类别分布: {cat_counts}")
    print(f"耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
