#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 另一条情感打分路线: 直接用京东评论语料上微调过的 uer/roberta 二分类模型,
# 拿正向概率当情感分, 再按阈值切成三类。
# 如果已经用 DeepSeek 标的数据自己微调过三分类模型, 也可以用 --model 指定它。
# (最后跑全量数据用的是 infer_sentiment_skep.py, 这个文件留着复现实验路线)
# 输出还是 outputs/weibo_sentiment.csv, 分块读分块写, 不会把内存撑爆。
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def load_model(model_path, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval().to(device)
    n_labels = model.config.num_labels
    print(f"模型: {model_path}  num_labels={n_labels}  device={device}")
    return tok, model, n_labels


def infer_chunk(texts, tok, model, n_labels, device, max_len=128):
    """返回连续情感分 score ∈ [0,1]"""
    import torch
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    if n_labels == 2:      # [neg, pos] -> 正向概率
        return probs[:, 1]
    elif n_labels == 3:    # [neg, neu, pos] -> 期望分
        return probs[:, 2] * 1.0 + probs[:, 1] * 0.5
    raise ValueError(f"不支持的标签数: {n_labels}")


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
    ap.add_argument("--model", default=str(C.MODEL_DIR / "uer__roberta-base-finetuned-jd-binary-chinese"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=50000)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--device", default="cuda:1", help="GPU0 被占用, 默认用 cuda:1; 无 GPU 用 cpu")
    args = ap.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    tok, model, n_labels = load_model(args.model, device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reader = pd.read_csv(args.input, chunksize=args.chunk_size, low_memory=False)
    total, t0, first = 0, time.time(), True
    cat_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}

    for chunk in reader:
        texts = chunk[C.COL_TEXT].fillna("").astype(str).str.slice(0, 256).tolist()
        scores = []
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            if any(b.strip() for b in batch):
                scores.extend(infer_chunk(batch, tok, model, n_labels, device, args.max_len).tolist())
            else:
                scores.extend([0.5] * len(batch))
        chunk[C.COL_SENT] = np.round(scores, 4)
        chunk[C.COL_SENT_CAT] = chunk[C.COL_SENT].map(score_to_category)
        for k in cat_counts:
            cat_counts[k] += int((chunk[C.COL_SENT_CAT] == k).sum())

        chunk.to_csv(out_path, mode="w" if first else "a", header=first,
                     index=False, encoding="utf-8-sig")
        first = False
        total += len(chunk)
        speed = total / (time.time() - t0)
        print(f"  已处理 {total:,} 条  ({speed:.0f} 条/s)", flush=True)

    print(f"\n完成! 共 {total:,} 条 -> {out_path}")
    print(f"类别分布: {cat_counts}")
    print(f"耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
