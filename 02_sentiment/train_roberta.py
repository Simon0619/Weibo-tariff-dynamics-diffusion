#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 用上一步 DeepSeek 标的标签, 微调 chinese-roberta-wwm-ext 做三分类。
# 超参数就按论文里写的: 学习率 2e-5, batch 32, 3 个 epoch。
# 标签文件可以是 teacher_labels.csv, 也可以直接用之前那 500 条的 Chip 表。
# (当时服务器下不了 huggingface, 这个脚本没在这台机器上跑, 留着保证流程完整)
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}


def load_labels(path):
    df = pd.read_csv(path, low_memory=False)
    # 自己跑的 teacher 脚本里列名叫 teacher_label, 原来那份 Chip 表里叫 qwen_sentiment
    lab_col = "teacher_label" if "teacher_label" in df.columns else "qwen_sentiment"
    df = df.rename(columns={lab_col: "label"})
    df = df.dropna(subset=[C.COL_TEXT, "label"])
    df["label"] = df["label"].str.lower().map(LABEL2ID)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    print(f"标签数据: {len(df)} 条  分布: {df['label'].value_counts().to_dict()}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out", default=str(C.MODEL_DIR / "roberta_finetuned"))
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (AdamW, AutoModelForSequenceClassification,
                              AutoTokenizer, get_linear_schedule_with_warmup)

    device = args.device if torch.cuda.is_available() else "cpu"
    df = load_labels(args.labels)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3).to(device)

    class DS(Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            row = df.iloc[i]
            enc = tok(str(row[C.COL_TEXT])[:256], padding="max_length",
                      truncation=True, max_length=args.max_len, return_tensors="pt")
            return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(row["label"])

    loader = DataLoader(DS(), batch_size=args.batch_size, shuffle=True)
    opt = AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, 0, len(loader) * args.epochs)

    for ep in range(args.epochs):
        model.train()
        tot_loss, correct, n = 0.0, 0, 0
        for batch, labels in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            out = model(**batch, labels=labels)
            out.loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            tot_loss += out.loss.item() * len(labels)
            correct += (out.logits.argmax(-1) == labels).sum().item()
            n += len(labels)
        print(f"epoch {ep+1}/{args.epochs}  loss={tot_loss/n:.4f}  acc={correct/n:.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
