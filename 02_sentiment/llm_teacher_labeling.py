#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 找 DeepSeek 帮忙标注数据 (相当于拿大模型当老师, 后面再用这些标签微调小模型)。
# 按关键词分层抽样, 尽量让每个主题都能抽到。需要 API key,
# 不跑这个脚本的话也可以直接用之前已经标好的那 500 条 (Chip_deepseek-v3.2.csv)。
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

PROMPT_TEMPLATE = (
    "请判断以下微博文本的情感倾向, 只回答 positive、negative 或 neutral 之一, 不要输出其他内容。\n"
    "判断标准: 对中美贸易/关税议题表达支持、乐观、赞扬为 positive; "
    "表达反对、担忧、愤怒、悲观为 negative; 客观陈述或无明确倾向为 neutral。\n\n"
    "微博文本: {text}"
)


def label_one(client, text, model, max_retry=3):
    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=text[:500])}],
                temperature=0,
                max_tokens=8,
            )
            ans = resp.choices[0].message.content.strip().lower()
            for lab in ("positive", "negative", "neutral"):
                if lab in ans:
                    return lab
            return "neutral"
        except Exception as e:
            if attempt == max_retry - 1:
                print(f"  [失败] {e}")
                return None
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(C.DATA_FULL))
    ap.add_argument("--output", default=str(Path(__file__).parent / "teacher_labels.csv"))
    ap.add_argument("--n", type=int, default=2000, help="抽多少条去标")
    ap.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--model", default="deepseek-chat", help="DeepSeek-V3 对应的模型名")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("缺少 DeepSeek API key: 设置环境变量 DEEPSEEK_API_KEY 或者用 --api-key 传入")
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("请先 pip install openai")

    df = pd.read_csv(args.input, usecols=[C.COL_ID, C.COL_KEYWORD, C.COL_TEXT], low_memory=False)
    df = df.dropna(subset=[C.COL_TEXT])
    # 按关键词分层抽样, 每个目录都按比例抽一点
    frac = min(1.0, args.n * 1.2 / len(df))
    sample = (df.groupby(C.COL_KEYWORD, group_keys=False)
                .apply(lambda g: g.sample(frac=frac, random_state=args.seed))
                .sample(n=min(args.n, len(df)), random_state=args.seed)
                .reset_index(drop=True))
    print(f"抽样 {len(sample)} 条, 覆盖 {sample[C.COL_KEYWORD].nunique()} 个关键词")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    results = [None] * len(sample)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(label_one, client, t, args.model): i
                for i, t in enumerate(sample[C.COL_TEXT].tolist())}
        for j, fut in enumerate(as_completed(futs), 1):
            results[futs[fut]] = fut.result()
            if j % 100 == 0:
                print(f"  {j}/{len(sample)}  ({(time.time()-t0)/60:.1f} min)")

    sample["teacher_label"] = results
    sample = sample.dropna(subset=["teacher_label"])
    sample.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"完成: {len(sample)} 条 -> {args.output}")
    print(sample["teacher_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
