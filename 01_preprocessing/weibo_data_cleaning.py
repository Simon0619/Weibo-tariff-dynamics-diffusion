#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 第二步清洗: 删广告机器人帖、按文本去重、只留含关键词的帖子,
# 再去掉 @ 和表情这些噪声, 最后用 jieba 分好词, 输出 weibo_cleaned_full.csv
# 运行: python weibo_data_cleaning.py --input 上一步的合并表.csv --output weibo_cleaned_full.csv
import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

AD_KEYWORDS = [
    "优惠", "促销", "红包", "代购", "推广", "私信", "领取", "关注公众号",
    "点下方链接", "长按识别二维码", "免费领取", "限时", "秒杀", "抢购",
    "加微信", "加V", "扫码", "点击链接", "复制链接",
]
URL_PATTERN = re.compile(r'http[s]?://\S+')
MENTION_PATTERN = re.compile(r'@\S+')
HASHTAG_PATTERN = re.compile(r'#(.*?)#')


def load_word_set(path):
    words = set()
    if path and Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            words = {line.strip() for line in f if line.strip()}
        print(f"加载词表 {path}: {len(words)} 个")
    else:
        print(f"[警告] 词表不存在: {path}")
    return words


def is_rubbish(text):
    """垃圾/机器人帖检测"""
    x = str(text)
    if len(URL_PATTERN.findall(x)) >= 2:
        return True
    if len(x.strip()) < 5:
        return True
    return any(kw in x for kw in AD_KEYWORDS)


def denoise_text(text):
    """去 @ / #话题#保留内容 / 去 emoji / 合并空格"""
    import emoji
    x = str(text)
    x = MENTION_PATTERN.sub("", x)
    x = HASHTAG_PATTERN.sub(r'\1', x)
    x = emoji.replace_emoji(x, replace="")
    return re.sub(r'\s+', ' ', x).strip()


def main():
    ap = argparse.ArgumentParser(description="微博文本完整清洗")
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--text-col', default='微博正文')
    ap.add_argument('--stopwords', default=None)
    ap.add_argument('--keywords', default=None, help='关键词文件; 不提供则跳过关键词过滤')
    args = ap.parse_args()

    import emoji  # noqa: F401
    import jieba

    # ---- 读取 ----
    df = None
    for enc in ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']:
        try:
            df = pd.read_csv(args.input, encoding=enc, dtype=str, low_memory=False)
            print(f"使用编码 {enc} 读取成功, {len(df)} 行")
            break
        except Exception:
            continue
    if df is None:
        raise SystemExit(f"无法读取 {args.input}")
    if args.text_col not in df.columns:
        raise SystemExit(f"找不到文本列 {args.text_col}, 现有列: {list(df.columns)}")

    df["text_raw"] = df[args.text_col].astype(str)
    df["text_clean"] = df["text_raw"]

    # ---- Step 1 垃圾过滤 ----
    print("\n[Step 1] 垃圾/机器人帖过滤...")
    df["is_rubbish"] = df["text_clean"].progress_apply(is_rubbish)
    before = len(df)
    df = df[~df["is_rubbish"]].drop(columns=["is_rubbish"])
    print(f"  {before} -> {len(df)} (删除 {before - len(df)})")

    # ---- Step 2 去重 + 关键词过滤 ----
    print("\n[Step 2] 文本去重...")
    before = len(df)
    df = df.drop_duplicates(subset=["text_clean"], keep='first')
    print(f"  {before} -> {len(df)} (删除 {before - len(df)})")

    if args.keywords:
        keywords = load_word_set(args.keywords)
        if keywords:
            pattern = re.compile("|".join(map(re.escape, keywords)))
            before = len(df)
            df = df[df["text_clean"].apply(lambda x: bool(pattern.search(str(x))))].copy()
            print(f"  关键词过滤: {before} -> {len(df)} (删除 {before - len(df)})")

    # ---- Step 3 去噪 ----
    print("\n[Step 3] 去噪 (@/话题/emoji/空格)...")
    df["text_clean"] = df["text_clean"].progress_apply(denoise_text)

    # ---- Step 4 分词 + 停用词 ----
    print("\n[Step 4] jieba 分词...")
    df["tokens"] = df["text_clean"].progress_apply(lambda x: jieba.lcut(str(x), HMM=True))
    stopwords = load_word_set(args.stopwords)
    if stopwords:
        print("  移除停用词...")
        df["tokens_nostop"] = df["tokens"].progress_apply(
            lambda toks: [w for w in toks if w not in stopwords and w.strip()])
    else:
        df["tokens_nostop"] = df["tokens"]

    df["tokens_str"] = df["tokens"].apply(lambda x: " ".join(x))
    df["tokens_nostop_str"] = df["tokens_nostop"].apply(lambda x: " ".join(x))

    # ---- Step 5 保存 ----
    keep = ['id', '用户id', '用户昵称', '发布时间', '发布位置', '转发数', '评论数', '点赞数',
            'source_dir', 'source_file', 'ip', 'retweet_id', 'user_authentication',
            '会员类型', '会员等级', 'text_raw', 'text_clean', 'tokens_str', 'tokens_nostop_str']
    out_cols = [c for c in keep if c in df.columns]
    df[out_cols].to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n保存 {len(df)} 行 -> {args.output}")
    print(f"输出列: {out_cols}")


if __name__ == '__main__':
    main()
