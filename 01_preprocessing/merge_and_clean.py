#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 第一步用的脚本: 把各个关键词文件夹里的 csv 全部读进来合成一个大表,
# 然后做最简单的清洗 (去重、去空值、把数字列转成数值)
# 运行: python merge_and_clean.py --root 数据所在目录 --out-merged xxx.csv --out-cleaned xxx.csv
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

EXCLUDE_FILES = {
    'merged_output.csv', 'merged_all_data.csv', 'cleaned_weibo_data.csv',
    'combined_data.csv', 'weibo_cleaned_data.csv', 'filtered_posts.csv',
    'filtered_posts1.csv', 'weibo_cleaned_full.csv',
}
EXCLUDE_DIRS = {'.ipynb_checkpoints', 'Geo_analysis'}


def read_csv_robust(file_path):
    """多种编码尝试读取 CSV"""
    for enc in ['utf-8-sig', 'utf-8', 'gb18030', 'gbk', 'latin1']:
        try:
            return pd.read_csv(file_path, encoding=enc, dtype=str, low_memory=False)
        except Exception:
            continue
    try:
        return pd.read_csv(file_path, dtype=str, low_memory=False, encoding_errors='replace')
    except Exception as e:
        print(f"  [错误] 无法读取: {file_path} -> {e}")
        return None


def collect_csv_files(root_folder):
    csv_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for filename in filenames:
            if filename.lower().endswith('.csv') and filename not in EXCLUDE_FILES:
                csv_files.append(os.path.join(dirpath, filename))
    return csv_files


def merge_csv_files(csv_files):
    frames = []
    print(f"\n开始合并 {len(csv_files)} 个 CSV 文件...\n")
    for i, file_path in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] 读取: {file_path}")
        df = read_csv_robust(file_path)
        if df is not None and not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            df['source_dir'] = Path(file_path).parent.name
            df['source_file'] = Path(file_path).name
            frames.append(df)
            print(f"         行数: {len(df)}")
        else:
            print("         [警告] 文件为空或读取失败")
    if not frames:
        return None
    merged_df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\n合并完成! 总行数: {len(merged_df)}")
    return merged_df


def clean_data(df):
    print("\n开始数据清洗...\n")
    original_count = len(df)
    print(f"原始数据行数: {original_count}")

    df = df.drop_duplicates()
    print(f"[1] 去除完全重复行后: {len(df)} 行")

    if 'id' in df.columns:
        df = df.drop_duplicates(subset=['id'], keep='first')
        print(f"[2] 基于 ID 去重后: {len(df)} 行")

    if '微博正文' in df.columns:
        before = len(df)
        df = df.dropna(subset=['微博正文'])
        df = df[df['微博正文'].str.strip() != '']
        print(f"[3] 去除 '微博正文' 为空后: {len(df)} 行 (删除 {before - len(df)})")

    for col in ['微博正文', '发布位置', '用户id', '用户昵称']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

    for col in ['转发数', '评论数', '点赞数']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    if '发布时间' in df.columns:
        df['发布时间'] = pd.to_datetime(df['发布时间'], errors='coerce')

    print(f"\n清洗完成! 最终 {len(df)} 行, 共删除 {original_count - len(df)} 行, "
          f"保留 {len(df)/original_count*100:.2f}%")
    return df


def generate_report(df):
    print("\n" + "=" * 60 + "\n              数据清洗报告\n" + "=" * 60)
    print(f"\n总行数: {len(df)}  总列数: {len(df.columns)}")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  - {col}: {non_null} 非空 ({non_null/len(df)*100:.1f}%)")
    if 'source_dir' in df.columns:
        print("\n各主题分布 (Top 20):")
        for topic, count in df['source_dir'].value_counts().head(20).items():
            print(f"  - {topic}: {count} 条")
    if '发布时间' in df.columns and df['发布时间'].notna().any():
        print(f"\n时间范围: {df['发布时间'].min()} ~ {df['发布时间'].max()}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="微博数据 CSV 合并与清洗")
    ap.add_argument('--root', required=True, help='CSV 根目录')
    ap.add_argument('--out-merged', default='merged_all_data.csv')
    ap.add_argument('--out-cleaned', default='cleaned_weibo_data.csv')
    args = ap.parse_args()

    out_merged = os.path.join(args.root, args.out_merged)
    out_cleaned = os.path.join(args.root, args.out_cleaned)

    print("=" * 60 + f"\n微博数据合并与清洗  {datetime.now():%Y-%m-%d %H:%M:%S}\n" + "=" * 60)
    csv_files = collect_csv_files(args.root)
    print(f"找到 {len(csv_files)} 个 CSV 文件")
    if not csv_files:
        sys.exit("未找到 CSV 文件")

    merged_df = merge_csv_files(csv_files)
    if merged_df is None or merged_df.empty:
        sys.exit("合并失败或数据为空")

    merged_df.to_csv(out_merged, index=False, encoding='utf-8-sig')
    print(f"已保存合并数据: {out_merged}")

    cleaned_df = clean_data(merged_df)
    cleaned_df.to_csv(out_cleaned, index=False, encoding='utf-8-sig')
    print(f"已保存清洗数据: {out_cleaned}")
    generate_report(cleaned_df)


if __name__ == '__main__':
    main()
