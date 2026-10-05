#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 从 hf-mirror 镜像站把 HuggingFace 的模型文件下到本地 models 文件夹。
# 这台机器系统证书太老, 直连会报 SSL 错, 所以这里关掉了证书校验。
# 运行: python download_model.py 模型名 (比如 uer/roberta-base-finetuned-jd-binary-chinese)
import argparse
import urllib3
from pathlib import Path

import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MIRROR = "https://hf-mirror.com"
# 需要的模型文件 (按优先级尝试)
CANDIDATE_FILES = [
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "vocab.txt",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
]


def download(repo_id, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = []
    for fname in CANDIDATE_FILES:
        url = f"{MIRROR}/{repo_id}/resolve/main/{fname}"
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  跳过(已存在): {fname}")
            ok.append(fname)
            continue
        try:
            r = requests.get(url, timeout=120, verify=False, stream=True)
            if r.status_code != 200:
                print(f"  [{r.status_code}] {fname} (不存在, 跳过)")
                continue
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            print(f"  下载完成: {fname} ({dest.stat().st_size/1e6:.1f} MB)")
            ok.append(fname)
        except Exception as e:
            print(f"  [失败] {fname}: {e}")
    if "config.json" not in ok:
        raise SystemExit("核心文件 config.json 下载失败")
    print(f"模型已就绪: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).resolve().parent.parent / "models" / args.repo_id.replace("/", "__")
    download(args.repo_id, out)
