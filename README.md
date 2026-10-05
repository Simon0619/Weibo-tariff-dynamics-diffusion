# Weibo Tariff Dynamics Diffusion

## Overview

This repository contains the revised code for the paper:

**Spatio-temporal Dynamics and Network Diffusion of Weibo Discussions on Tariff-related Events**

The project analyzes Weibo posts from multiple perspectives, including preprocessing, sentiment classification, temporal dynamics, regional variation, spatial modeling, and repost network diffusion.

## Project Structure

```text
.
├── 01_preprocessing/      # Data merging and cleaning scripts
├── 02_sentiment/          # Sentiment labeling, training, and inference
├── 03_temporal/           # Temporal trend analysis
├── 04_regional/           # Regional comparison analysis
├── 05_spatial/            # Spatial and GWR analysis
├── 06_network/            # Repost cascade and network diffusion analysis
├── data/                  # Data files, not uploaded due to size limits
├── models/                # Local model files, not uploaded due to size limits
├── outputs/               # Generated outputs
├── config.py              # Configuration file
├── requirements           # Python dependencies
└── README.md              # Project description
Data Notice
Large raw data files are not included in this repository due to GitHub file size limits.
Please place the required data files under the local data/ directory before running the scripts.
Workflow
1. Preprocess raw Weibo data:
python 01_preprocessing/merge_and_clean.py
python 01_preprocessing/weibo_data_cleaning.py
2. Run sentiment analysis:
python 02_sentiment/infer_sentiment.py
3. Analyze temporal dynamics:
python 03_temporal/temporal_dynamics.py
4. Analyze regional and spatial patterns:
python 04_regional/cross_regional.py
python 05_spatial/gwr_analysis.py
5. Build repost diffusion network:
python 06_network/retweet_chain_builder.py
python 06_network/cascade_analysis.py
Requirements
Install dependencies with:
pip install -r requirements
Notes
This repository is intended for academic research and reproducible analysis.