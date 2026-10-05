# -*- coding: utf-8 -*-
"""
微博情感分析程序 - 学术研究版本
================================

研究目标：分析中美贸易战背景下微博用户情绪极性与时空特征

模型说明：
We employ a pre-trained Chinese RoBERTa model fine-tuned for binary sentiment 
classification (UER RoBERTa-JD), which outputs a continuous sentiment probability 
score in the range [0,1], representing positive emotional polarity.

功能模块：
1. BERT 情感分析 - 情绪极性强度 (emotional polarity intensity)
2. 时间序列分析 - 事件驱动的情感演变
3. 空间分析支持 - 省级情感指标
4. 网络分析支持 - 节点情感属性

作者：Weibo Summary Project
日期：2026-01-05
"""

# =============================================================================
# 第一部分：环境设置与依赖导入
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 可视化库
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 第二部分：全局配置参数
# =============================================================================

class Config:
    """全局配置参数"""
    
    # ---------- 模型配置 ----------
    # 使用预训练的中文RoBERTa情感分类模型
    BERT_MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"
    
    # ---------- 数据列配置 ----------
    TEXT_COLUMN = "text_clean"          # 输入文本列（已清洗）
    SENTIMENT_COLUMN = "sentiment_score"  # 输出情感分值列
    TIME_COLUMN = "发布时间"              # 时间列
    PROVINCE_COLUMN = "发布位置"          # 省份/位置列（用于空间分析）
    USER_COLUMN = "用户昵称"              # 用户列（用于网络分析）
    
    # ---------- 时间范围配置 ----------
    # 主分析时间范围
    ANALYSIS_START_DATE = "2025-01-01"
    ANALYSIS_END_DATE = "2025-06-30"
    
    # 事件分析窗口 (April 2025 Tariff Shock Event)
    EVENT_CENTER_DATE = "2025-04-09"     # 事件中心日 t₀
    EVENT_PRE_DAYS = 7                    # 事件前天数
    EVENT_POST_DAYS = 14                  # 事件后天数
    EVENT_WINDOW_START = "2025-04-02"    # 事件窗口开始
    EVENT_WINDOW_END = "2025-04-23"      # 事件窗口结束
    
    # ---------- 情感阈值配置 ----------
    # 用于将连续分值转换为类别标签
    POSITIVE_THRESHOLD = 0.6   # sentiment_score >= 0.6 → 正面
    NEGATIVE_THRESHOLD = 0.4   # sentiment_score <= 0.4 → 负面
    # 0.4 < sentiment_score < 0.6 → 中性
    
    # ---------- 输出配置 ----------
    OUTPUT_DIR = r"d:\weibo_summary\analysis_output"
    DATA_PATH = r"d:\weibo_summary\weibo_cleaned_full.csv"


# =============================================================================
# 第三部分：数据加载与预处理
# =============================================================================

def load_data(file_path: str = None) -> pd.DataFrame:
    """
    加载清洗后的微博数据
    
    参数:
        file_path: CSV文件路径（默认使用Config.DATA_PATH）
        
    返回:
        DataFrame: 加载的数据
    """
    file_path = file_path or Config.DATA_PATH
    print(f"正在加载数据: {file_path}")
    
    # 尝试不同编码读取
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"✓ 成功使用 {encoding} 编码加载数据")
            print(f"  数据形状: {df.shape}")
            print(f"  列名: {list(df.columns)}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"✗ 加载出错: {e}")
            continue
    
    raise ValueError(f"无法加载数据文件: {file_path}")


def filter_by_time_range(df: pd.DataFrame, 
                         start_date: str = None, 
                         end_date: str = None,
                         time_column: str = None) -> pd.DataFrame:
    """
    按时间范围筛选数据
    
    参数:
        df: 输入DataFrame
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        time_column: 时间列名
        
    返回:
        DataFrame: 筛选后的数据
    """
    time_column = time_column or Config.TIME_COLUMN
    start_date = start_date or Config.ANALYSIS_START_DATE
    end_date = end_date or Config.ANALYSIS_END_DATE
    
    # 转换时间列
    df['datetime'] = pd.to_datetime(df[time_column], errors='coerce')
    
    # 筛选时间范围
    mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)
    filtered_df = df[mask].copy()
    
    print(f"✓ 时间筛选: {start_date} ~ {end_date}")
    print(f"  筛选前: {len(df)} 条 → 筛选后: {len(filtered_df)} 条")
    
    return filtered_df


# =============================================================================
# 第四部分：BERT 情感分析
# =============================================================================

class BertSentimentAnalyzer:
    """
    基于BERT的中文情感极性分析器
    
    模型输出：
    - sentiment_score ∈ (0,1)
    - 越接近 1 → 情绪越正向 / 越强烈支持
    - 越接近 0 → 情绪越负向 / 越强烈反对
    
    这是"情绪极性强度"(emotional polarity intensity)，
    而非"情绪类型分类"(如愤怒/恐惧/喜悦)
    """
    
    def __init__(self, model_name: str = None):
        """
        初始化BERT情感分析器
        
        参数:
            model_name: HuggingFace模型名称
        """
        self.model_name = model_name or Config.BERT_MODEL_NAME
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = -1  # CPU: -1, GPU: 0
        
    def load_model(self):
        """加载BERT模型和分词器"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
            
            print(f"正在加载模型: {self.model_name}")
            print("  (首次加载可能需要下载模型文件，请耐心等待...)")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # 检查是否有GPU
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = 0
                    print("  ✓ 检测到GPU，将使用GPU加速")
                else:
                    print("  ✓ 未检测到GPU，将使用CPU")
            except:
                print("  ✓ 将使用CPU")
            
            self.pipeline = pipeline(
                "sentiment-analysis", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=self.device,
                max_length=512,
                truncation=True
            )
            
            print("✓ 模型加载完成！")
            
        except ImportError:
            print("✗ 请先安装依赖库:")
            print("  pip install transformers torch")
            raise
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise
    
    def _extract_score(self, result: dict) -> float:
        """
        从模型输出中提取情感分值
        
        模型输出格式可能是:
        - {'label': 'LABEL_1', 'score': 0.95}  # LABEL_1 = positive
        - {'label': 'LABEL_0', 'score': 0.80}  # LABEL_0 = negative
        
        我们统一转换为: 0-1 分值，1=正面，0=负面
        """
        label = result.get('label', '')
        score = result.get('score', 0.5)
        
        # 如果是负面标签，反转分数
        if 'LABEL_0' in label or 'negative' in label.lower() or 'neg' in label.lower():
            return 1 - score
        else:
            return score
    
    def analyze_single(self, text: str) -> float:
        """
        分析单条文本的情感分值
        
        参数:
            text: 输入文本
            
        返回:
            float: 情感分值 (0-1)
        """
        if self.pipeline is None:
            self.load_model()
        
        if not text or not isinstance(text, str) or not text.strip():
            return 0.5  # 空文本返回中性
        
        try:
            result = self.pipeline(text[:500])[0]  # 截断至500字符
            return self._extract_score(result)
        except Exception as e:
            return 0.5  # 出错返回中性
    
    def analyze_batch(self, texts: list, batch_size: int = 32) -> list:
        """
        批量分析文本情感
        
        参数:
            texts: 文本列表
            batch_size: 批处理大小
            
        返回:
            list: 情感分值列表
        """
        if self.pipeline is None:
            self.load_model()
        
        scores = []
        
        # 预处理文本
        processed_texts = []
        for t in texts:
            if t and isinstance(t, str) and t.strip():
                processed_texts.append(t[:500])
            else:
                processed_texts.append("")
        
        # 批量处理
        print(f"开始情感分析，共 {len(processed_texts)} 条文本...")
        
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="情感分析"):
            batch = processed_texts[i:i + batch_size]
            
            # 过滤空文本
            valid_indices = [j for j, t in enumerate(batch) if t.strip()]
            valid_texts = [batch[j] for j in valid_indices]
            
            # 初始化批次分数
            batch_scores = [0.5] * len(batch)  # 默认中性
            
            if valid_texts:
                try:
                    batch_results = self.pipeline(valid_texts)
                    for idx, result in zip(valid_indices, batch_results):
                        batch_scores[idx] = self._extract_score(result)
                except Exception as e:
                    pass  # 保持默认中性分数
            
            scores.extend(batch_scores)
        
        return scores
    
    def analyze_dataframe(self, df: pd.DataFrame, 
                         text_column: str = None,
                         output_column: str = None) -> pd.DataFrame:
        """
        对DataFrame进行情感分析
        
        参数:
            df: 输入DataFrame
            text_column: 文本列名
            output_column: 输出列名
            
        返回:
            DataFrame: 添加情感分析结果的DataFrame
        """
        text_column = text_column or Config.TEXT_COLUMN
        output_column = output_column or Config.SENTIMENT_COLUMN
        
        texts = df[text_column].fillna('').tolist()
        scores = self.analyze_batch(texts)
        
        df[output_column] = scores
        
        # 添加情感类别标签
        df['sentiment_category'] = df[output_column].apply(self._categorize_sentiment)
        
        # 输出统计信息
        print("\n✓ 情感分析完成！")
        print(f"  情感分值分布:")
        print(f"    均值: {df[output_column].mean():.4f}")
        print(f"    标准差: {df[output_column].std():.4f}")
        print(f"    最小值: {df[output_column].min():.4f}")
        print(f"    最大值: {df[output_column].max():.4f}")
        print(f"\n  情感类别分布:")
        print(df['sentiment_category'].value_counts().to_string())
        
        return df
    
    def _categorize_sentiment(self, score: float) -> str:
        """
        将情感分值转换为类别标签
        
        参数:
            score: 情感分值 (0-1)
            
        返回:
            str: 情感类别 (Positive/Negative/Neutral)
        """
        if score >= Config.POSITIVE_THRESHOLD:
            return 'Positive'
        elif score <= Config.NEGATIVE_THRESHOLD:
            return 'Negative'
        else:
            return 'Neutral'


# =============================================================================
# 第五部分：时间序列分析
# =============================================================================

class TimeSeriesEventAnalyzer:
    """
    基于时间的事件时间序列分析器
    
    核心功能：
    - 微博发布量时间趋势
    - 情感变化趋势（事件前后对比）
    - 事件窗口分析（April 2025 Tariff Shock Event）
    - 情绪"相变"检测
    """
    
    def __init__(self, df: pd.DataFrame, time_column: str = None):
        """
        初始化时间序列分析器
        
        参数:
            df: 包含时间和情感信息的DataFrame
            time_column: 时间列名
        """
        self.df = df.copy()
        self.time_column = time_column or Config.TIME_COLUMN
        self._preprocess_time()
        
    def _preprocess_time(self):
        """预处理时间列"""
        # 转换时间格式
        if 'datetime' not in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df[self.time_column], errors='coerce')
        
        # 提取时间特征
        self.df['date'] = self.df['datetime'].dt.date
        self.df['year'] = self.df['datetime'].dt.year
        self.df['month'] = self.df['datetime'].dt.month
        self.df['day'] = self.df['datetime'].dt.day
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['weekday'] = self.df['datetime'].dt.weekday
        self.df['year_month'] = self.df['datetime'].dt.to_period('M')
        
        # 过滤有效时间记录
        valid_count = self.df['datetime'].notna().sum()
        print(f"✓ 时间范围: {self.df['datetime'].min()} ~ {self.df['datetime'].max()}")
        print(f"  有效时间记录: {valid_count} / {len(self.df)}")
    
    def volume_trend(self, freq: str = 'D', output_path: str = None) -> pd.DataFrame:
        """
        分析微博发布量趋势
        
        参数:
            freq: 时间频率 ('D'日, 'W'周, 'M'月, 'H'小时)
            output_path: 图片保存路径
            
        返回:
            DataFrame: 时间序列统计
        """
        # 按时间分组统计
        volume = self.df.set_index('datetime').resample(freq).size()
        volume_df = volume.reset_index()
        volume_df.columns = ['datetime', 'count']
        
        # 可视化
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(volume_df['datetime'], volume_df['count'], linewidth=1.5, color='#2196F3')
        ax.fill_between(volume_df['datetime'], volume_df['count'], alpha=0.3, color='#2196F3')
        
        # 标记事件窗口
        event_start = pd.to_datetime(Config.EVENT_WINDOW_START)
        event_end = pd.to_datetime(Config.EVENT_WINDOW_END)
        event_center = pd.to_datetime(Config.EVENT_CENTER_DATE)
        
        ax.axvspan(event_start, event_end, alpha=0.2, color='red', label='Event Window')
        ax.axvline(event_center, color='red', linestyle='--', linewidth=2, label=f't₀ = {Config.EVENT_CENTER_DATE}')
        
        ax.set_title('微博发布量时间趋势 (Volume Trend)', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间 (Time)')
        ax.set_ylabel('发布数量 (Count)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图片已保存: {output_path}")
        
        plt.show()
        
        return volume_df
    
    def sentiment_trend(self, freq: str = 'D', output_path: str = None) -> pd.DataFrame:
        """
        分析情感变化趋势
        
        参数:
            freq: 时间频率
            output_path: 图片保存路径
            
        返回:
            DataFrame: 情感趋势统计
        """
        sentiment_col = Config.SENTIMENT_COLUMN
        
        if sentiment_col not in self.df.columns:
            raise ValueError("请先运行情感分析")
        
        # 按时间分组计算情感统计
        sentiment_ts = self.df.set_index('datetime').resample(freq).agg({
            sentiment_col: ['mean', 'std', 'count']
        })
        sentiment_ts.columns = ['mean_sentiment', 'std_sentiment', 'count']
        sentiment_ts = sentiment_ts.reset_index()
        
        # 可视化
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 子图1：平均情感分值
        ax1 = axes[0]
        ax1.plot(sentiment_ts['datetime'], sentiment_ts['mean_sentiment'], 
                linewidth=2, color='#4CAF50', label='Mean Sentiment')
        ax1.fill_between(sentiment_ts['datetime'], 
                        sentiment_ts['mean_sentiment'] - sentiment_ts['std_sentiment'],
                        sentiment_ts['mean_sentiment'] + sentiment_ts['std_sentiment'],
                        alpha=0.2, color='#4CAF50', label='±1 Std')
        
        # 添加阈值线
        ax1.axhline(Config.POSITIVE_THRESHOLD, color='green', linestyle=':', alpha=0.7, label='Positive Threshold')
        ax1.axhline(Config.NEGATIVE_THRESHOLD, color='red', linestyle=':', alpha=0.7, label='Negative Threshold')
        
        # 标记事件窗口
        event_start = pd.to_datetime(Config.EVENT_WINDOW_START)
        event_end = pd.to_datetime(Config.EVENT_WINDOW_END)
        event_center = pd.to_datetime(Config.EVENT_CENTER_DATE)
        
        ax1.axvspan(event_start, event_end, alpha=0.2, color='red')
        ax1.axvline(event_center, color='red', linestyle='--', linewidth=2)
        
        ax1.set_ylabel('情感分值 (Sentiment Score)')
        ax1.set_title('情感变化趋势 (Sentiment Trend)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 子图2：发布量
        ax2 = axes[1]
        ax2.bar(sentiment_ts['datetime'], sentiment_ts['count'], 
               width=1, color='#2196F3', alpha=0.7, label='Daily Count')
        ax2.axvspan(event_start, event_end, alpha=0.2, color='red', label='Event Window')
        ax2.axvline(event_center, color='red', linestyle='--', linewidth=2, label=f't₀')
        
        ax2.set_xlabel('时间 (Time)')
        ax2.set_ylabel('发布数量 (Count)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图片已保存: {output_path}")
        
        plt.show()
        
        return sentiment_ts
    
    def event_window_analysis(self, output_path: str = None) -> dict:
        """
        事件窗口分析 (April 2025 Tariff Shock Event)
        
        比较事件前后的情感变化
        
        参数:
            output_path: 图片保存路径
            
        返回:
            dict: 事件分析结果
        """
        sentiment_col = Config.SENTIMENT_COLUMN
        
        if sentiment_col not in self.df.columns:
            raise ValueError("请先运行情感分析")
        
        event_center = pd.to_datetime(Config.EVENT_CENTER_DATE)
        event_start = pd.to_datetime(Config.EVENT_WINDOW_START)
        event_end = pd.to_datetime(Config.EVENT_WINDOW_END)
        
        # 划分时间段
        pre_event = self.df[(self.df['datetime'] >= event_start) & 
                           (self.df['datetime'] < event_center)]
        post_event = self.df[(self.df['datetime'] >= event_center) & 
                            (self.df['datetime'] <= event_end)]
        
        # 计算统计量
        results = {
            'event_center': Config.EVENT_CENTER_DATE,
            'window': f"{Config.EVENT_WINDOW_START} ~ {Config.EVENT_WINDOW_END}",
            'pre_event': {
                'count': len(pre_event),
                'mean_sentiment': pre_event[sentiment_col].mean(),
                'std_sentiment': pre_event[sentiment_col].std(),
                'positive_ratio': (pre_event[sentiment_col] >= Config.POSITIVE_THRESHOLD).mean(),
                'negative_ratio': (pre_event[sentiment_col] <= Config.NEGATIVE_THRESHOLD).mean(),
            },
            'post_event': {
                'count': len(post_event),
                'mean_sentiment': post_event[sentiment_col].mean(),
                'std_sentiment': post_event[sentiment_col].std(),
                'positive_ratio': (post_event[sentiment_col] >= Config.POSITIVE_THRESHOLD).mean(),
                'negative_ratio': (post_event[sentiment_col] <= Config.NEGATIVE_THRESHOLD).mean(),
            }
        }
        
        # 计算变化量
        results['change'] = {
            'sentiment_change': results['post_event']['mean_sentiment'] - results['pre_event']['mean_sentiment'],
            'positive_ratio_change': results['post_event']['positive_ratio'] - results['pre_event']['positive_ratio'],
            'negative_ratio_change': results['post_event']['negative_ratio'] - results['pre_event']['negative_ratio'],
        }
        
        # 打印结果
        print("\n" + "=" * 60)
        print("事件窗口分析 (April 2025 Tariff Shock Event)")
        print("=" * 60)
        print(f"事件中心日 t₀: {results['event_center']}")
        print(f"分析窗口: {results['window']}")
        print(f"\n事件前 (Pre-event, t₀ - 7 days):")
        print(f"  样本量: {results['pre_event']['count']}")
        print(f"  平均情感: {results['pre_event']['mean_sentiment']:.4f}")
        print(f"  正面比例: {results['pre_event']['positive_ratio']:.2%}")
        print(f"  负面比例: {results['pre_event']['negative_ratio']:.2%}")
        print(f"\n事件后 (Post-event, t₀ + 14 days):")
        print(f"  样本量: {results['post_event']['count']}")
        print(f"  平均情感: {results['post_event']['mean_sentiment']:.4f}")
        print(f"  正面比例: {results['post_event']['positive_ratio']:.2%}")
        print(f"  负面比例: {results['post_event']['negative_ratio']:.2%}")
        print(f"\n变化量 (Change):")
        print(f"  情感变化: {results['change']['sentiment_change']:+.4f}")
        print(f"  正面比例变化: {results['change']['positive_ratio_change']:+.2%}")
        print(f"  负面比例变化: {results['change']['negative_ratio_change']:+.2%}")
        print("=" * 60)
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 子图1：情感分布对比
        ax1 = axes[0]
        labels = ['Pre-event', 'Post-event']
        means = [results['pre_event']['mean_sentiment'], results['post_event']['mean_sentiment']]
        stds = [results['pre_event']['std_sentiment'], results['post_event']['std_sentiment']]
        colors = ['#2196F3', '#F44336']
        
        bars = ax1.bar(labels, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
        ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('平均情感分值 (Mean Sentiment)')
        ax1.set_title('事件前后情感对比', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, mean in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 子图2：正负情绪比例对比
        ax2 = axes[1]
        x = np.arange(2)
        width = 0.35
        
        positive_ratios = [results['pre_event']['positive_ratio'], results['post_event']['positive_ratio']]
        negative_ratios = [results['pre_event']['negative_ratio'], results['post_event']['negative_ratio']]
        
        bars1 = ax2.bar(x - width/2, positive_ratios, width, label='Positive', color='#4CAF50', alpha=0.8)
        bars2 = ax2.bar(x + width/2, negative_ratios, width, label='Negative', color='#F44336', alpha=0.8)
        
        ax2.set_ylabel('比例 (Ratio)')
        ax2.set_title('正负情绪比例对比', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图片已保存: {output_path}")
        
        plt.show()
        
        return results
    
    def hourly_pattern(self, output_path: str = None) -> pd.DataFrame:
        """
        分析每小时发布和情感模式
        
        参数:
            output_path: 图片保存路径
            
        返回:
            DataFrame: 小时分布统计
        """
        sentiment_col = Config.SENTIMENT_COLUMN
        
        hourly_stats = self.df.groupby('hour').agg({
            sentiment_col: ['mean', 'count'] if sentiment_col in self.df.columns else 'count'
        })
        
        if sentiment_col in self.df.columns:
            hourly_stats.columns = ['mean_sentiment', 'count']
        else:
            hourly_stats.columns = ['count']
            hourly_stats['mean_sentiment'] = 0.5
        
        hourly_stats = hourly_stats.reset_index()
        
        # 可视化
        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        # 柱状图：发布量
        bars = ax1.bar(hourly_stats['hour'], hourly_stats['count'], 
                      color='#2196F3', alpha=0.7, label='发布量')
        ax1.set_xlabel('小时 (Hour)')
        ax1.set_ylabel('发布数量 (Count)', color='#2196F3')
        ax1.tick_params(axis='y', labelcolor='#2196F3')
        ax1.set_xticks(range(24))
        
        # 折线图：平均情感
        if sentiment_col in self.df.columns:
            ax2 = ax1.twinx()
            ax2.plot(hourly_stats['hour'], hourly_stats['mean_sentiment'], 
                    color='#F44336', linewidth=2, marker='o', label='平均情感')
            ax2.set_ylabel('平均情感 (Mean Sentiment)', color='#F44336')
            ax2.tick_params(axis='y', labelcolor='#F44336')
            ax2.set_ylim(0, 1)
        
        ax1.set_title('每小时发布量与情感分布', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图片已保存: {output_path}")
        
        plt.show()
        
        return hourly_stats


# =============================================================================
# 第六部分：空间分析支持
# =============================================================================

class SpatialSentimentAnalyzer:
    """
    省级空间情感分析器
    
    计算指标：
    - 省级平均情感值
    - 省级情感方差
    - 正/负情绪比例
    """
    
    def __init__(self, df: pd.DataFrame, province_column: str = None):
        """
        初始化空间分析器
        
        参数:
            df: 包含情感和位置信息的DataFrame
            province_column: 省份列名
        """
        self.df = df.copy()
        self.province_column = province_column or Config.PROVINCE_COLUMN
        
    def provincial_statistics(self) -> pd.DataFrame:
        """
        计算省级情感统计
        
        返回:
            DataFrame: 各省情感统计
        """
        sentiment_col = Config.SENTIMENT_COLUMN
        
        if sentiment_col not in self.df.columns:
            raise ValueError("请先运行情感分析")
        
        if self.province_column not in self.df.columns:
            raise ValueError(f"数据中不存在列: {self.province_column}")
        
        # 按省份分组统计
        provincial = self.df.groupby(self.province_column).agg({
            sentiment_col: ['mean', 'std', 'count']
        })
        provincial.columns = ['mean_sentiment', 'std_sentiment', 'count']
        
        # 计算正负比例
        def calc_ratios(group):
            return pd.Series({
                'positive_ratio': (group[sentiment_col] >= Config.POSITIVE_THRESHOLD).mean(),
                'negative_ratio': (group[sentiment_col] <= Config.NEGATIVE_THRESHOLD).mean(),
                'neutral_ratio': ((group[sentiment_col] > Config.NEGATIVE_THRESHOLD) & 
                                 (group[sentiment_col] < Config.POSITIVE_THRESHOLD)).mean()
            })
        
        ratios = self.df.groupby(self.province_column).apply(calc_ratios)
        provincial = provincial.join(ratios)
        provincial = provincial.reset_index()
        provincial = provincial.rename(columns={self.province_column: 'province'})
        
        # 按平均情感排序
        provincial = provincial.sort_values('mean_sentiment', ascending=False)
        
        print("\n✓ 省级情感统计:")
        print(provincial.to_string())
        
        return provincial
    
    def visualize_provincial(self, output_path: str = None) -> None:
        """
        可视化省级情感分布
        
        参数:
            output_path: 图片保存路径
        """
        stats = self.provincial_statistics()
        
        # 只显示前20个省份
        top_provinces = stats.head(20)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['#4CAF50' if x >= Config.POSITIVE_THRESHOLD 
                 else '#F44336' if x <= Config.NEGATIVE_THRESHOLD 
                 else '#9E9E9E' 
                 for x in top_provinces['mean_sentiment']]
        
        bars = ax.barh(range(len(top_provinces)), top_provinces['mean_sentiment'], 
                      color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(top_provinces)))
        ax.set_yticklabels(top_provinces['province'])
        ax.set_xlabel('平均情感分值 (Mean Sentiment)')
        ax.set_title('省级情感分布 (Top 20)', fontsize=14, fontweight='bold')
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(Config.POSITIVE_THRESHOLD, color='green', linestyle=':', alpha=0.5)
        ax.axvline(Config.NEGATIVE_THRESHOLD, color='red', linestyle=':', alpha=0.5)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图片已保存: {output_path}")
        
        plt.show()


# =============================================================================
# 第七部分：网络分析支持
# =============================================================================

class NetworkSentimentAnalyzer:
    """
    网络分析情感属性计算器
    
    为网络分析提供节点情感属性：
    - 用户平均情感值 (mean_sentiment)
    - 发布数量
    - 情感一致性（标准差）
    """
    
    def __init__(self, df: pd.DataFrame, user_column: str = None):
        """
        初始化网络情感分析器
        
        参数:
            df: 包含情感和用户信息的DataFrame
            user_column: 用户列名
        """
        self.df = df.copy()
        self.user_column = user_column or Config.USER_COLUMN
        
    def user_sentiment_attributes(self) -> pd.DataFrame:
        """
        计算用户级情感属性（作为网络节点属性）
        
        返回:
            DataFrame: 用户情感属性
        """
        sentiment_col = Config.SENTIMENT_COLUMN
        
        if sentiment_col not in self.df.columns:
            raise ValueError("请先运行情感分析")
        
        if self.user_column not in self.df.columns:
            raise ValueError(f"数据中不存在列: {self.user_column}")
        
        # 按用户分组统计
        user_stats = self.df.groupby(self.user_column).agg({
            sentiment_col: ['mean', 'std', 'count']
        })
        user_stats.columns = ['mean_sentiment', 'std_sentiment', 'post_count']
        user_stats = user_stats.reset_index()
        user_stats = user_stats.rename(columns={self.user_column: 'user'})
        
        # 填充缺失的标准差
        user_stats['std_sentiment'] = user_stats['std_sentiment'].fillna(0)
        
        # 计算情感极端性（距离中性的程度）
        user_stats['sentiment_extremity'] = abs(user_stats['mean_sentiment'] - 0.5)
        
        # 分类用户情感倾向
        user_stats['sentiment_type'] = user_stats['mean_sentiment'].apply(
            lambda x: 'Positive' if x >= Config.POSITIVE_THRESHOLD 
                      else 'Negative' if x <= Config.NEGATIVE_THRESHOLD 
                      else 'Neutral'
        )
        
        print(f"\n✓ 用户情感属性计算完成:")
        print(f"  总用户数: {len(user_stats)}")
        print(f"  正面倾向用户: {(user_stats['sentiment_type'] == 'Positive').sum()}")
        print(f"  负面倾向用户: {(user_stats['sentiment_type'] == 'Negative').sum()}")
        print(f"  中性用户: {(user_stats['sentiment_type'] == 'Neutral').sum()}")
        
        return user_stats
    
    def export_for_network(self, output_path: str = None) -> pd.DataFrame:
        """
        导出用于网络分析的数据
        
        参数:
            output_path: 输出文件路径
            
        返回:
            DataFrame: 网络节点属性
        """
        user_attrs = self.user_sentiment_attributes()
        
        if output_path:
            user_attrs.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✓ 网络节点属性已导出: {output_path}")
        
        return user_attrs


# =============================================================================
# 第八部分：综合分析流水线
# =============================================================================

class WeiboSentimentPipeline:
    """
    微博情感分析综合流水线
    
    整合：
    1. BERT情感分析
    2. 时间序列分析
    3. 事件窗口分析
    4. 空间分析
    5. 网络属性计算
    """
    
    def __init__(self, data_path: str = None):
        """
        初始化分析流水线
        
        参数:
            data_path: 数据文件路径
        """
        self.data_path = data_path or Config.DATA_PATH
        self.df = None
        self.output_dir = Config.OUTPUT_DIR
        
    def run(self, 
            run_sentiment: bool = True,
            run_timeseries: bool = True,
            run_event_analysis: bool = True,
            run_spatial: bool = False,
            run_network: bool = False,
            filter_time: bool = True) -> pd.DataFrame:
        """
        运行完整分析流水线
        
        参数:
            run_sentiment: 是否运行情感分析
            run_timeseries: 是否运行时间序列分析
            run_event_analysis: 是否运行事件窗口分析
            run_spatial: 是否运行空间分析
            run_network: 是否计算网络属性
            filter_time: 是否按时间范围筛选
        """
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. 加载数据
        print("\n" + "=" * 60)
        print("第1步：加载数据")
        print("=" * 60)
        self.df = load_data(self.data_path)
        
        # 2. 时间筛选
        if filter_time:
            print("\n" + "=" * 60)
            print("第2步：时间范围筛选")
            print("=" * 60)
            self.df = filter_by_time_range(self.df)
        
        # 3. 情感分析
        if run_sentiment:
            print("\n" + "=" * 60)
            print("第3步：BERT情感分析")
            print("=" * 60)
            analyzer = BertSentimentAnalyzer()
            self.df = analyzer.analyze_dataframe(self.df)
        
        # 4. 时间序列分析
        if run_timeseries and Config.SENTIMENT_COLUMN in self.df.columns:
            print("\n" + "=" * 60)
            print("第4步：时间序列分析")
            print("=" * 60)
            ts_analyzer = TimeSeriesEventAnalyzer(self.df)
            
            ts_analyzer.volume_trend('D', f"{self.output_dir}/volume_trend.png")
            ts_analyzer.sentiment_trend('D', f"{self.output_dir}/sentiment_trend.png")
            ts_analyzer.hourly_pattern(f"{self.output_dir}/hourly_pattern.png")
        
        # 5. 事件窗口分析
        if run_event_analysis and Config.SENTIMENT_COLUMN in self.df.columns:
            print("\n" + "=" * 60)
            print("第5步：事件窗口分析")
            print("=" * 60)
            ts_analyzer = TimeSeriesEventAnalyzer(self.df)
            ts_analyzer.event_window_analysis(f"{self.output_dir}/event_analysis.png")
        
        # 6. 空间分析
        if run_spatial and Config.PROVINCE_COLUMN in self.df.columns:
            print("\n" + "=" * 60)
            print("第6步：空间分析")
            print("=" * 60)
            spatial_analyzer = SpatialSentimentAnalyzer(self.df)
            spatial_analyzer.visualize_provincial(f"{self.output_dir}/provincial_sentiment.png")
        
        # 7. 网络属性
        if run_network and Config.USER_COLUMN in self.df.columns:
            print("\n" + "=" * 60)
            print("第7步：网络属性计算")
            print("=" * 60)
            network_analyzer = NetworkSentimentAnalyzer(self.df)
            network_analyzer.export_for_network(f"{self.output_dir}/network_node_attributes.csv")
        
        # 8. 保存结果
        output_file = f"{self.output_dir}/sentiment_analysis_result.csv"
        self.df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ 分析结果已保存: {output_file}")
        
        print("\n" + "=" * 60)
        print("✓ 分析完成！")
        print("=" * 60)
        
        return self.df


# =============================================================================
# 第九部分：主函数入口
# =============================================================================

if __name__ == "__main__":
    """
    运行方式：
    
    方式1：使用流水线一键分析
    >>> pipeline = WeiboSentimentPipeline()
    >>> result_df = pipeline.run()
    
    方式2：分步骤手动分析
    >>> df = load_data()
    >>> df = filter_by_time_range(df)
    >>> analyzer = BertSentimentAnalyzer()
    >>> df = analyzer.analyze_dataframe(df)
    >>> ts_analyzer = TimeSeriesEventAnalyzer(df)
    >>> ts_analyzer.sentiment_trend()
    """
    
    print("=" * 60)
    print("微博情感分析程序 - 学术研究版本")
    print("=" * 60)
    print(f"模型: {Config.BERT_MODEL_NAME}")
    print(f"数据: {Config.DATA_PATH}")
    print(f"时间范围: {Config.ANALYSIS_START_DATE} ~ {Config.ANALYSIS_END_DATE}")
    print(f"事件窗口: {Config.EVENT_WINDOW_START} ~ {Config.EVENT_WINDOW_END}")
    print(f"事件中心: {Config.EVENT_CENTER_DATE}")
    print("=" * 60)
    
    # 取消下面的注释以运行分析
    # pipeline = WeiboSentimentPipeline()
    # result_df = pipeline.run(
    #     run_sentiment=True,
    #     run_timeseries=True,
    #     run_event_analysis=True,
    #     run_spatial=True,
    #     run_network=True,
    #     filter_time=True
    # )
    
    print("\n程序已准备就绪！")
    print("取消上方代码块的注释以开始分析。")
