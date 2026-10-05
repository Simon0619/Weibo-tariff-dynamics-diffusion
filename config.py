# -*- coding: utf-8 -*-
# 各个脚本共用的一些路径和参数都放在这里, 改数的时候只需要改这一个文件

from pathlib import Path


def setup_plot():
    # 画图的时候中文会显示成方块, 这里把系统里的中文字体加进去
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
               "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
    # 这个 ttc 注册进来的名字是 JP, 不过汉字都能正常显示
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC",
                                       "WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# 原始数据在 Master 里, 清洗后的数据复制了一份到项目的 data 文件夹
MASTER_DIR = Path("/home/dengjianwei/Master")
CODE_DIR = Path("/home/dengjianwei/Master-revised-code")
DATA_DIR = CODE_DIR / "data"

# 优先用项目内 data 里的副本, 找不到再去 Master 原目录拿
def _data_path(name):
    p = DATA_DIR / name
    return p if p.exists() else MASTER_DIR / name

DATA_FULL = _data_path("weibo_cleaned_full.csv")          # 清洗完的全量数据, 337546 条
KEYWORDS_FILE = _data_path("keywords.txt")                # 58 个检索关键词
CN_SHP = _data_path("Geo_analysis/cn_shp/cn.shp")        # 中国省级行政区边界
TEACHER_LABELS = _data_path("Test_outcomes/Chip_deepseek-v3.2.csv")  # 之前已经标好的 500 条

OUTPUT_DIR = CODE_DIR / "outputs"
MODEL_DIR = CODE_DIR / "models"
SENTIMENT_OUTPUT = OUTPUT_DIR / "weibo_sentiment.csv"     # 跑完情感模型后的大表

# 数据里各列的列名
COL_ID = "id"
COL_USER = "用户昵称"          # 这份数据没有 user_id, 只能拿昵称当用户标识
COL_TIME = "发布时间"
COL_TEXT = "text_clean"
COL_IP = "ip"                 # 省份或者海外国家名, 空值不少
COL_AUTH = "user_authentication"  # 普通用户/蓝V/黄V/红V/金V
COL_RETWEET = "retweet_id"
COL_KEYWORD = "source_dir"    # 这条是用哪个关键词爬下来的
COL_SENT = "sentiment_score"
COL_SENT_CAT = "sentiment_category"

# 情感分的划分: 模型给出的是正向概率, 再按阈值切成三类
POS_THRESHOLD = 0.6
NEG_THRESHOLD = 0.4
SENTIMENT_CLASSES = ["Negative", "Neutral", "Positive"]

# 分析的时间范围, 图文件名上写的也是 2025_01_07
ANALYSIS_START = "2025-01-01"
ANALYSIS_END = "2025-07-07"

# 阶段自动检测用的参数: 前 30 天当基线, 7 日滑动平均, 偏离基线 0.20 算异常
SMA_WINDOW = 7
TAU = 0.20
BASELINE_DAYS = 30

# 检测出来的三个事件阶段 (画生命周期图和 GWR 分阶段建模都要用)
PHASES = [
    {"name": "Phase 1", "center": "2025-02-13", "start": "2025-02-06", "end": "2025-02-27"},
    {"name": "Phase 2", "center": "2025-04-09", "start": "2025-04-02", "end": "2025-04-23"},
    {"name": "Phase 3", "center": "2025-05-14", "start": "2025-05-07", "end": "2025-05-28"},
]
PHASE_WINDOW_DAYS = 15   # 生命周期图取中心日前后各 15 天

# 关键词分到四个主题簇, 一条微博可以同时属于多个簇
KEYWORD_CLUSTERS = {
    "Strategic": ["美国", "特朗普", "拜登", "欧盟", "WTO", "301调查", "稀土", "经济制裁",
                  "反制", "反制措施", "豁免", "民意", "政治舆论", "政策反馈", "抗议",
                  "双边关系", "国际合作", "合作", "协议", "中美达成协议", "多边贸易"],
    "Trade": ["贸易战", "关税", "关税政策", "贸易政策", "贸易壁垒", "壁垒", "进口", "出口",
              "外贸", "进口替代", "国产替代", "保护主义", "自由贸易区", "全球贸易战",
              "跨境电商", "供应链", "中国制造", "大豆", "汽车", "光伏", "电动车", "中美贸易",
              # 下面几个是实际数据里多出来的贸易类目录, 不在最初的 58 个关键词里
              "产业转移", "新能源", "反倾销", "RCEP", "对华关税", "贸易逆差", "钢铁"],
    "Livelihood": ["经济", "全球经济", "通货膨胀", "汇率", "汇率波动", "美元", "股市波动",
                   "全球股市", "A股", "投资者信心", "资本外流", "国际资本流动", "黄金",
                   "GDP"],
    "Technology": ["芯片", "半导体"],
}
CLUSTER_ORDER = ["Strategic", "Trade", "Livelihood", "Technology"]

# 五类认证用户在表里的写法和图里的英文简称对应
USER_TIER_MAP = {
    "金V": "GoldV", "红V": "RedV", "黄V": "YellowV", "蓝V": "BlueV", "普通用户": "Regular",
}
TIER_ORDER = ["GoldV", "RedV", "YellowV", "BlueV", "Regular"]

# ip 里的省份中文名对应到 shp 文件里的英文名
PROVINCE_EN = {
    "北京": "Beijing Municipality", "天津": "Tianjin Municipality", "上海": "Shanghai Municipality",
    "重庆": "Chongqing Municipality", "河北": "Hebei Province", "山西": "Shanxi Province",
    "内蒙古": "Inner Mongolia Autonomous Region", "辽宁": "Liaoning Province", "吉林": "Jilin Province",
    "黑龙江": "Heilongjiang Province", "江苏": "Jiangsu Province", "浙江": "Zhejiang Province",
    "安徽": "Anhui Province", "福建": "Fujian Province", "江西": "Jiangxi Province",
    "山东": "Shandong Province", "河南": "Henan Province", "湖北": "Hubei Province",
    "湖南": "Hunan Province", "广东": "Guangdong Province", "广西": "Guangxi Zhuang Autonomous Region",
    "海南": "Hainan Province", "四川": "Sichuan Province", "贵州": "Guizhou Province",
    "云南": "Yunnan Province", "西藏": "Tibet Autonomous Region", "陕西": "Shaanxi Province",
    "甘肃": "Gansu province", "青海": "Qinghai Province", "宁夏": "Ningxia Hui Autonomous Region",
    "新疆": "Xinjiang Uygur Autonomous Region", "台湾": "Taiwan Province",
    "香港": "Hong Kong Special Administrative Region", "澳门": "Macao Special Administrative Region",
}
# GWR 只在大陆省份上跑, 港澳台先排除掉
MAINLAND_PROVINCES = [p for p in PROVINCE_EN.values()
                      if "Hong Kong" not in p and "Macao" not in p and "Taiwan" not in p]

CHINA_PROVINCE_SET = set(PROVINCE_EN.keys())

# ip 字段里这些写法其实也是港澳台
PROVINCE_ALIAS = {
    "中国香港": "香港", "香港": "香港",
    "中国澳门": "澳门", "澳门": "澳门",
    "中国台湾": "台湾", "台湾": "台湾",
}
# 这几个值定位不到具体地方, 画图统计的时候都跳过
GENERIC_REGIONS = {"海外", "其他", "中国", "国内", "未知", "国外"}

# ip 里国家名的几个俗称
COUNTRY_ALIAS = {
    "印尼": "印度尼西亚", "刚果金": "刚果民主共和国", "波黑": "波斯尼亚和黑塞哥维那",
}


def normalize_province(ip):
    # 是中国省份就返回短名 (广东/香港这种), 否则返回 None
    if not isinstance(ip, str):
        return None
    ip = ip.strip()
    ip = PROVINCE_ALIAS.get(ip, ip)
    return ip if ip in CHINA_PROVINCE_SET else None


def classify_ip(ip):
    # 返回 (区域, 具体地名), 国内就是 ("china", 省名), 海外 ("overseas", 国名)
    if not isinstance(ip, str):
        return None, None
    ip = ip.strip()
    if not ip or ip in GENERIC_REGIONS:
        return None, None
    prov = normalize_province(ip)
    if prov:
        return "china", prov
    return "overseas", COUNTRY_ALIAS.get(ip, ip)

# GWR 的设定: 自适应 bi-square 核, 用 AICc 挑带宽
GWR_KERNEL = "bisquare"
GWR_TARGET_BW = 27        # 论文里报告的带宽, 留着对照用

# 级联分析的两个规模门槛
CASCADE_MIN_SIZE_STRICT = 5   # size > 5
CASCADE_MIN_SIZE_LOOSE = 2    # size > 2

# 画图统一用的 dpi 和三种情感的颜色
FIG_DPI = 300
COLOR_NEG = "#D62728"   # 负向红
COLOR_NEU = "#7F7F7F"   # 中性灰
COLOR_POS = "#1F77B4"   # 正向蓝
