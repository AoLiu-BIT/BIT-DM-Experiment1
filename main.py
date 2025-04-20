import pandas as pd
import pyarrow.parquet as pq
import time
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from matplotlib import rcParams
import json

rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

dataset = '30G'

time1 = time.time()

# 1. 数据加载

files = glob.glob('/data/liuao/' + dataset + '_data/part-*.parquet')
dfs = []
for file in tqdm(files, desc="读取Parquet文件"):
    dfs.append(pq.read_table(file).to_pandas())
df = pd.concat(dfs, ignore_index=True)

time2 = time.time()
print("\n1. 【数据加载】")
print("【数据加载】耗时：{:.2f}s".format(time2 - time1))


# 2. 数据预处理

time3 = time.time()
print("\n2. 【数据预处理】")
print("处理日期格式ing")
df['registration_date'] = pd.to_datetime(df['registration_date'], utc=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
current_date = datetime(2025, 4, 10, tzinfo=datetime.now().astimezone().tzinfo)

print("\n性别数量统计：（未指定性别对应行不作删除）")
print(df['gender'].value_counts().to_string())

# —— 异常值检测函数 —— #
def check_outliers(col_name, series):
    """返回异常值的布尔Series"""
    if col_name == 'id':
        return (series <= 0) | (series != series.astype(int))
    elif col_name == 'age':
        return (series < 10) | (series > 100)
    elif col_name == 'income':
        return series < 0
    elif col_name == 'gender':
        return ~series.isin(['男', '女', '其他', '未指定'])
    elif col_name == 'email':
        # 简单的“xxx@yyy”格式校验
        return ~series.str.contains(r'^[^@]+@[^@]+\.[^@]+$', na=False)
    elif col_name == 'phone_number':
        # 至少 8 位数字，而且只允许数字、连字符或空格
        return ~series.str.match(r'^[\d\-\s]{8,}$', na=False)
    elif col_name == 'timestamp':
        # 非法时间或未来/过早的时间
        return series.isna() | (series > current_date) | (series < pd.Timestamp('2010-01-01', tz=current_date.tzinfo))
    elif col_name == 'registration_date':
        # 非法时间或未来的注册日期
        return series.isna() | (series > current_date)
    elif col_name == 'user_name':
        return series.str.strip().eq('')  # 不能为空
    elif col_name == 'chinese_name':
        return series.str.strip().eq('')  # 不能为空
    elif col_name == 'purchase_history':
        # 简单 JSON 结构校验
        def valid_json(x):
            try:
                obj = json.loads(x)
                return isinstance(obj, dict)
            except:
                return False
        return ~series.fillna('').apply(valid_json)
    elif col_name == 'is_active':
        return ~series.isin([True, False])
    # 其他列默认不检测
    return pd.Series(False, index=series.index)

# 先检查缺失值
print("检查数据质量ing")
missing = df.isnull().sum()
print(missing)
print(f"总记录数：{len(df):,}")

# 删除关键字段缺失
print("删除关键字段缺失记录ing")
df_clean = df.dropna(subset=['id', 'timestamp', 'registration_date'])
print(f"删除后记录数：{len(df_clean):,}")

# 对每列打标签
print("检测异常值ing")
outlier_flags = pd.DataFrame(index=df_clean.index)
for col in tqdm([
        'id', 'age', 'income', 'gender', 'email',
        'phone_number', 'timestamp', 'registration_date',
        'user_name', 'chinese_name', 'purchase_history', 'credit_score'
    ], desc="各列异常检测"):
    if col in df_clean.columns:
        outlier_flags[col] = check_outliers(col, df_clean[col])
    else:
        outlier_flags[col] = False

# 汇总并输出
total_outliers = outlier_flags.any(axis=1).sum()
print(f"检测到 {total_outliers} 条包含 ≥1 项异常值的记录")
print("每列异常值数量：")
print(outlier_flags.sum().to_dict())

# 处理异常
print("剔除包含异常值的记录ing")
df_clean = df_clean[~outlier_flags.any(axis=1)]
print(f"剔除后剩余记录：{len(df_clean):,}")

# 信用评分再做一次离群截断
print("处理信用评分异常值ing")
q1 = df_clean['credit_score'].quantile(0.25)
q3 = df_clean['credit_score'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df_clean['credit_score'] = df_clean['credit_score'].clip(lower_bound, upper_bound)

time4 = time.time()
print("【数据预处理】耗时：{:.2f}s".format(time4 - time3))

# 3. 探索性分析与可视化

print("\n3. 【数据可视化】")
print("生成可视化图表ing")
plt.figure(figsize=(22, 6))

# 可视化1： 国家分布饼图（前10）
plt.subplot(1, 3, 1)

# 处理国家数据（显示前10大国家）
country_counts = df_clean['country'].value_counts()
n_top = 10
top_countries = country_counts.head(n_top).copy()
top_countries['其他'] = country_counts[n_top:].sum()

# 生成颜色列表（使用色环算法）
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("mycmap", ['#4B0082','#0000FF','#00FF00','#FFFF00','#FF7F00','#FF0000'])
colors = [cmap(i/n_top) for i in range(n_top)] + ['#CCCCCC']

country_name_mapping = {
    '中国': 'China',
    '巴西': 'Brazil',
    '日本': 'Japan',
    '澳大利亚': 'Australia',
    '美国': 'United States',
    '印度': 'India',
    '俄罗斯': 'Russia',
    '法国': 'France',
    '德国': 'Germany',
    '英国': 'Britain',
    '其他': 'Others'
}

translated_labels = [country_name_mapping.get(x, x) for x in top_countries.index]

# 绘制饼图
patches, texts, autotexts = plt.pie(
    top_countries,
    labels=translated_labels,
    autopct=lambda p: f'{p:.1f}% ({int(p/100*top_countries.sum()):,})',
    colors=colors,
    startangle=90,
    pctdistance=0.8,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    textprops={'fontsize': 8}
)

# 优化标签显示
plt.title(f'Country Distribution Top {n_top}', fontsize=12, pad=20)

# 可视化2：注册年份分布
plt.subplot(1, 3, 2)
df_clean['reg_year'] = df_clean['registration_date'].dt.year
year_counts = df_clean['reg_year'].value_counts().sort_index()
year_counts.plot(kind='bar', color='skyblue')
plt.title('Register Year Distribution')
plt.xlabel('year')

# 可视化3：信用评分分布
plt.subplot(1, 3, 3)
sns.histplot(df_clean['credit_score'], bins=30, kde=True, color='green')
plt.title('Credit Score Distribution')

plt.tight_layout(pad=3.0)
plt.savefig('./fig/'+dataset+'/fig1.png', dpi=300, bbox_inches='tight')
plt.close()

time5 = time.time()
print("【数据可视化】耗时：{:.2f}s".format(time5 - time4))

# 4. 识别潜在高价值用户
print("\n4. 【识别潜在高价值用户】")
print("开始识别高价值用户ing")

# 只使用收入和信用分
value_cols = ['income', 'credit_score']
df_value = df_clean[value_cols].copy()

# 填充缺失值
df_value = df_value.fillna(df_value.median(numeric_only=True))

# 标准化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_value)

# 聚类
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
df_clean['cluster'] = kmeans.fit_predict(df_scaled)

# 聚类结果均值
cluster_summary = df_clean.groupby('cluster')[['income', 'credit_score']].mean()
cluster_summary['用户数量'] = df_clean['cluster'].value_counts().sort_index()
print("各用户簇的特征均值：")
print(cluster_summary)

# 标准化后评估综合得分
scaler = StandardScaler()
scaled_summary = scaler.fit_transform(cluster_summary[['income', 'credit_score']])
cluster_scores = scaled_summary.mean(axis=1)
high_value_cluster = cluster_scores.argmax()

# 打标签
df_clean['is_high_value'] = (df_clean['cluster'] == high_value_cluster).astype(int)
num_high_value = df_clean['is_high_value'].sum()
print(f"识别出高价值用户数：{num_high_value:,}，占比：{num_high_value / len(df_clean):.2%}")

time6 = time.time()
print("【识别潜在高价值用户】耗时：{:.2f}s".format(time6 - time5))

# 5. 高价值用户数据可视化
print("\n5. 【高价值用户数据可视化】")
print("生成可视化图表ing")

plt.figure(figsize=(10, 4))

# 收入分布
plt.subplot(1, 2, 1)
sns.kdeplot(df_clean[df_clean['is_high_value'] == 1]['income'], label='High Value', fill=True)
sns.kdeplot(df_clean[df_clean['is_high_value'] == 0]['income'], label='Others', fill=True)
plt.title('Income Distribution Comparison')
plt.legend()

# 信用评分分布
plt.subplot(1, 2, 2)
sns.kdeplot(df_clean[df_clean['is_high_value'] == 1]['credit_score'], label='High Value', fill=True)
sns.kdeplot(df_clean[df_clean['is_high_value'] == 0]['credit_score'], label='Others', fill=True)
plt.title('Credit Score Distribution Comparison')
plt.legend()

plt.tight_layout()

plt.savefig('./fig/'+dataset+'/fig2.png', dpi=300, bbox_inches='tight')
plt.close()

time7 = time.time()
print("【高价值用户数据可视化】耗时：{:.2f}s".format(time7 - time6))
