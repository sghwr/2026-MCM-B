import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


data_raw = pd.read_csv(r"D:\USELESS\数据分析学习\数学建模学习\2026美赛\B\references\space_missions.csv", encoding='latin1')

data = data_raw[['Date', 'MissionStatus']]

# 1. 转换 Date 列为年份
data['year'] = pd.to_datetime(data['Date']).dt.year

# 2. 统计每年出现次数
count = data['year'].value_counts().sort_index()  # 按年份排序

# 3. 把 counts 加回 df
# 方法1：用 map
data['counts'] = data['year'].map(count)

year_counts = data['year'].value_counts().sort_index()  # 按年份排序

# 2️⃣ 转成新的 DataFrame
df_counts = year_counts.reset_index()
df_counts.columns = ['year', 'counts']


df_counts = df_counts.iloc[:65]
# 3️⃣ 查看结果
print(df_counts)


# 假设 df_counts 已经有 'year' 和 'counts'
year = df_counts['year'].astype(int)
launches = df_counts['counts'].to_numpy()

# 未来年份
year_future = np.arange(year.min(), 2051)

# 缩放年份
t = (year - year.min()) / 10
t_future = (year_future - year.min()) / 10

# 激进上界 K
K = launches.max() * 2  # 激进情景，但不过分

# Logistic 函数
def logistic(t, r, t0, K):
    return K / (1 + np.exp(-r * (t - t0)))

def logistic_fixed_K(t, r, t0):
    return logistic(t, r, t0, K)

# 拟合
p0 = [0.5, np.median(t)]
params, _ = curve_fit(logistic_fixed_K, t, launches, p0=p0, maxfev=10000)
r, t0 = params

# 预测
pred = logistic(t_future, r, t0, K)

# 画图
plt.figure(figsize=(8,5))
plt.scatter(year, launches, label='Historical')
plt.plot(year_future, pred, color='red', label='Logistic Fit')
plt.axhline(K, linestyle='--', alpha=0.3, label='Capacity K')
plt.xlabel('Year')
plt.ylabel('Launches')
plt.legend()
plt.grid(alpha=0.3)
plt.show()