import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False    # 负号显示正常

data_raw = pd.read_csv(r"D:\USELESS\数据分析学习\数学建模学习\2026美赛\B\references\spacex_launch_data.csv")

print(data_raw.head())

data_selected = data_raw[['Date', 'Payload Mass (kg)']]


data_selected[pd.to_numeric(data_selected['Payload Mass (kg)'], errors='coerce').notna()]
data_selected = data_selected.drop(index=[0, 1, 32, 46, 48])



#print(data_selected)


#---------------------
#利用清洗后数据进行预测
#---------------------



hist_val = data_selected[['Date', 'Payload Mass (kg)']]
#new_row = {'Date': '2025-01-01', 'Payload Mass (kg)': 150000}
#hist_val = pd.concat([hist_val, pd.DataFrame([new_row])], ignore_index=True)
print(hist_val)





df = hist_val.copy()

# 转换 Date 到年份
df['Year'] = pd.to_datetime(df['Date']).dt.year

# 清理 Payload 列（去掉逗号并转整数）
df['Payload Mass (kg)'] = df['Payload Mass (kg)'].astype(str).str.replace(',', '', regex=False).astype(int)

# 历史年份和载荷
t_hist = df['Year'].values
y_hist = df['Payload Mass (kg)'].values

# -------------------------
# 2️⃣ 定义 logistic 函数
def logistic(t, r, t0, K):
    return K / (1 + np.exp(-r*(t - t0)))

# 给定合理最大值 K（比如历史最大值的 5 倍）
K_guess = y_hist.max() * 5

# 拟合 r 和 t0
popt, _ = curve_fit(lambda t, r, t0: logistic(t, r, t0, K=K_guess),
                    t_hist, y_hist,
                    p0=[0.3, t_hist.mean()])

r_fit, t0_fit = popt

# -------------------------
# 3️⃣ 生成 2012–2150 年预测
years_all = np.arange(t_hist.min(), 2151)
payload_all = logistic(years_all, r_fit, t0_fit, K=K_guess)

# -------------------------
# 4️⃣ 可视化
plt.figure(figsize=(12,6))
plt.scatter(t_hist, y_hist, color='blue', label='历史数据')
plt.plot(years_all, payload_all, color='green', linestyle='--', label='预测 S 型曲线')
plt.xlabel('Year')
plt.ylabel('Payload Mass (kg)')
plt.title('SpaceX Payload Growth Projection (Ignoring 2050 Point)')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# 5️⃣ 生成预测表格
projection_df = pd.DataFrame({
    'Year': years_all,
    'Projected Payload (kg)': payload_all.astype(int)
})

# 查看前后几行
print(projection_df.head())
print(projection_df.tail())