import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =========================
# 1. 时间参数
# =========================

T = 365              # 时间长度（例如 120 天 / 月）【可自行修改】
delta_t = 1.0        # Δt（时间步长，天或月）【可自行修改】

# =========================
# 2. 系统物理参数（❗不估计❗）
# =========================

eta = 0.98           # 水回收闭合度 η 【可自行修改】

# ---------- 请在此处填写 ----------
SUPPLY_MEAN = 30000       # ← 这里填入 \overline{Supply}
CONSUMPTION_MEAN = 30000   # ← 这里填入 \overline{Consumption}
# ---------------------------------

# 稳态确定性漂移项（常数项）
# Δt * ( \bar{Supply} - (1-η)\bar{Consumption} )
drift = delta_t * (
    SUPPLY_MEAN - (1.0 - eta) * CONSUMPTION_MEAN
)

# =========================
# 3. 构造外生扰动（用水波动 ε_t）
# =========================

# 这里假设你已有用水波动数据
# 若没有，可用占位数据，之后替换
epsilon_consumption = np.zeros(T)

# 外生变量进入模型时的系数是 -(1-η)*Δt
exog = -(1.0 - eta) * delta_t * epsilon_consumption

# =========================
# 4. 构造水库存时间序列 W_t
# =========================

# 示例：这里用占位数据
# 实际使用时，请替换为你们仿真或观测得到的 W_t
W = np.zeros(T)

# =========================
# 5. 建立 ARMA(1,1) + 外生变量 模型
# =========================

model = SARIMAX(
    W,
    order=(1, 0, 1),        # ARMA(1,1)
    exog=exog,
    trend='c',              # 常数项
    enforce_stationarity=False,
    enforce_invertibility=False
)

# =========================
# 6. 拟合模型
# =========================

results = model.fit(disp=False)

# =========================
# 7. 手动替换常数项（❗关键步骤❗）
# =========================

# statsmodels 默认会估计常数项
# 我们用物理模型中的 drift 强制替换

params = results.params.copy()
params['const'] = drift

# =========================
# 8. 输出结果
# =========================

print(results.summary())

print("\n=== 手动指定的物理常数项 ===")
print(f"Drift (Δt*(Supply_mean - (1-η)*Consumption_mean)) = {drift}")

print("\n=== AR / MA 系数 ===")
print(f"AR(1) = {params.get('ar.L1')}")
print(f"MA(1) = {params.get('ma.L1')}")
