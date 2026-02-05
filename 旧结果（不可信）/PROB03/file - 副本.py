import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

T = 365
delta_t = 1.0

eta = 0.98

SUPPLY_MEAN = 2458
CONSUMPTION_MEAN = 30000

drift = delta_t * (
    SUPPLY_MEAN - (1.0 - eta) * CONSUMPTION_MEAN
)

epsilon_consumption = np.zeros(T)


exog = (-(1.0 - eta) * delta_t * epsilon_consumption).reshape(-1, 1)

# ⚠️ 修 2：避免全零序列（统计退化）
W = 1e-6 * np.random.randn(T)

model = SARIMAX(
    W,
    order=(1, 0, 1),
    exog=exog,
    trend='c',
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)
params = results.params.copy()

const_idx = results.param_names.index('const')
params[const_idx] = drift
# ⚠️ 修 3：真正替换物理常数项


print(results.summary())

print("\n=== 手动指定的物理常数项 ===")
print(f"Drift = {drift}")
print("\n=== AR / MA 系数 ===")
print(f"AR(1) = {params[1]}")
print(f"MA(1) = {params[2]}")
