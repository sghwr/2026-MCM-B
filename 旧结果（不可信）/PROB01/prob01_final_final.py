from deap import base, creator, tools
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==============================
# 问题参数
# ==============================
TMAX = 200          # 为演示缩短长度，实际可设1000
K = 10
P = 3

D_total = 1e8
Cap_rocket = 150        
Cap_lift = 179000       
L_max = 800

Cost_rocket = 500000
Cost_lift = 100000

# ==============================
# DEAP 初始化
# ==============================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 火箭发射次数 (整数)
toolbox.register("attr_R", lambda: random.randint(0, L_max))
# 电梯运输量 (连续)
toolbox.register("attr_E", lambda: random.uniform(0, Cap_lift))

def init_individual():
    """保证初始化总运输量大约 D_total / TMAX"""
    ind = []
    avg_rocket_per_year = D_total / TMAX / 2 / Cap_rocket / K
    avg_lift_per_year = D_total / TMAX / 2 / P

    for _ in range(K*TMAX):
        val = max(1, int(random.gauss(avg_rocket_per_year, avg_rocket_per_year*0.5)))
        ind.append(min(val, L_max))

    for _ in range(P*TMAX):
        val = random.gauss(avg_lift_per_year, avg_lift_per_year*0.5)
        val = max(0, min(val, Cap_lift))
        ind.append(val)

    return creator.Individual(ind)

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ==============================
# Fitness 分配
# ==============================
def assign_fitness(pop, w1=0.5, w2=0.5):
    ZC_list = []
    ZT_list = []

    for ind in pop:
        R = np.array(ind[:K*TMAX]).reshape(K, TMAX)
        E = np.array(ind[K*TMAX:]).reshape(P, TMAX)

        # 总成本
        rocket_tons = np.sum(R) * Cap_rocket
        lift_tons = np.sum(E)
        ZC = Cost_rocket * rocket_tons + Cost_lift * lift_tons
        # 总工期
        transported = 0
        ZT = TMAX
        for t in range(TMAX):
            R_t = np.sum(R[:, t]) * Cap_rocket
            E_t = np.sum(E[:, t])
            transported += R_t + E_t
            if transported >= D_total:
                ZT = t + 1
                break

        # 罚函数: 总运输量不足时增加 fitness
        
        penalty = max(0, (D_total - transported)/D_total) * 1e6

        ind.ZC = ZC
        ind.ZT = ZT
        ind.penalty = penalty

        ZC_list.append(ZC)
        ZT_list.append(ZT)

    # 种群内标准化
    std_C = np.std(ZC_list) + 1e-6
    std_T = np.std(ZT_list) + 1e-6

    for ind in pop:
        fitness_val = w1 * ind.ZC / std_C + w2 * ind.ZT / std_T + ind.penalty
        ind.fitness.values = (fitness_val,)

# ==============================
# 遗传算子
# ==============================
toolbox.register("mate", tools.cxTwoPoint)

def mutate_bound(ind, mu_R=0, sigma_R=100, mu_E=0, sigma_E=5e4, indpb=0.1):
    for i in range(len(ind)):
        if random.random() < indpb:
            if i < K*TMAX:
                ind[i] += int(round(random.gauss(mu_R, sigma_R)))
                ind[i] = max(0, min(L_max, ind[i]))
            else:
                ind[i] += random.gauss(mu_E, sigma_E)
                ind[i] = max(0, min(Cap_lift, ind[i]))
    return ind,

toolbox.register("mutate", mutate_bound)
toolbox.register("select", tools.selTournament, tournsize=3)

# ==============================
# 主程序
# ==============================
def main():
    pop = toolbox.population(n=100)
    NGEN = 500
    CXPB = 0.7
    MUTPB = 0.2

    assign_fitness(pop)

    for gen in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # 变异
        for ind in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(ind)
                del ind.fitness.values

        assign_fitness(offspring)
        pop[:] = offspring

        best = tools.selBest(pop, 1)[0]
        total_transported = np.sum(best[:K*TMAX])*Cap_rocket + np.sum(best[K*TMAX:])
        print(f"Gen {gen}: Z={best.fitness.values[0]:.3e}, "
              f"ZC={best.ZC:.3e}, ZT={best.ZT}, TotalTransport={total_transported:.1e}")

    return tools.selBest(pop, 1)[0]

# ==============================
# 运行 GA
# ==============================
best_ind = main()

R = np.array(best_ind[:K*TMAX]).reshape(K, TMAX)
E = np.array(best_ind[K*TMAX:]).reshape(P, TMAX)

ROCKET = pd.DataFrame(R)
ELEVATOR = pd.DataFrame(E)
ROCKET.to_excel((r"D:\USELESS\数据分析学习\数学建模学习\2026美赛\B\PROB01\ROCKET1.xlsx"))
ELEVATOR.to_excel((r"D:\USELESS\数据分析学习\数学建模学习\2026美赛\B\PROB01\ELEVATOR1.xlsx"))
print("最终总运输量 =", np.sum(R)*Cap_rocket + np.sum(E))
print('火箭运输量 = ', np.sum(R)*Cap_rocket)
print('太空电梯运输量 = ', np.sum(E))



K, TMAX = R.shape
P = E.shape[0]

# 计算每年总运输量
yearly_transport = np.array([R[:, t].sum()*Cap_rocket + E[:, t].sum() for t in range(TMAX)])

# 按年份运输量从大到小排序
sorted_idx = np.argsort(-yearly_transport)

# 初始化新的矩阵
R_new = np.zeros_like(R)
E_new = np.zeros_like(E)

transported = 0
ZT_new = 0

for t_new, t_old in enumerate(sorted_idx):
    # 尽量保持原有分配比例，但不超过上限
    R_year = np.minimum(R[:, t_old], L_max)
    E_year = np.minimum(E[:, t_old], Cap_lift)

    # 计算这一年的运输量
    transported += R_year.sum()*Cap_rocket + E_year.sum()
    
    # 如果运输量超过 D_total，则按比例缩减这一年的运输量
    if transported > D_total:
        excess = transported - D_total
        # 按比例缩减火箭和电梯
        total_year = R_year.sum()*Cap_rocket + E_year.sum()
        if total_year > 0:
            ratio = (total_year - excess) / total_year
            R_year = np.floor(R_year * ratio)  # 保留整数
            E_year = E_year * ratio
        transported = D_total
        ZT_new = t_new + 1
        R_new[:, t_new] = R_year
        E_new[:, t_new] = E_year
        break
    
    R_new[:, t_new] = R_year
    E_new[:, t_new] = E_year
    ZT_new = t_new + 1

print(f"压缩后最早完成 D_total 的年份 ZT_new = {ZT_new}")
print(f"压缩后总运输量 = {transported}")

rocket_tons_new = np.sum(R_new[:, :ZT_new]) * Cap_rocket
lift_tons_new = np.sum(E_new[:, :ZT_new])
ZC_new = Cost_rocket * rocket_tons_new + Cost_lift * lift_tons_new

print(f"压缩后总成本 = {ZC_new:.3e}")
print(f"火箭总运输量 = {R_new.sum()*Cap_rocket}")
print(f"电梯总运输量 = {E_new.sum()}")


R = np.array(best_ind[:K*TMAX]).reshape(K, TMAX)
E = np.array(best_ind[K*TMAX:]).reshape(P, TMAX)

ROCKET = pd.DataFrame(R)
ELEVATOR = pd.DataFrame(E)

ROCKET.to_excel((r"D:\USELESS\数据分析学习\数学建模学习\2026美赛\B\PROB01\ROCKET_final.xlsx"))
ELEVATOR.to_excel((r"D:\USELESS\数据分析学习\数学建模学习\2026美赛\B\PROB01\_final.xlsx"))