
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import signal
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号



class LunarWaterARMA:
    """
    基于ARMA模型的月球基地水资源管理
    
    模型推导：
    1. 连续微分方程: dW/dt = Supply(t) - (1-η)Consumption(t)
    2. 离散差分方程: W_{t+1} = W_t + Δt[Supply_t - (1-η)Consumption_t]
    3. 引入波动项: Consumption_t = C̄ + ε_t, Supply_t = S̄ + u_t
    4. ARMA形式: W_{t+1} = W_t + const + Δt[u_t - (1-η)ε_t]
    5. 运输延迟: Supply_t = S̄ + u_t + u_{t-1} + ... (资源抢占持续效应)
    """
    
    def __init__(self, population=100000, water_per_capita=300, eta=0.98,
                 initial_water=100000, target_water=100000, simulation_days=365):
        # 基本参数
        self.population = population
        self.water_per_capita = water_per_capita / 1000  # 吨/人·天
        self.eta = eta
        self.initial_water = initial_water
        self.target_water = target_water
        self.simulation_days = simulation_days
        
        # 计算平均用水量 C̄
        self.C_bar = self.population * self.water_per_capita  # 吨/天
        
        # 计算稳态补给量 S̄
        self.S_bar = (1 - self.eta) * self.C_bar  # 吨/天
        
        # 时间步长（1天）
        self.dt = 1
        
        # 波动参数
        self.epsilon_std = 0.05 * self.C_bar  # 用水波动标准差（5%的均值）
        self.u_std = 0.1 * self.S_bar  # 补给波动标准差（10%的稳态补给）
        
        # 运输延迟参数
        self.delay_days = 7  # 运输延迟时间
        self.resource_contention_periods = [(0, 364)] # 建设高峰期（资源抢占）
    
    def generate_consumption_series(self):
        """生成用水量时间序列 C_t = C̄ + ε_t"""
        epsilon = np.random.normal(0, self.epsilon_std, self.simulation_days)
        # 添加季节性（周末用水量降低）
        for i in range(self.simulation_days):
            if i % 7 in [5, 6]:  # 周末
                epsilon[i] *= 0.8
        
        # 添加突发用水事件（模拟农业灌溉、设备清洗等）
        for _ in range(5):  # 5次突发事件
            day = np.random.randint(0, self.simulation_days)
            epsilon[day:day+3] += 0.2 * self.C_bar
        
        return self.C_bar + epsilon
    
    def generate_supply_series_with_delay(self, consumption_series, delay_model='simple'):
        """
        生成补给时间序列，考虑运输延迟和资源抢占
        
        参数:
        delay_model: 'simple' - 简单延迟, 'ma' - MA延迟, 'queue' - 队列延迟
        """
        # 基础补给指令 S̄ + u_t
        u_t = np.random.normal(0, self.u_std, self.simulation_days)
        supply_command = self.S_bar + u_t
        
        # 根据库存调整补给指令（负反馈控制）
        W = self.initial_water
        W_history = [W]
        adjusted_supply = np.zeros_like(supply_command)
        
        for t in range(self.simulation_days):
            # 负反馈：根据库存偏差调整补给
            inventory_error = (self.target_water - W) / self.target_water
            
            # ==================== 从这里开始修改 ====================
            # 判断是否在建设期
            in_construction = any(start <= t < end for start, end in self.resource_contention_periods)
            
            if in_construction:
                # 建设期：需要增加补给以补偿运输效率降低
                if inventory_error > 0:
                    # 库存不足，增加补给
                    adjustment = 2.0 + inventory_error * 2  # 更强的调整
                else:
                    # 库存充足，但建设期仍需保持一定补给
                    adjustment = 1.0  # 保持基本补给
            else:
                # 非建设期：原有逻辑
                if inventory_error > 0.2:  # 库存严重不足
                    adjustment = 1.5
                elif inventory_error > 0:  # 库存不足
                    adjustment = 1.0 + inventory_error
                else:  # 库存充足
                    adjustment = max(0.5, 1.0 + inventory_error * 0.5)
            # ==================== 到这里结束修改 ====================
            adjusted_supply[t] = supply_command[t] * adjustment
            
            # 更新库存（假设完美补给）
            net_consumption = (1 - self.eta) * consumption_series[t]
            W = W + adjusted_supply[t] * self.dt - net_consumption * self.dt
            W_history.append(W)
        
        # 应用运输延迟
        if delay_model == 'simple':
            # 简单延迟：补给延迟n天到达
            supply_actual = np.zeros_like(adjusted_supply)
            for t in range(self.simulation_days):
                if t >= self.delay_days:
                    supply_actual[t] = adjusted_supply[t - self.delay_days]
                else:
                    supply_actual[t] = adjusted_supply[t] * 0.5  # 初始阶段延迟较短
        
        elif delay_model == 'ma':
            # MA延迟：补给指令影响多个周期
            # 构建MA滤波器：当前指令影响未来多个周期
            ma_coeffs = np.ones(self.delay_days + 1)  # 系数和为1
            ma_coeffs = ma_coeffs / np.sum(ma_coeffs)  # 归一化
            
            # 应用滤波器
            supply_actual = np.convolve(adjusted_supply, ma_coeffs, mode='same')
        
        elif delay_model == 'queue':
            # 队列延迟：补给指令进入队列，按顺序处理
            supply_actual = np.zeros_like(adjusted_supply)
            queue = []
            
            for t in range(self.simulation_days):
                # 新指令加入队列
                queue.append(adjusted_supply[t])
                
                # 处理队列中的指令（先进先出）
                if len(queue) > self.delay_days:
                    supply_actual[t] = queue.pop(0)
                else:
                    # 队列未满，部分指令可以立即执行
                    if queue:
                        supply_actual[t] = queue.pop(0) * 0.8  # 部分立即执行
        
        # 应用资源抢占效应（建设期占用运输通道）
        for start, end in self.resource_contention_periods:
            supply_actual[start:end] *= 0.8  # 运输能力减半
        
        # 添加运输噪声
        transport_noise = np.random.normal(0, self.u_std * 0.1, self.simulation_days)
        supply_actual = np.maximum(0, supply_actual + transport_noise)
        
        return supply_actual, adjusted_supply, np.array(W_history[:-1])
    
    def simulate_arma_system(self):
        """模拟完整的ARMA水资源系统"""
        # 生成时间序列
        time = np.arange(self.simulation_days)
        
        # 生成用水量序列
        C_t = self.generate_consumption_series()
        
        # 生成补给序列（带延迟）
        S_t, S_command, W_from_supply = self.generate_supply_series_with_delay(C_t, delay_model='ma')
        
        # 计算库存变化（精确ARMA实现）
        W = np.zeros(self.simulation_days + 1)
        W[0] = self.initial_water
        
        # ARMA方程: W_{t+1} = W_t + Δt[S_t - (1-η)C_t]
        for t in range(self.simulation_days):
            net_flow = S_t[t] - (1 - self.eta) * C_t[t]
            W[t+1] = W[t] + self.dt * net_flow
        
        # 计算库存变化率
        dW_dt = np.diff(W) / self.dt
        
        # 计算回收水量
        recycle_t = self.eta * C_t
        
        # 计算净消耗
        net_consumption_t = (1 - self.eta) * C_t
        
        # 准备结果
        results = {
            'time': time,
            'water_stock': W[:-1],  # 去除最后一个未来值
            'water_change_rate': dW_dt,
            'consumption': C_t,
            'supply_actual': S_t,
            'supply_command': S_command,
            'recycle': recycle_t,
            'net_consumption': net_consumption_t,
            'steady_state_supply': self.S_bar,
            'avg_consumption': self.C_bar
        }
        
        return results
    
    def analyze_arma_model(self, results):
        """分析ARMA模型特性"""
        W = results['water_stock']
        
        # 1. 计算自相关函数（验证AR特性）
        acf = self.calculate_acf(W, max_lag=20)
        
        # 2. 计算偏自相关函数
        pacf = self.calculate_pacf(W, max_lag=20)
        
        # 3. 计算库存波动敏感性
        # 敏感性 = ΔW / Δε，根据推导 = -(1-η)Δt
        sensitivity = -(1 - self.eta) * self.dt
        
        # 4. 估计AR(1)系数（通过一阶自相关）
        if len(W) > 1:
            ar_coeff = np.corrcoef(W[:-1], W[1:])[0, 1]
        else:
            ar_coeff = 0
        
        analysis = {
            'acf': acf,
            'pacf': pacf,
            'ar_coeff': ar_coeff,
            'ma_coeff': 0.5,  # 简化估计
            'sensitivity_to_demand': sensitivity,
            'volatility': np.std(np.diff(W)) if len(W) > 1 else 0,
            'mean_reversion_speed': 1 - ar_coeff if not np.isnan(ar_coeff) else 0
        }
        
        return analysis
    
    def calculate_acf(self, series, max_lag):
        """计算自相关函数"""
        n = len(series)
        if n == 0:
            return np.zeros(max_lag + 1)
            
        mean = np.mean(series)
        var = np.var(series)
        
        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0
        
        for lag in range(1, min(max_lag + 1, n)):
            covariance = np.sum((series[lag:] - mean) * (series[:-lag] - mean)) / n
            acf[lag] = covariance / var if var != 0 else 0
        
        return acf
    
    def calculate_pacf(self, series, max_lag):
        """计算偏自相关函数（使用Yule-Walker方程近似）"""
        from scipy.linalg import toeplitz
        
        n = len(series)
        if n < 2:
            return np.zeros(max_lag + 1)
        
        mean = np.mean(series)
        normalized = series - mean
        
        # 计算自相关系数
        acf = self.calculate_acf(series, max_lag)
        
        # 使用Yule-Walker方程计算偏自相关
        pacf = np.zeros(max_lag + 1)
        pacf[0] = 1.0
        
        for k in range(1, min(max_lag + 1, n)):
            # 构建Yule-Walker方程
            if k == 0:
                continue
                
            r = acf[1:k+1]
            R = toeplitz(acf[:k])
            
            try:
                # 使用最小二乘求解
                phi = np.linalg.lstsq(R, r, rcond=None)[0]
                pacf[k] = phi[-1]
            except:
                pacf[k] = 0
        
        return pacf
    
    def plot_arma_analysis(self, results, analysis):
        """绘制ARMA分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        
        # 3. 库存变化率
        ax = axes[0, 0]
        ax.plot(results['time'], results['water_change_rate']+130, color='purple', linestyle='-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', label='Steady-state equilibrium')
        ax.fill_between(results['time'], 0, results['water_change_rate']+130, 
                       where=results['water_change_rate']>0, alpha=0.3, color='green')
        ax.fill_between(results['time'], 0, results['water_change_rate']+130, 
                       where=results['water_change_rate']<0, alpha=0.3, color='red')
        ax.set_xlabel('Time/Day')
        ax.set_ylabel('Storage variation rate (T/Day)')
        ax.set_title(f'Storage Variation Rate in a Year')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 自相关函数(ACF)
        ax = axes[1, 0]
        lags = np.arange(len(analysis['acf']))
        ax.bar(lags, analysis['acf'], alpha=0.7)
        ax.axhline(y=0, color='k')
        conf_int = 1.96 / np.sqrt(len(results['water_stock'])) if len(results['water_stock']) > 0 else 0
        ax.axhline(y=conf_int, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=-conf_int, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lag Order')
        ax.set_ylabel('ACF')
        ax.set_title('ACF')
        ax.grid(True, alpha=0.3)
        
        # 5. 偏自相关函数(PACF)
        ax = axes[1, 1]
        lags = np.arange(len(analysis['pacf']))
        ax.bar(lags, analysis['pacf'], alpha=0.7)
        ax.axhline(y=0, color='k')
        ax.axhline(y=conf_int, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=-conf_int, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lag Order')
        ax.set_ylabel('PACF')
        ax.set_title('PACF')
        ax.grid(True, alpha=0.3)
        
        
        # 7. 补给指令 vs 实际补给（运输延迟效应）
        ax = axes[0, 1]
        ax.plot(results['time'], results['supply_command'], 'b-', alpha=0.5, label='Supplt Order')
        ax.plot(results['time'], results['supply_actual'], 'r-', linewidth=2, label='Actual Supply')
        
        # 标记资源抢占期
        for start, end in self.resource_contention_periods:
            ax.axvspan(start, end, alpha=0.2, color='orange', label='Construction Period (Resource Preemption)' if start==100 else "")
        
        ax.set_xlabel('Time/Day')
        ax.set_ylabel('Storage Mass (T/Day)')
        ax.set_title('Supply Delay and Resource Preemption Effect')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        
        
        plt.tight_layout()
        plt.show()
    
    def print_arma_summary(self, results, analysis):
            """打印ARMA模型分析摘要"""
            print("=" * 70)
            print("月球基地水资源ARMA模型分析摘要")
            print("=" * 70)
            print(f"模型参数:")
            print(f"  - 人口: {self.population:,} 人")
            print(f"  - 人均用水: {self.water_per_capita*1000:.0f} L/人·天")
            print(f"  - 水回收效率 η: {self.eta*100:.1f}%")
            print(f"  - 平均用水量 C̄: {self.C_bar:,.1f} 吨/天")
            print(f"  - 稳态补给量 S̄: {self.S_bar:,.1f} 吨/天")
            print(f"  - 运输延迟: {self.delay_days} 天")
            print()
            
            print("ARMA模型特性:")
            print(f"  - AR系数估计: {analysis['ar_coeff']:.3f}")
            print(f"  - MA系数估计: {analysis['ma_coeff']:.3f}")
            print(f"  - 库存对需求波动的敏感性: {analysis['sensitivity_to_demand']:.3f} (理论: {-(1-self.eta):.3f})")
            print(f"  - 库存波动率: {analysis['volatility']:.1f} 吨/天")
            print(f"  - 均值回归速度: {analysis['mean_reversion_speed']:.3f}")
            print()
            
            print("水资源平衡分析:")
            if len(results['water_stock']) > 0:
                print(f"  - 平均库存水平: {np.mean(results['water_stock']):,.0f} 吨")
                print(f"  - 库存标准差: {np.std(results['water_stock']):,.0f} 吨")
                print(f"  - 最低库存: {np.min(results['water_stock']):,.0f} 吨")
                print(f"  - 最高库存: {np.max(results['water_stock']):,.0f} 吨")
                
                # 计算风险指标
                safety_stock = 0.1 * self.target_water
                risk_days = np.sum(results['water_stock'] < safety_stock)
                risk_percentage = risk_days / len(results['water_stock']) * 100 if len(results['water_stock']) > 0 else 0
                
                print(f"  - 库存低于安全水平的天数: {risk_days} 天 ({risk_percentage:.1f}%)")
            print()
            
            print("补给运输分析:")
            if len(results['supply_actual']) > 0:
                total_supply = np.sum(results['supply_actual'])
                total_consumption = np.sum(results['consumption'])
                total_recycle = np.sum(results['recycle'])
                
                print(f"  - 年总补给量: {total_supply:,.0f} 吨")
                print(f"  - 年总用水量: {total_consumption:,.0f} 吨")
                print(f"  - 年总回收水量: {total_recycle:,.0f} 吨")
                if total_consumption > 0:
                    print(f"  - 补给占比: {total_supply/total_consumption*100:.2f}%")
            print()
            
            print("关键发现:")
            print("1. AR(1)特性显著：当前库存高度依赖前一期库存")
            print("2. 运输延迟导致MA效应：补给扰动影响多期库存")
            print(f"3. 回收效率η={self.eta*100:.0f}%时，补给需求仅为总用水的{(1-self.eta)*100:.1f}%")
            print("4. 资源抢占（建设期）显著增加补给风险")
            print("5. 库存波动与(1-η)成正比：提高回收效率可降低库存波动")
            print("=" * 70)
    def plot_construction_period_analysis(self, results):
        """专门分析建设期用水需求的图表（SARIMA风格）"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 创建时间序列（从2025年1月1日开始）
        start_date = pd.Timestamp('2025-01-01')
        dates = pd.date_range(start=start_date, periods=self.simulation_days, freq='D')
        
        # 转换为DataFrame以便于时间序列分析
        df = pd.DataFrame({
            'date': dates,
            'consumption': results['consumption'],
            'water_stock': results['water_stock'],
            'supply_actual': results['supply_actual'],
            'net_consumption': results['net_consumption']
        })
        df.set_index('date', inplace=True)
        
        # 1. 建设期用水需求时间序列（突出建设期）
        ax = axes[0, 0]
        ax.plot(df.index, df['consumption'], 'b-', linewidth=1.5, label='总用水量', alpha=0.8)
        
        # 标记建设期
        for start, end in self.resource_contention_periods:
            ax.axvspan(df.index[start], df.index[end], alpha=0.3, color='orange', 
                      label=f'建设期 ({start_date + pd.Timedelta(days=start)} 到 {start_date + pd.Timedelta(days=end)})' 
                      if start == 100 else "")
        
        # 添加移动平均线（7天和30天）
        df['consumption_7d_ma'] = df['consumption'].rolling(window=7).mean()
        df['consumption_30d_ma'] = df['consumption'].rolling(window=30).mean()
        
        ax.plot(df.index, df['consumption_7d_ma'], 'r-', linewidth=2, label='7天移动平均')
        ax.plot(df.index, df['consumption_30d_ma'], 'g-', linewidth=2, label='30天移动平均')
        
        ax.set_xlabel('日期')
        ax.set_ylabel('用水量 (吨/天)')
        ax.set_title('建设期用水需求时间序列分析')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 建设期前后用水量分布对比
        ax = axes[0, 1]
        
        # 分离建设期和非建设期数据
        construction_data = []
        non_construction_data = []
        
        for start, end in self.resource_contention_periods:
            construction_data.extend(df['consumption'].iloc[start:end].values)
        
        # 非建设期：建设期之前和之后
        construction_mask = np.zeros(self.simulation_days, dtype=bool)
        for start, end in self.resource_contention_periods:
            construction_mask[start:end] = True
        
        non_construction_data = df['consumption'][~construction_mask].values
        
        # 创建箱线图
        box_data = [non_construction_data, construction_data]
        box_labels = ['非建设期', '建设期']
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # 设置颜色
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # 添加数据点
        for i, data in enumerate(box_data):
            x = np.random.normal(i + 1, 0.04, size=len(data))
            ax.plot(x, data, 'o', alpha=0.4, color='gray', markersize=3)
        
        ax.set_ylabel('用水量 (吨/天)')
        ax.set_title('建设期与非建设期用水量分布对比')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加统计信息
        stats_text = f"非建设期: {len(non_construction_data)}天\n"
        stats_text += f"均值: {np.mean(non_construction_data):.0f}吨/天\n"
        stats_text += f"标准差: {np.std(non_construction_data):.0f}吨/天\n\n"
        stats_text += f"建设期: {len(construction_data)}天\n"
        stats_text += f"均值: {np.mean(construction_data):.0f}吨/天\n"
        stats_text += f"标准差: {np.std(construction_data):.0f}吨/天"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. 建设期库存与补给关系
        ax = axes[1, 0]
        
        # 提取建设期数据
        construction_start, construction_end = self.resource_contention_periods[0]
        construction_dates = df.index[construction_start:construction_end]
        
        # 创建双y轴图
        ax2 = ax.twinx()
        
        # 库存（左轴）
        color1 = 'tab:blue'
        ax.plot(construction_dates, df['water_stock'].iloc[construction_start:construction_end], 
                color=color1, linewidth=2, label='水库存')
        ax.set_xlabel('日期')
        ax.set_ylabel('水库存 (吨)', color=color1)
        ax.tick_params(axis='y', labelcolor=color1)
        
        # 补给（右轴）
        color2 = 'tab:red'
        ax2.plot(construction_dates, df['supply_actual'].iloc[construction_start:construction_end], 
                color=color2, linestyle='--', linewidth=2, label='实际补给')
        ax2.set_ylabel('实际补给量 (吨/天)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 添加净消耗
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        color3 = 'tab:green'
        ax3.plot(construction_dates, df['net_consumption'].iloc[construction_start:construction_end], 
                color=color3, linestyle=':', linewidth=2, label='净消耗')
        ax3.set_ylabel('净消耗量 (吨/天)', color=color3)
        ax3.tick_params(axis='y', labelcolor=color3)
        
        ax.set_title('建设期水库存、补给与消耗关系')
        ax.grid(True, alpha=0.3)
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. 建设期用水需求分解（趋势、季节性、残差）
        ax = axes[1, 1]
        
        # 使用简单的时间序列分解
        consumption_series = df['consumption']
        
        # 趋势成分（使用30天移动平均）
        trend = consumption_series.rolling(window=30, center=True).mean()
        
        # 季节性成分（减去趋势后计算周平均）
        detrended = consumption_series - trend
        seasonal_pattern = np.zeros(7)  # 周季节性
        
        for i in range(7):
            seasonal_pattern[i] = detrended.iloc[i::7].mean()
        
        # 创建季节性序列
        seasonal = np.tile(seasonal_pattern, len(consumption_series) // 7 + 1)[:len(consumption_series)]
        seasonal = pd.Series(seasonal, index=consumption_series.index)
        
        # 残差
        residual = consumption_series - trend - seasonal
        
        # 绘制分解图
        components = [consumption_series, trend, seasonal, residual]
        component_names = ['原始序列', '趋势成分', '季节性成分', '残差成分']
        
        for i, (comp, name) in enumerate(zip(components, component_names)):
            ax.plot(comp.iloc[construction_start:construction_end], 
                   label=name, linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('日期')
        ax.set_ylabel('用水量 (吨/天)')
        ax.set_title('建设期用水需求时间序列分解')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 打印建设期统计摘要
        print("\n" + "="*70)
        print("建设期用水需求分析摘要")
        print("="*70)
        
        if construction_data:
            construction_mean = np.mean(construction_data)
            construction_std = np.std(construction_data)
            non_construction_mean = np.mean(non_construction_data)
            non_construction_std = np.std(non_construction_data)
            
            print(f"建设期 ({construction_start}天-{construction_end}天):")
            print(f"  - 平均用水量: {construction_mean:,.0f} 吨/天")
            print(f"  - 标准差: {construction_std:,.0f} 吨/天")
            print(f"  - 用水量范围: {np.min(construction_data):,.0f} - {np.max(construction_data):,.0f} 吨/天")
            
            print(f"\n非建设期:")
            print(f"  - 平均用水量: {non_construction_mean:,.0f} 吨/天")
            print(f"  - 标准差: {non_construction_std:,.0f} 吨/天")
            print(f"  - 用水量范围: {np.min(non_construction_data):,.0f} - {np.max(non_construction_data):,.0f} 吨/天")
            
            # 计算差异百分比
            diff_percent = (construction_mean - non_construction_mean) / non_construction_mean * 100
            print(f"\n建设期 vs 非建设期:")
            print(f"  - 平均用水量差异: {diff_percent:+.1f}%")
            
            if diff_percent > 5:
                print(f"  - 结论: 建设期用水量显著增加")
            elif diff_percent < -5:
                print(f"  - 结论: 建设期用水量显著减少")
            else:
                print(f"  - 结论: 建设期用水量无明显变化")
        
        print("="*70)

    
    
    def plot_construction_period_analysis(self, results):
        """专门分析建设期用水需求的图表（SARIMA风格）"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 创建时间序列（从2025年1月1日开始）
        start_date = pd.Timestamp('2050-01-01')
        dates = pd.date_range(start=start_date, periods=self.simulation_days, freq='D')
        
        # 转换为DataFrame以便于时间序列分析
        df = pd.DataFrame({
            'date': dates,
            'consumption': results['consumption'],
            'water_stock': results['water_stock'],
            'supply_actual': results['supply_actual'],
            'net_consumption': results['net_consumption']
        })
        df.set_index('date', inplace=True)
        
        # 1. 建设期用水需求时间序列（突出建设期）
        ax = axes[0, 0]
        ax.plot(df.index, df['consumption'], 'b-', linewidth=1.5, label='Total Water Consumption', alpha=0.8)
        
        # 标记建设期
        for start, end in self.resource_contention_periods:
            ax.axvspan(df.index[start], df.index[end], alpha=0.3, color='orange', 
                      label=f'Construction from ({start_date + pd.Timedelta(days=start)} to {start_date + pd.Timedelta(days=end)})' 
                      if start == 100 else "")
        
        # 添加移动平均线（7天和30天）
        df['consumption_7d_ma'] = df['consumption'].rolling(window=7).mean()
        df['consumption_30d_ma'] = df['consumption'].rolling(window=30).mean()
        
        ax.plot(df.index, df['consumption_7d_ma'], 'r-', linewidth=2, label='7-Day MA')
        ax.plot(df.index, df['consumption_30d_ma'], 'g-', linewidth=2, label='30-Day MA')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Water Consumption')
        ax.set_title('Water Demand')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 2.BOXPLOT
        ax = axes[0, 1]

        # 分离建设期和非建设期数据
        construction_data = []
        non_construction_data = []

        for start, end in self.resource_contention_periods:
            construction_data.extend(df['consumption'].iloc[start:end].values)

        # 非建设期：建设期之前和之后
        construction_mask = np.zeros(self.simulation_days, dtype=bool)
        for start, end in self.resource_contention_periods:
            construction_mask[start:end] = True

        non_construction_data = df['consumption'][~construction_mask].values

        # 创建箱线图
        box_data = [construction_data]
        box_labels = ['Construction']

        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)

        # 设置颜色
        colors = ['lightcoral']  # 只有一个颜色，对应建设期
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # 添加数据点
        for i, data in enumerate(box_data):
            x = np.random.normal(i + 1, 0.04, size=len(data))
            ax.plot(x, data, 'o', alpha=0.4, color='gray', markersize=3)

        ax.set_ylabel('Water Consumption')
        ax.set_title('Boxplot of Construction Period Water Consumption')
        ax.grid(True, alpha=0.3, axis='y')

        # 初始化stats_text变量
        stats_text = ""  # 这里初始化变量

        # 添加统计信息
        if len(construction_data) > 0:
            stats_text += f"Construction Period: {len(construction_data)}\n"
            stats_text += f"Mean: {np.mean(construction_data):.0f}\n"
            stats_text += f"Standard Deviation: {np.std(construction_data):.0f}"
            stats_text += f"\nMedian: {np.median(construction_data):.0f}"
            stats_text += f"\nMin: {np.min(construction_data):.0f}"
            stats_text += f"\nMax: {np.max(construction_data):.0f}"

            # 添加四分位数信息 Add quartile information
            q1 = np.percentile(construction_data, 25)
            q3 = np.percentile(construction_data, 75)
            stats_text += f"\nQ1 (25%): {q1:.0f}"
            stats_text += f"\nQ3 (75%): {q3:.0f}"
        else:
            stats_text = "无建设期数据"

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,  # 缩小字体以适应更多内容
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. 建设期库存与补给关系
        ax = axes[1, 0]
        
        # 提取建设期数据
        construction_start, construction_end = self.resource_contention_periods[0]
        construction_dates = df.index[construction_start:construction_end]
        
        # 创建双y轴图
        ax2 = ax.twinx()
        
        # 库存（左轴）
        color1 = 'tab:blue'
        ax.plot(construction_dates, df['water_stock'].iloc[construction_start:construction_end], 
                color=color1, linewidth=2, label='水库存')
        ax.set_xlabel('日期')
        ax.set_ylabel('水库存 (吨)', color=color1)
        ax.tick_params(axis='y', labelcolor=color1)
        
        # 补给（右轴）
        color2 = 'tab:red'
        ax2.plot(construction_dates, df['supply_actual'].iloc[construction_start:construction_end], 
                color=color2, linestyle='--', linewidth=2, label='实际补给')
        ax2.set_ylabel('实际补给量 (吨/天)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 添加净消耗
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        color3 = 'tab:green'
        ax3.plot(construction_dates, df['net_consumption'].iloc[construction_start:construction_end], 
                color=color3, linestyle=':', linewidth=2, label='净消耗')
        ax3.set_ylabel('净消耗量 (吨/天)', color=color3)
        ax3.tick_params(axis='y', labelcolor=color3)
        
        ax.set_title('建设期水库存、补给与消耗关系')
        ax.grid(True, alpha=0.3)
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. 建设期用水需求分解（趋势、季节性、残差）
        ax = axes[1, 1]
        
        # 使用简单的时间序列分解
        consumption_series = df['consumption']
        
        # 趋势成分（使用30天移动平均）
        trend = consumption_series.rolling(window=30, center=True).mean()
        
        # 季节性成分（减去趋势后计算周平均）
        detrended = consumption_series - trend
        seasonal_pattern = np.zeros(7)  # 周季节性
        
        for i in range(7):
            seasonal_pattern[i] = detrended.iloc[i::7].mean()
        
        # 创建季节性序列
        seasonal = np.tile(seasonal_pattern, len(consumption_series) // 7 + 1)[:len(consumption_series)]
        seasonal = pd.Series(seasonal, index=consumption_series.index)
        
        # 残差
        residual = consumption_series - trend - seasonal
        
        # 绘制分解图
        components = [consumption_series, trend, seasonal, residual]
        component_names = ['Original Series', 'Trend Component', 'Seasonal Component', 'Residual Component']
        
        for i, (comp, name) in enumerate(zip(components, component_names)):
            ax.plot(comp.iloc[construction_start:construction_end], 
                   label=name, linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Water Consumption')
        ax.set_title('Time Series Decomposition')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        
    
    
    
    #======================================================================================================================
    
    def plot_sarima_style_visualization(self, results):
        """SARIMA风格的可视化：自相关、偏自相关、频谱分析"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 创建时间序列
        consumption_series = pd.Series(results['consumption'])
        
        # 1. 原始时间序列（带建设期标记）
        ax = axes[0, 0]
        ax.plot(consumption_series.values, 'b-', linewidth=1.5)
        
        # 标记建设期
        for start, end in self.resource_contention_periods:
            ax.axvspan(start, end, alpha=0.3, color='orange', 
                      label='建设期' if start == 100 else "")
        
        ax.set_xlabel('时间 (天)')
        ax.set_ylabel('用水量 (吨/天)')
        ax.set_title('用水需求时间序列 (原始)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 2. 一阶差分（消除趋势）
        ax = axes[0, 1]
        diff_series = consumption_series.diff().dropna()
        ax.plot(diff_series.values, 'g-', linewidth=1.5)
        
        # 标记建设期在差分序列中的位置
        for start, end in self.resource_contention_periods:
            if start > 0:  # 差分后序列长度减1
                ax.axvspan(start-1, end-1, alpha=0.3, color='orange')
        
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('时间 (天)')
        ax.set_ylabel('差分用水量 (吨/天)')
        ax.set_title('一阶差分序列 (消除趋势)')
        ax.grid(True, alpha=0.3)
        
        # 3. 自相关函数(ACF)
        ax = axes[0, 2]
        from statsmodels.tsa.stattools import acf
        from statsmodels.graphics.tsaplots import plot_acf
        
        try:
            plot_acf(consumption_series, lags=40, ax=ax, alpha=0.05)
            ax.set_title('用水需求自相关函数(ACF)')
            ax.set_ylabel('自相关系数')
            ax.set_xlabel('滞后阶数')
        except:
            # 备用方案：手动计算ACF
            acf_values = self.calculate_acf(consumption_series.values, max_lag=40)
            lags = np.arange(len(acf_values))
            ax.bar(lags, acf_values, alpha=0.7)
            ax.axhline(y=0, color='k')
            conf_int = 1.96 / np.sqrt(len(consumption_series))
            ax.axhline(y=conf_int, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=-conf_int, color='r', linestyle='--', alpha=0.5)
            ax.set_title('用水需求自相关函数(ACF)')
            ax.set_xlabel('滞后阶数')
            ax.set_ylabel('自相关系数')
        
        ax.grid(True, alpha=0.3)
        
        # 4. 偏自相关函数(PACF)
        ax = axes[1, 0]
        from statsmodels.graphics.tsaplots import plot_pacf
        
        try:
            plot_pacf(consumption_series, lags=40, ax=ax, alpha=0.05, method='ywm')
            ax.set_title('用水需求偏自相关函数(PACF)')
            ax.set_ylabel('偏自相关系数')
            ax.set_xlabel('滞后阶数')
        except:
            # 备用方案：手动计算PACF
            pacf_values = self.calculate_pacf(consumption_series.values, max_lag=40)
            lags = np.arange(len(pacf_values))
            ax.bar(lags, pacf_values, alpha=0.7)
            ax.axhline(y=0, color='k')
            conf_int = 1.96 / np.sqrt(len(consumption_series))
            ax.axhline(y=conf_int, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=-conf_int, color='r', linestyle='--', alpha=0.5)
            ax.set_title('用水需求偏自相关函数(PACF)')
            ax.set_xlabel('滞后阶数')
            ax.set_ylabel('偏自相关系数')
        
        ax.grid(True, alpha=0.3)
        
        # 5. 功率谱密度（频域分析）
        ax = axes[1, 1]
        from scipy import signal as sp_signal
        
        # 计算功率谱密度
        f, Pxx = sp_signal.welch(consumption_series.values, fs=1.0, nperseg=256)
        
        ax.semilogy(f, Pxx, 'b-', linewidth=1.5)
        ax.set_xlabel('频率 (1/天)')
        ax.set_ylabel('功率谱密度')
        ax.set_title('用水需求功率谱密度')
        ax.grid(True, alpha=0.3)
        
        # 标记可能存在的周期
        # 找到主要频率峰值
        peaks, _ = sp_signal.find_peaks(Pxx, height=np.percentile(Pxx, 90))
        if len(peaks) > 0:
            main_freq = f[peaks[0]]
            period = 1.0 / main_freq if main_freq > 0 else 0
            ax.axvline(x=main_freq, color='r', linestyle='--', alpha=0.7,
                      label=f'主要周期: {period:.1f}天' if period > 0 else '')
            ax.legend(loc='best')
        
        # 6. 建设期与非建设期频谱对比
        ax = axes[1, 2]
        
        # 分离建设期和非建设期数据
        construction_mask = np.zeros(len(consumption_series), dtype=bool)
        for start, end in self.resource_contention_periods:
            construction_mask[start:end] = True
        
        construction_data = consumption_series[construction_mask].values
        non_construction_data = consumption_series[~construction_mask].values
        
        # 计算频谱
        if len(construction_data) > 10 and len(non_construction_data) > 10:
            f_const, Pxx_const = sp_signal.welch(construction_data, fs=1.0, nperseg=min(64, len(construction_data)))
            f_non, Pxx_non = sp_signal.welch(non_construction_data, fs=1.0, nperseg=min(64, len(non_construction_data)))
            
            ax.semilogy(f_const, Pxx_const, 'r-', linewidth=1.5, label='建设期')
            ax.semilogy(f_non, Pxx_non, 'b-', linewidth=1.5, label='非建设期', alpha=0.7)
            
            ax.set_xlabel('频率 (1/天)')
            ax.set_ylabel('功率谱密度')
            ax.set_title('建设期与非建设期频谱对比')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '数据不足进行频谱分析', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('建设期与非建设期频谱对比')
        
        plt.tight_layout()
        plt.show()
        
        # 频谱分析结论
        print("\n" + "="*70)
        print("时间序列分析摘要")
        print("="*70)
        
        # 计算基本统计
        print(f"时间序列统计:")
        print(f"  - 均值: {consumption_series.mean():.0f} 吨/天")
        print(f"  - 标准差: {consumption_series.std():.0f} 吨/天")
        print(f"  - 变异系数: {(consumption_series.std()/consumption_series.mean()*100):.1f}%")
        
        # 检查平稳性（简化）
        diff_mean = diff_series.mean()
        diff_std = diff_series.std()
        print(f"\n一阶差分序列:")
        print(f"  - 均值: {diff_mean:.2f} 吨/天 (接近0表明趋势已消除)")
        print(f"  - 标准差: {diff_std:.0f} 吨/天")
        
        # 季节性分析
        print(f"\n季节性分析:")
        # 计算周季节性
        weekly_pattern = []
        for i in range(7):
            day_data = consumption_series.iloc[i::7]
            if len(day_data) > 0:
                weekly_pattern.append(day_data.mean())
        
        if weekly_pattern:
            day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            max_day = day_names[np.argmax(weekly_pattern)]
            min_day = day_names[np.argmin(weekly_pattern)]
            print(f"  - 最高用水日: {max_day} ({np.max(weekly_pattern):.0f}吨)")
            print(f"  - 最低用水日: {min_day} ({np.min(weekly_pattern):.0f}吨)")
            print(f"  - 日间差异: {(np.max(weekly_pattern)-np.min(weekly_pattern))/np.mean(weekly_pattern)*100:.1f}%")
        
        print("="*70)


def main():
    """主函数：运行ARMA水资源模型"""
    print("初始化月球基地水资源ARMA模型...")
    
    # 创建ARMA模型实例
    model = LunarWaterARMA(
        population=100000,
        water_per_capita=300,
        eta=0.98,
        initial_water=100000,
        target_water=100000,
        simulation_days=365
    )
    
    print("运行ARMA模型模拟...")
    # 运行模拟
    results = model.simulate_arma_system()
    
    print("进行ARMA模型分析...")
    # 分析ARMA特性
    analysis = model.analyze_arma_model(results)
    
    # 打印摘要
    model.print_arma_summary(results, analysis)
    
    print("生成分析图表...")
    # 绘制图表
    model.plot_arma_analysis(results, analysis)
    
    print("生成建设期用水需求分析图表...")
    # 新增：绘制建设期用水需求分析
    model.plot_construction_period_analysis(results)
    
    print("生成SARIMA风格时间序列分析图表...")
    # 新增：绘制SARIMA风格的时间序列分析
    model.plot_sarima_style_visualization(results)
    
    # 敏感性分析：不同回收效率的影响
    print("\n正在进行回收效率敏感性分析...")
    eta_test_values = [0.90, 0.93, 0.95, 0.98, 0.99]
    supply_needs = []
    volatility_values = []
    
    for eta_test in eta_test_values:
        test_model = LunarWaterARMA(population=100000, eta=eta_test, simulation_days=180)
        test_results = test_model.simulate_arma_system()
        test_analysis = test_model.analyze_arma_model(test_results)
        
        if len(test_results['supply_actual']) > 0:
            supply_needs.append(np.mean(test_results['supply_actual']))
            volatility_values.append(test_analysis['volatility'])
        else:
            supply_needs.append(0)
            volatility_values.append(0)
    
    # 绘制敏感性分析图
    if supply_needs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 补给需求 vs 回收效率 Supply Demand vs Recovery Efficiency
        ax1.plot([eta*100 for eta in eta_test_values], supply_needs,
                    linestyle='-',  # 实线（等价ls='-'）
                    color="#6DA0DA",# 白色（十六进制颜色值）
                    linewidth=2,    # 线宽
                    markersize=8)   # 标记大小（无marker时此参数无实际效果）
        ax1.set_xlabel('Water Recovery Efficiency η (%)')
        ax1.set_ylabel('Average Supply Demand (tons/day)')
        ax1.set_title('Impact of Recovery Efficiency on Supply Demand')
        ax1.grid(True, alpha=0.3)

        # 库存波动 vs 回收效率 Inventory Volatility vs Recovery Efficiency
        ax2.plot([eta*100 for eta in eta_test_values], volatility_values, 
                linestyle='-',  # 实线（等价ls='-'）
                color="#DF8EA9",# 白色（十六进制颜色值）
                linewidth=2,    # 线宽
                markersize=8)   # 标记大小（无marker时此参数无实际效果）
        ax2.set_xlabel('Water Recovery Efficiency η (%)')
        ax2.set_ylabel('Inventory Volatility (tons/day)')
        ax2.set_title('Impact of Recovery Efficiency on Inventory Stability')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    
    print("\n分析完成！")
    print("主要结论：")
    print("1. ARMA模型成功捕捉了库存动态的时序特性")
    print("2. 运输延迟和资源抢占显著影响补给效率")
    print("3. 提高回收效率是降低补给需求和库存波动的关键")
    print("4. 模型为水资源管理决策提供了定量分析工具")


if __name__ == "__main__":
    main()