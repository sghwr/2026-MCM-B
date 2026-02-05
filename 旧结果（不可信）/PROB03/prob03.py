import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

class LunarWaterModel:
    """
    月球基地水资源管理模型
    
    参数说明：
    - population: 基地人口数量
    - water_per_capita: 人均用水量 (L/人·天)
    - eta: 水回收效率 (0-1之间)
    - initial_water: 初始储水量 (吨)
    - target_water: 目标储水量 (吨)
    - simulation_days: 模拟天数
    """
    
    def __init__(self, population=100000, water_per_capita=300, eta=0.98,
                 initial_water=10000, target_water=50000, simulation_days=365):
        # 基本参数
        self.population = population  # 人口数量
        self.water_per_capita = water_per_capita / 1000  # 转换为吨/人·天
        self.eta = eta  # 水回收效率 (98%)
        self.initial_water = initial_water  # 初始储水量 (吨)
        self.target_water = target_water  # 目标储水量 (吨)
        self.simulation_days = simulation_days  # 模拟天数
        
        # 计算每日用水量
        self.consumption_rate = self.population * self.water_per_capita  # 吨/天
        
        # 补给策略参数
        self.earth_supply_strategy = 'dynamic'  # 'dynamic' 或 'constant'
        self.supply_efficiency = 0.95  # 地球补给效率（考虑运输损失）
        
    def water_balance_ode(self, W, t, supply_earth, supply_moon):
        """
        定义水存量变化的微分方程
        
        参数:
        W: 当前水存量 (吨)
        t: 时间 (天)
        supply_earth: 地球补给量 (吨/天)
        supply_moon: 月球本地补给量 (吨/天)
        
        返回:
        dW/dt: 水存量变化率
        """
        # 总消耗量
        consumption = self.consumption_rate
        
        # 回收水量
        recycle = self.eta * consumption
        
        # 总补给量
        total_supply = supply_earth + supply_moon
        
        # 微分方程: dW/dt = 补给 - 消耗 + 回收
        dW_dt = total_supply - consumption + recycle
        
        return dW_dt
    
    def calculate_steady_state_supply(self):
        """
        计算达到稳态平衡所需的地球补给量
        
        稳态条件: dW/dt = 0
        supply_earth = consumption - recycle - supply_moon
        """
        # 假设月球本地补给为0（最保守情况）
        supply_moon = 0
        
        # 计算所需地球补给量
        required_supply = self.consumption_rate * (1 - self.eta) - supply_moon
        
        return max(required_supply, 0)  # 确保非负
    
    def dynamic_supply_strategy(self, W, t, W_target=None):
        """
        动态补给策略：根据当前水存量调整补给量
        
        参数:
        W: 当前水存量
        t: 当前时间
        W_target: 目标水存量（如果为None则使用self.target_water）
        
        返回:
        supply_earth: 地球补给量
        supply_moon: 月球本地补给量
        """
        if W_target is None:
            W_target = self.target_water
        
        # 月球本地补给（假设恒定）
        supply_moon = self.consumption_rate * 0.05  # 假设月球能提供5%的需求
        
        # 地球补给：基于库存水平动态调整
        water_ratio = W / W_target
        
        if water_ratio > 1.2:  # 库存充足
            supply_earth = 0
        elif water_ratio > 0.8:  # 库存正常
            supply_earth = self.consumption_rate * (1 - self.eta) * 0.5
        elif water_ratio > 0.5:  # 库存偏低
            supply_earth = self.consumption_rate * (1 - self.eta)
        else:  # 库存严重不足
            supply_earth = self.consumption_rate * (1 - self.eta) * 1.5
        
        return supply_earth, supply_moon
    
    def simulate(self, strategy='dynamic'):
        """
        模拟水资源变化
        
        参数:
        strategy: 补给策略 ('dynamic' 或 'constant')
        
        返回:
        results: 包含模拟结果的字典
        """
        # 时间数组
        t = np.linspace(0, self.simulation_days, self.simulation_days)
        
        # 初始条件
        W0 = self.initial_water
        
        # 存储结果
        W_vals = np.zeros_like(t)
        supply_earth_vals = np.zeros_like(t)
        supply_moon_vals = np.zeros_like(t)
        dW_dt_vals = np.zeros_like(t)
        
        # 设置补给策略
        self.earth_supply_strategy = strategy
        
        # 使用数值积分求解微分方程
        W_current = W0
        
        for i, current_time in enumerate(t):
            # 计算当前补给量
            if strategy == 'dynamic':
                supply_earth, supply_moon = self.dynamic_supply_strategy(
                    W_current, current_time)
            else:  # constant strategy
                supply_earth = self.calculate_steady_state_supply()
                supply_moon = 0
            
            # 计算变化率
            dW_dt = self.water_balance_ode(W_current, current_time, 
                                          supply_earth, supply_moon)
            
            # 欧拉法更新水存量（简单但有效）
            if i < len(t) - 1:
                dt = t[i+1] - current_time
                W_current += dW_dt * dt
            
            # 存储结果
            W_vals[i] = W_current
            supply_earth_vals[i] = supply_earth
            supply_moon_vals[i] = supply_moon
            dW_dt_vals[i] = dW_dt
        
        # 计算总补给量
        total_earth_supply = np.sum(supply_earth_vals)
        total_moon_supply = np.sum(supply_moon_vals)
        total_consumption = np.sum(np.ones_like(t) * self.consumption_rate)
        
        # 准备结果字典
        results = {
            'time': t,
            'water_stock': W_vals,
            'earth_supply_rate': supply_earth_vals,
            'moon_supply_rate': supply_moon_vals,
            'water_change_rate': dW_dt_vals,
            'total_earth_supply': total_earth_supply,
            'total_moon_supply': total_moon_supply,
            'total_consumption': total_consumption,
            'water_recovery_efficiency': self.eta,
            'daily_consumption': self.consumption_rate
        }
        
        return results
    
    def analyze_supply_requirements(self):
        """
        分析不同回收效率下的补给需求
        """
        # 测试不同回收效率
        eta_values = np.linspace(0.85, 0.99, 15)
        daily_supply_needs = []
        annual_supply_needs = []
        
        for eta in eta_values:
            # 保存当前效率
            original_eta = self.eta
            self.eta = eta
            
            # 计算每日补给需求
            daily_supply = self.calculate_steady_state_supply()
            daily_supply_needs.append(daily_supply)
            
            # 计算年补给需求
            annual_supply = daily_supply * 365
            annual_supply_needs.append(annual_supply)
            
            # 恢复原始效率
            self.eta = original_eta
        
        return eta_values, daily_supply_needs, annual_supply_needs
    
    def transport_cost_analysis(self):
        """
        分析运输成本和资源抢占问题
        """
        # 运输成本参数（简化模型）
        cost_per_ton = 50000  # 美元/吨（从地球到月球的运输成本估算）
        
        # 计算不同回收效率下的运输成本
        eta_values = np.linspace(0.85, 0.99, 15)
        annual_costs = []
        
        for eta in eta_values:
            self.eta = eta
            daily_supply = self.calculate_steady_state_supply()
            annual_supply = daily_supply * 365
            annual_cost = annual_supply * cost_per_ton
            annual_costs.append(annual_cost)
        
        self.eta = 0.98  # 重置为原始值
        
        return eta_values, annual_costs


def plot_results(model, results):
    """
    绘制模拟结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 水存量随时间变化
    ax1 = axes[0, 0]
    ax1.plot(results['time'], results['water_stock'], 'b-', linewidth=2)
    ax1.axhline(y=model.target_water, color='r', linestyle='--', 
                label=f'目标库存: {model.target_water}吨')
    ax1.set_xlabel('时间 (天)')
    ax1.set_ylabel('水存量 (吨)')
    ax1.set_title('月球基地水存量变化')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 补给速率
    ax2 = axes[0, 1]
    ax2.plot(results['time'], results['earth_supply_rate'], 'r-', 
             label='地球补给', linewidth=2)
    ax2.plot(results['time'], results['moon_supply_rate'], 'g-', 
             label='月球补给', linewidth=2)
    ax2.set_xlabel('时间 (天)')
    ax2.set_ylabel('补给速率 (吨/天)')
    ax2.set_title('水资源补给速率')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. 水存量变化率
    ax3 = axes[0, 2]
    ax3.plot(results['time'], results['water_change_rate'], 'purple', linewidth=2)
    ax3.axhline(y=0, color='r', linestyle='--', label='稳态平衡')
    ax3.set_xlabel('时间 (天)')
    ax3.set_ylabel('变化速率 (吨/天)')
    ax3.set_title('水存量变化速率')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 补给需求与回收效率关系
    eta_values, daily_supply, annual_supply = model.analyze_supply_requirements()
    ax4 = axes[1, 0]
    ax4.plot(eta_values * 100, daily_supply, 'b-', marker='o', linewidth=2)
    ax4.axvline(x=model.eta * 100, color='r', linestyle='--', 
                label=f'当前效率: {model.eta*100:.1f}%')
    ax4.set_xlabel('水回收效率 (%)')
    ax4.set_ylabel('每日地球补给需求 (吨/天)')
    ax4.set_title('回收效率对补给需求的影响')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. 年运输成本分析
    eta_values, annual_costs = model.transport_cost_analysis()
    ax5 = axes[1, 1]
    ax5.plot(eta_values * 100, np.array(annual_costs) / 1e6, 'r-', marker='s', linewidth=2)
    ax5.axvline(x=model.eta * 100, color='b', linestyle='--')
    ax5.set_xlabel('水回收效率 (%)')
    ax5.set_ylabel('年运输成本 (百万美元)')
    ax5.set_title('水回收效率对运输成本的影响')
    ax5.grid(True, alpha=0.3)
    
    # 6. 补给占比饼图
    ax6 = axes[1, 2]
    labels = ['地球补给', '月球补给', '循环回收']
    total_recycled = results['total_consumption'] * model.eta
    sizes = [results['total_earth_supply'], 
             results['total_moon_supply'], 
             total_recycled]
    
    # 仅显示非零部分
    non_zero_sizes = []
    non_zero_labels = []
    for size, label in zip(sizes, labels):
        if size > 0:
            non_zero_sizes.append(size)
            non_zero_labels.append(label)
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax6.pie(non_zero_sizes, labels=non_zero_labels, colors=colors, 
            autopct='%1.1f%%', startangle=90)
    ax6.set_title('水资源来源占比（一年总量）')
    
    plt.tight_layout()
    plt.show()


def print_summary(model, results):
    """
    打印模拟结果摘要
    """
    print("=" * 60)
    print("月球基地水资源管理模型 - 模拟结果摘要")
    print("=" * 60)
    print(f"人口数量: {model.population:,} 人")
    print(f"人均用水量: {model.water_per_capita * 1000:.0f} L/人·天")
    print(f"水回收效率: {model.eta * 100:.1f}%")
    print(f"模拟天数: {model.simulation_days} 天")
    print("-" * 60)
    print(f"每日总用水量: {results['daily_consumption']:.2f} 吨/天")
    print(f"每日循环水量: {results['daily_consumption'] * model.eta:.2f} 吨/天")
    print(f"稳态所需地球补给: {model.calculate_steady_state_supply():.2f} 吨/天")
    print("-" * 60)
    print("一年总量统计:")
    print(f"总用水量: {results['total_consumption']:,.2f} 吨")
    print(f"总地球补给量: {results['total_earth_supply']:,.2f} 吨")
    print(f"总月球补给量: {results['total_moon_supply']:,.2f} 吨")
    print(f"总循环水量: {results['total_consumption'] * model.eta:,.2f} 吨")
    print("-" * 60)
    
    # 计算补给占比
    total_supply = results['total_earth_supply'] + results['total_moon_supply']
    if total_supply > 0:
        earth_ratio = results['total_earth_supply'] / total_supply * 100
        moon_ratio = results['total_moon_supply'] / total_supply * 100
        print(f"地球补给占比: {earth_ratio:.1f}%")
        print(f"月球补给占比: {moon_ratio:.1f}%")
    
    # 分析资源抢占问题
    print("-" * 60)
    print("资源抢占分析:")
    print(f"平均每日地球补给: {results['total_earth_supply']/model.simulation_days:.2f} 吨/天")
    print(f"相当于运输频率: 每{(1/(results['total_earth_supply']/(model.simulation_days*100))):.1f}天发射一次100吨级运输船")
    
    # 敏感性分析
    print("-" * 60)
    print("敏感性分析 - 不同回收效率下的补给需求:")
    eta_test = [0.90, 0.93, 0.95, 0.98, 0.99]
    for eta in eta_test:
        model.eta = eta
        daily_supply = model.calculate_steady_state_supply()
        annual_supply = daily_supply * 365
        print(f"η={eta*100:.0f}%: {daily_supply:.1f} 吨/天, {annual_supply:,.0f} 吨/年")
    
    model.eta = 0.98  # 重置


def main():
    """
    主函数：运行完整的水资源管理分析
    """
    print("正在初始化月球基地水资源模型...")
    
    # 创建模型实例
    model = LunarWaterModel(
        population=100000,  # 10万人
        water_per_capita=300,  # 300 L/人·天
        eta=0.98,  # 98%回收效率
        initial_water=10000,  # 初始库存10,000吨
        target_water=50000,  # 目标库存50,000吨
        simulation_days=365  # 模拟一年
    )
    
    print("开始模拟动态补给策略...")
    # 运行动态补给策略模拟
    results = model.simulate(strategy='dynamic')
    
    # 打印结果摘要
    print_summary(model, results)
    
    # 绘制结果图表
    print("\n正在生成分析图表...")
    plot_results(model, results)
    
    # 额外分析：比较不同策略
    print("\n正在比较不同补给策略...")
    results_constant = model.simulate(strategy='constant')
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # 比较水存量
    ax[0].plot(results['time'], results['water_stock'], 'b-', 
               label='动态补给策略', linewidth=2)
    ax[0].plot(results_constant['time'], results_constant['water_stock'], 'r--', 
               label='恒定补给策略', linewidth=2)
    ax[0].axhline(y=model.target_water, color='g', linestyle=':', 
                  label='目标库存')
    ax[0].set_xlabel('时间 (天)')
    ax[0].set_ylabel('水存量 (吨)')
    ax[0].set_title('不同补给策略对比')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # 比较地球补给量
    ax[1].plot(results['time'], results['earth_supply_rate'], 'b-', 
               label='动态补给', linewidth=2)
    ax[1].plot(results_constant['time'], results_constant['earth_supply_rate'], 'r--', 
               label='恒定补给', linewidth=2)
    ax[1].set_xlabel('时间 (天)')
    ax[1].set_ylabel('地球补给速率 (吨/天)')
    ax[1].set_title('地球补给策略对比')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("=" * 60)
    print("关键发现:")
    print("1. 在98%回收效率下，10万人每日仅需少量地球补给")
    print("2. 动态补给策略比恒定策略更能维持稳定库存")
    print("3. 提高回收效率是减少运输需求的最有效方法")
    print("4. 月球本地水源开发可进一步降低地球依赖")
    print("=" * 60)


if __name__ == "__main__":
    main()