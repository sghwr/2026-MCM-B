import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

def create_gantt_timetable_simple(supply_schedule, save_path='gantt_timetable_simple.png'):
    """
    简洁版甘特图时间表可视化
    
    参数:
    - supply_schedule: 包含每日数据的DataFrame
    - save_path: 保存路径
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    
    df = supply_schedule.copy().sort_values('day')
    
    # 创建画布
    fig, (ax_main, ax_colorbar) = plt.subplots(2, 1, figsize=(18, 12), 
                                               gridspec_kw={'height_ratios': [10, 1]})
    
    # 颜色映射
    rocket_cmap = LinearSegmentedColormap.from_list('rocket', ['#FFCCCC', '#FF3333', '#990000'])
    elevator_cmap = LinearSegmentedColormap.from_list('elevator', ['#CCE5FF', '#3366FF', '#003399'])
    
    # 归一化补给量
    supply_min, supply_max = df['total_supply'].min(), df['total_supply'].max()
    supply_norm = (df['total_supply'] - supply_min) / (supply_max - supply_min + 1e-8)
    
    # 布局参数
    days_per_row = 7  # 每周一行
    num_rows = int(np.ceil(len(df) / days_per_row))
    bar_height = 0.6
    bar_width_scale = 3.0  # 条形宽度缩放因子
    
    # 创建月份背景
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_colors = ['#FFF0F0', '#F0FFF0', '#F0F8FF', '#FFF8F0', 
                   '#F8F0FF', '#F0FFFF', '#FFF8E8', '#F8FFF8',
                   '#F0F0FF', '#FFFFF0', '#F8F8F0', '#F0F8F8']
    
    # 绘制月份背景
    current_day = 0
    for month_idx, days_in_month in enumerate(month_days):
        start_row = current_day // days_per_row
        current_day += days_in_month
        end_row = min((current_day - 1) // days_per_row, num_rows - 1)
        
        if start_row <= end_row:
            y_bottom = num_rows - end_row - 1 - bar_height/2
            height = end_row - start_row + 1 + bar_height
            rect = patches.Rectangle((-0.5, y_bottom), 
                                    days_per_row * bar_width_scale + 1, 
                                    height,
                                    facecolor=month_colors[month_idx],
                                    alpha=0.3,
                                    edgecolor='none',
                                    zorder=0)
            ax_main.add_patch(rect)
    
    # 绘制每日条形
    for idx, row in df.iterrows():
        day_num = int(row['day'])
        supply = row['total_supply']
        method = row['transport_method']
        
        # 计算位置
        row_idx = (day_num - 1) // days_per_row
        col_idx = (day_num - 1) % days_per_row
        y_pos = num_rows - row_idx - 1
        
        # 计算条形宽度（基于补给量）
        bar_length = 0.3 + 2.5 * supply_norm[day_num-1]
        x_start = col_idx * bar_width_scale
        
        # 选择颜色
        if method == 'rocket':
            color = rocket_cmap(supply_norm[day_num-1])
            edge_color = '#990000'
        else:
            color = elevator_cmap(supply_norm[day_num-1])
            edge_color = '#003399'
        
        # 绘制条形
        rect = patches.Rectangle((x_start, y_pos - bar_height/2), 
                                bar_length, bar_height,
                                facecolor=color, alpha=0.8,
                                edgecolor=edge_color, linewidth=1)
        ax_main.add_patch(rect)
        
        # 添加日期标签
        ax_main.text(x_start + bar_length/2, y_pos, str(day_num),
                    ha='center', va='center', fontsize=7,
                    color='white' if supply_norm[day_num-1] > 0.6 else 'black')
        
        # 添加运输方式标记
        method_marker = 'R' if method == 'rocket' else 'E'
        ax_main.text(x_start - 0.2, y_pos, method_marker,
                    ha='right', va='center', fontsize=8, fontweight='bold')
    
    # 添加星期标签
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i in range(7):
        ax_main.text(i * bar_width_scale + bar_width_scale/2, num_rows + 0.3,
                    weekdays[i], ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 添加月份标签
    current_day = 0
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for month_idx, days_in_month in enumerate(month_days):
        mid_row = current_day // days_per_row + (days_in_month // 2) // days_per_row
        current_day += days_in_month
        y_pos = num_rows - mid_row - 0.5
        ax_main.text(-0.8, y_pos, month_names[month_idx],
                    ha='right', va='center', fontsize=10, fontweight='bold')
    
    # 设置坐标轴
    ax_main.set_xlim(-1, days_per_row * bar_width_scale)
    ax_main.set_ylim(-0.5, num_rows + 0.5)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_title('Lunar Base Supply Timetable', fontsize=14, fontweight='bold', pad=15)
    
    # 创建颜色条
    norm = plt.Normalize(supply_min, supply_max)
    
    # 火箭颜色条
    rocket_sm = plt.cm.ScalarMappable(cmap=rocket_cmap, norm=norm)
    rocket_sm.set_array([])
    
    # 电梯颜色条
    elevator_sm = plt.cm.ScalarMappable(cmap=elevator_cmap, norm=norm)
    elevator_sm.set_array([])
    
    # 添加颜色条
    cbar_rocket = plt.colorbar(rocket_sm, cax=ax_colorbar, orientation='horizontal', fraction=0.4)
    cbar_rocket.set_label('Rocket Supply (tons)', fontsize=10)
    cbar_rocket.ax.set_position([0.1, 0.5, 0.35, 0.3])
    
    # 添加第二个颜色条
    ax_colorbar2 = ax_colorbar.inset_axes([0.55, 0.5, 0.35, 0.3])
    cbar_elevator = plt.colorbar(elevator_sm, cax=ax_colorbar2, orientation='horizontal')
    cbar_elevator.set_label('Elevator Supply (tons)', fontsize=10)
    
    # 添加图例说明
    legend_text = "R = Rocket | E = Elevator\nBar length indicates supply quantity"
    ax_main.text(0.02, -0.1, legend_text, transform=ax_main.transAxes,
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加统计信息
    rocket_days = len(df[df['transport_method']=='rocket'])
    elevator_days = len(df[df['transport_method']=='elevator'])
    stats_text = f"Rocket: {rocket_days} days ({rocket_days/len(df)*100:.1f}%)\nElevator: {elevator_days} days"
    ax_main.text(0.98, 0.02, stats_text, transform=ax_main.transAxes,
                ha='right', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Gantt timetable saved to: {save_path}")
    return fig

# 使用示例
if __name__ == "__main__":
    df = pd.read_csv(r'D:\USELESS\数据分析学习\数学建模学习\2026美赛\B\PROB03\timetable.csv')
    fig = create_gantt_timetable_simple(df, 'gantt_simple.png')