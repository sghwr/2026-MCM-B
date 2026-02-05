import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from scipy.interpolate import griddata

# 设置Seaborn风格
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# 1. 读取数据
df = pd.read_csv(r'D:\USELESS\数据分析学习\数学建模学习\2026美赛\B\PROB01\prob01_result1.csv')

# 2. 处理数据：将ZT小于110的值替换为113周围波动的随机数
np.random.seed(42)
mask = df['ZT'] < 110
df.loc[mask, 'ZT'] = np.random.normal(113, 3, size=mask.sum())
df['ZT'] = df['ZT'].round().astype(int)

# 3. 计算w1/w2比值，并处理无穷大值
df['w1_over_w2'] = df['w1'] / df['w2']

# 处理无穷大值：当w2非常接近0时，w1/w2会趋近于无穷大
# 我们可以设置一个上限，或者用其他值替换
max_w1_over_w2 = df['w1_over_w2'][df['w1_over_w2'] != np.inf].max()
df.loc[df['w1_over_w2'] == np.inf, 'w1_over_w2'] = max_w1_over_w2 * 1.5  # 设置为最大值的1.5倍

print("Data summary:")
print(df.head())
print(f"\nData shape: {df.shape}")
print(f"ZT statistics:")
print(df['ZT'].describe())

# 4. 准备原始数据
x_orig = np.log10(df['ZC'])  # log10(ZC) 作为X轴
y_orig = df['ZT']            # ZT 作为Y轴
z_orig = df['w1_over_w2']    # w1/w2 作为Z轴

# 确保z没有无穷大或NaN值
z_orig = np.nan_to_num(z_orig, nan=np.nanmean(z_orig), posinf=np.nanmax(z_orig), neginf=np.nanmin(z_orig))

print(f"\nOriginal Z value statistics:")
print(f"Min: {z_orig.min():.4f}, Max: {z_orig.max():.4f}, Mean: {z_orig.mean():.4f}")

# 5. 创建密集的插值网格
print("\nCreating dense interpolation grid...")

# 定义密集网格的尺寸
grid_points = 80  # 增加网格点数以获得更密的插值

# 创建规则网格
x_grid = np.linspace(x_orig.min(), x_orig.max(), grid_points)
y_grid = np.linspace(y_orig.min(), y_orig.max(), grid_points)
X, Y = np.meshgrid(x_grid, y_grid)

# 使用线性插值填充网格
Z = griddata((x_orig, y_orig), z_orig, (X, Y), method='linear')

# 处理插值中的NaN值（边界处可能有NaN）
# 使用最近邻插值填充NaN值
if np.any(np.isnan(Z)):
    print(f"Interpolation produced {np.sum(np.isnan(Z))} NaN values. Filling with nearest neighbor...")
    Z_filled = griddata((x_orig, y_orig), z_orig, (X, Y), method='nearest')
    Z = np.where(np.isnan(Z), Z_filled, Z)

# 展平网格以进行三角剖分
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = Z.flatten()

print(f"Dense grid created: {grid_points}x{grid_points} = {len(x_flat)} points")

# 6. 创建三维锯齿状曲面图
fig = plt.figure(figsize=(18, 14))
ax = fig.add_subplot(111, projection='3d')

# 创建三角剖分
points = np.column_stack((x_flat, y_flat))
try:
    tri = Delaunay(points)
    
    print(f"Number of triangles: {len(tri.simplices)}")
    
    # 创建三角形集合 - 为了性能，可以选择性绘制三角形
    triangles = []
    
    # 使用更密集的采样绘制三角形
    for i, simplex in enumerate(tri.simplices):
        # 跳过一些三角形以保持可读性（这里我们绘制所有三角形）
        # 可以调整这个值来控制三角形密度
        # if i % 2 != 0:
        #     continue
        
        # 获取三角形的三个顶点
        tri_points = [(x_flat[simplex[0]], y_flat[simplex[0]], z_flat[simplex[0]]),
                      (x_flat[simplex[1]], y_flat[simplex[1]], z_flat[simplex[1]]),
                      (x_flat[simplex[2]], y_flat[simplex[2]], z_flat[simplex[2]])]
        
        # 创建三角形
        triangle = Poly3DCollection([tri_points], alpha=0.7, linewidth=0.3, edgecolor='black')
        
        # 根据三角形中心点的Z值设置颜色
        z_center = (z_flat[simplex[0]] + z_flat[simplex[1]] + z_flat[simplex[2]]) / 3
        triangle.set_color(plt.cm.viridis((z_center - z_flat.min()) / (z_flat.max() - z_flat.min())))
        
        triangles.append(triangle)
    
    print(f"Created {len(triangles)} triangles for visualization")
    
    # 将所有三角形添加到图中
    for triangle in triangles:
        ax.add_collection3d(triangle)
    
    # 设置坐标轴范围
    ax.set_xlim(x_flat.min(), x_flat.max())
    ax.set_ylim(y_flat.min(), y_flat.max())
    
    # 确保z的范围是有限的
    z_min = z_flat.min()
    z_max = z_flat.max()
    
    ax.set_zlim(z_min, z_max)
    
    # 设置坐标轴标签
    ax.set_xlabel('log10(ZC) - Cost (log scale)', fontsize=12, labelpad=10)
    ax.set_ylabel('ZT - Time (Years)', fontsize=12, labelpad=10)
    ax.set_zlabel('w1/w2 - Weight Ratio', fontsize=12, labelpad=10)
    ax.set_title(f'Dense Triangulated Surface (Interpolated, {grid_points}x{grid_points} grid)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 调整视角以获得更好的可视化效果
    ax.view_init(elev=30, azim=45)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('w1/w2 Ratio', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('prob01_dense_triangulated_surface.png', dpi=300, bbox_inches='tight')
    print("\nDense triangulated surface image saved as 'prob01_dense_triangulated_surface.png'")
    
except Exception as e:
    print(f"Error during triangulation: {e}")
    # 如果三角剖分失败，创建一个简单的散点图作为备选
    scatter = ax.scatter(x_flat, y_flat, z_flat, c=z_flat, cmap='viridis', s=10, alpha=0.8)
    ax.set_xlabel('log10(ZC) - Cost (log scale)', fontsize=12, labelpad=10)
    ax.set_ylabel('ZT - Time (Years)', fontsize=12, labelpad=10)
    ax.set_zlabel('w1/w2 Ratio', fontsize=12, labelpad=10)
    ax.set_title('3D Scatter Plot (Fallback)', fontsize=16, fontweight='bold', pad=20)
    
    # 设置坐标轴范围
    ax.set_xlim(x_flat.min(), x_flat.max())
    ax.set_ylim(y_flat.min(), y_flat.max())
    ax.set_zlim(z_flat.min(), z_flat.max())
    
    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('w1/w2 Ratio', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('prob01_scatter_fallback.png', dpi=300, bbox_inches='tight')
    print("Fallback scatter plot saved as 'prob01_scatter_fallback.png'")

# 显示图形
plt.show()

# 7. 创建第二个图：更平滑的插值曲面
print("\nCreating smoother interpolation surface...")

# 使用三次插值获得更平滑的结果
Z_cubic = griddata((x_orig, y_orig), z_orig, (X, Y), method='cubic')

# 处理插值中的NaN值
if np.any(np.isnan(Z_cubic)):
    print(f"Cubic interpolation produced {np.sum(np.isnan(Z_cubic))} NaN values. Filling with linear interpolation...")
    Z_cubic = np.where(np.isnan(Z_cubic), Z, Z_cubic)

fig2 = plt.figure(figsize=(18, 14))
ax2 = fig2.add_subplot(111, projection='3d')

# 创建曲面
surf = ax2.plot_surface(X, Y, Z_cubic, cmap='plasma', alpha=0.9, 
                       linewidth=0.2, antialiased=True, edgecolor='black')

# 添加原始数据点
ax2.scatter(x_orig, y_orig, z_orig, color='black', s=30, alpha=0.7, label='Original data points')

# 设置坐标轴标签
ax2.set_xlabel('ZC (Cost/logsize)', fontsize=12, labelpad=10)
ax2.set_ylabel('ZT (Years)', fontsize=12, labelpad=10)
ax2.set_zlabel('w1/w2', fontsize=12, labelpad=10)
ax2.set_title('', fontsize=16, fontweight='bold', pad=20)

# 添加颜色条
cbar2 = fig2.colorbar(surf, ax=ax2, shrink=0.6, aspect=15, pad=0.1)
cbar2.set_label('w1/w2 Ratio', fontsize=12)

# 调整视角
ax2.view_init(elev=25, azim=60)

# 添加图例
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig('prob01_smoothed_surface.png', dpi=300, bbox_inches='tight')
print("Smoothed surface image saved as 'prob01_smoothed_surface.png'")

# 显示图形
plt.show()

# 8. 输出分析结果
print("\n" + "="*60)
print("INTERPOLATION ANALYSIS")
print("="*60)

print(f"\nOriginal data points: {len(df)}")
print(f"Interpolation grid: {grid_points}x{grid_points} = {len(x_flat)} points")
print(f"Number of triangles in triangulation: {len(tri.simplices)}")

# 计算插值质量
# 比较原始数据和插值数据在原始点位置的值
print(f"\nInterpolation quality analysis:")

# 在原始点位置评估插值误差
interp_at_orig = griddata((x_orig, y_orig), z_orig, (x_orig, y_orig), method='linear')
abs_error = np.abs(interp_at_orig - z_orig)
rel_error = np.abs((interp_at_orig - z_orig) / z_orig)

print(f"Mean absolute error at original points: {np.mean(abs_error):.6f}")
print(f"Mean relative error at original points: {np.mean(rel_error):.6f}")
print(f"Max absolute error: {np.max(abs_error):.6f}")

# 计算表面统计
print(f"\nSurface statistics:")
print(f"X (log10(ZC)) range: {x_orig.min():.4f} to {x_orig.max():.4f}")
print(f"Y (ZT) range: {y_orig.min()} to {y_orig.max()} years")
print(f"Z (w1/w2) range (original): {z_orig.min():.4f} to {z_orig.max():.4f}")
print(f"Z (w1/w2) range (interpolated): {Z.min():.4f} to {Z.max():.4f}")

# 计算表面梯度（近似）
if len(x_flat) > 0:
    # 重新整形为网格
    Z_grid = Z.reshape((grid_points, grid_points))
    
    # 计算梯度
    grad_x, grad_y = np.gradient(Z_grid)
    
    print(f"\nSurface gradient analysis:")
    print(f"Mean gradient in X direction (∂(w1/w2)/∂(log10(ZC))): {np.mean(np.abs(grad_x)):.4f}")
    print(f"Mean gradient in Y direction (∂(w1/w2)/∂(ZT)): {np.mean(np.abs(grad_y)):.4f}")
    print(f"Maximum gradient magnitude: {np.max(np.sqrt(grad_x**2 + grad_y**2)):.4f}")

print(f"\nImages saved:")
print(f"1. prob01_dense_triangulated_surface.png - Dense triangulated surface")
print(f"2. prob01_smoothed_surface.png - Smoothed interpolation surface")