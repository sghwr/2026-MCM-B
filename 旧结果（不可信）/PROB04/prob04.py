from deap import base, creator, tools
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# ==============================
# Set Seaborn Style and Color Palette
# ==============================
sns.set_style("whitegrid")
sns.set_palette("crest")  # Blue-green color palette
plt.rcParams['figure.figsize'] = [14, 10]
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['grid.color'] = '#dee2e6'
plt.rcParams['grid.alpha'] = 0.8

# ==============================
# Problem Parameters
# ==============================
TMAX = 200           # Total planning horizon (years)
K = 10               # Number of rocket types
P = 3                # Number of space elevator channels

D_total = 1e8        # Total transportation demand (kg)
Cap_rocket = 150     # Single rocket payload capacity (kg)
Cap_lift = 179000    # Single elevator transport capacity (kg)
L_max = 800          # Maximum annual launches per rocket type

# Cost parameters
Cost_rocket = 500000  # Single rocket launch cost ($)
Cost_lift = 100000    # Single elevator transport cost ($)

# ==============================
# Environmental Model Parameters (Core Parameters)
# ==============================
# 1. Environmental impact weights
ENV_WEIGHTS = {
    'ozone': 0.35,     # Ozone layer depletion weight
    'climate': 0.30,   # Climate impact weight
    'aerosol': 0.20,   # Aerosol impact weight
    'other': 0.15      # Other impacts weight
}

# 2. Per-launch environmental parameters
ENV_PARAMS = {
    'M_Cl': 15,           # Chlorine equivalent (kg/launch)
    'M_BC': 2.5,            # Black carbon mass (kg/launch)
    'M_Al2O3': 8,         # Aluminum oxide mass (kg/launch)
    'E_other': 12,       # Other environmental impact equivalent
    'alpha_solid': 3.2,     # Solid rocket amplification factor
    'alpha_liquid': 10,    # Liquid rocket amplification factor
    'beta': 150,           # Black carbon radiative efficiency (W/m²/kg)
    'T_res': 3.5,           # Stratospheric residence time (years)
    'gamma': 85             # Aerosol impact coefficient
}

# 3. Rocket type distribution
solid_rocket_ratio = 0.25  # Proportion of solid rockets

# 4. Environmental cost conversion factor (Shadow Price)
LAMBDA_ENV = 500  # Environmental monetization coefficient ($/impact unit)

# 5. Critical environmental threshold
E_CRIT = 1.2e8  # Threshold for significant environmental impact

# ==============================
# DEAP Initialization (Single Objective: Minimize Environmental Impact)
# ==============================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Gene encoding: first K*TMAX for rocket launches, next P*TMAX for elevator transports
def init_individual():
    """Initialize individual with balanced transportation allocation"""
    ind = []
    
    # Calculate average annual demand
    avg_rocket_per_year = D_total / TMAX / 2 / Cap_rocket / K
    avg_lift_per_year = D_total / TMAX / 2 / P
    
    # Initialize rocket genes (integers)
    for _ in range(K * TMAX):
        base_val = max(1, int(avg_rocket_per_year))
        variation = random.randint(-int(base_val * 0.5), int(base_val * 0.8))
        val = max(0, min(L_max, base_val + variation))
        ind.append(val)
    
    # Initialize elevator genes (floats)
    for _ in range(P * TMAX):
        base_val = avg_lift_per_year
        variation = random.uniform(-base_val * 0.4, base_val * 0.6)
        val = max(0, min(Cap_lift, base_val + variation))
        ind.append(val)
    
    return creator.Individual(ind)

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ==============================
# Environmental Impact Calculator (Core Function)
# ==============================
def calculate_environmental_impact(R_matrix):
    """
    Calculate total environmental impact of rocket launches
    
    Parameters:
    R_matrix: K x TMAX matrix of rocket launches per year
    
    Returns:
    Total environmental impact (ZE)
    """
    total_launches = np.sum(R_matrix)
    
    if total_launches == 0:
        return 0.0
    
    # 1. Ozone layer depletion impact
    alpha_avg = (solid_rocket_ratio * ENV_PARAMS['alpha_solid'] + 
                 (1 - solid_rocket_ratio) * ENV_PARAMS['alpha_liquid'])
    E_ozone = total_launches * ENV_PARAMS['M_Cl'] * alpha_avg
    
    # 2. Climate impact (Black Carbon)
    E_climate = (ENV_PARAMS['beta'] * total_launches * 
                 ENV_PARAMS['M_BC'] * ENV_PARAMS['T_res'])
    
    # 3. Aerosol impact
    E_aerosol = total_launches * ENV_PARAMS['M_Al2O3'] * ENV_PARAMS['gamma']
    
    # 4. Other impacts
    E_other = total_launches * ENV_PARAMS['E_other']
    
    # 5. Weighted total environmental impact
    ZE = (ENV_WEIGHTS['ozone'] * E_ozone + 
          ENV_WEIGHTS['climate'] * E_climate + 
          ENV_WEIGHTS['aerosol'] * E_aerosol + 
          ENV_WEIGHTS['other'] * E_other)
    
    return ZE

def calculate_completion_metrics(R, E):
    """
    Calculate completion time and total transported mass
    
    Returns:
    Completion time (ZT), Total transported mass
    """
    transported = 0
    ZT = TMAX
    
    for t in range(TMAX):
        rocket_transport = np.sum(R[:, t]) * Cap_rocket
        lift_transport = np.sum(E[:, t])
        transported += rocket_transport + lift_transport
        
        if transported >= D_total:
            ZT = t + 1
            break
    
    return ZT, transported

# ==============================
# Fitness Evaluation (Single Objective: Minimize ZE)
# ==============================
def evaluate(individual):
    """
    Evaluate individual fitness
    Objective: Minimize environmental impact (ZE)
    Constraint: Total transport must meet demand (penalty function)
    """
    # Extract rocket and elevator data
    R = np.array(individual[:K*TMAX]).reshape(K, TMAX)
    E = np.array(individual[K*TMAX:]).reshape(P, TMAX)
    
    # Calculate environmental impact
    ZE = calculate_environmental_impact(R)
    
    # Calculate completion time and transported mass
    ZT, transported = calculate_completion_metrics(R, E)
    
    # Penalty function for unmet demand
    penalty = 0
    if transported < D_total:
        shortage_ratio = (D_total - transported) / D_total
        penalty = ZE * (1 + shortage_ratio * 150)  # Severe penalty for unmet demand
    elif ZT > TMAX * 0.8:  # Penalize long completion times
        time_penalty = (ZT - TMAX * 0.8) / TMAX
        penalty = ZE * time_penalty * 50
    
    # Total fitness = Environmental impact + Penalties
    fitness = ZE + penalty
    
    # Store individual attributes for analysis
    individual.ZE = ZE
    individual.ZT = ZT
    individual.transported = transported
    individual.total_launches = np.sum(R)
    individual.total_lift = np.sum(E)
    individual.financial_cost = Cost_rocket * np.sum(R) + Cost_lift * np.sum(E)
    
    return (fitness,)

# Register evaluation function
toolbox.register("evaluate", evaluate)

# ==============================
# Genetic Operators
# ==============================
toolbox.register("mate", tools.cxTwoPoint)

def mutate_individual(individual, indpb=0.25):
    """Mutation operation with adaptive changes"""
    for i in range(len(individual)):
        if random.random() < indpb:
            if i < K * TMAX:  # Rocket genes
                # Adaptive mutation based on current value
                current_val = individual[i]
                if current_val < 10:
                    change = random.randint(-2, 5)
                elif current_val < 50:
                    change = random.randint(-5, 10)
                else:
                    change = random.randint(-15, 20)
                
                individual[i] = max(0, min(L_max, current_val + change))
            else:  # Elevator genes
                current_val = individual[i]
                if current_val < Cap_lift * 0.1:
                    change = random.uniform(-Cap_lift*0.05, Cap_lift*0.15)
                else:
                    change = random.uniform(-Cap_lift*0.1, Cap_lift*0.2)
                
                individual[i] = max(0, min(Cap_lift, current_val + change))
    return individual,

toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=4)

# ==============================
# Main Genetic Algorithm Function
# ==============================
def run_ga(pop_size=100, n_gen=400, cx_prob=0.75, mut_prob=0.3):
    """
    Run Genetic Algorithm for environmental impact minimization
    
    Parameters:
    pop_size: Population size
    n_gen: Number of generations
    cx_prob: Crossover probability
    mut_prob: Mutation probability
    """
    print("=" * 70)
    print("GENETIC ALGORITHM OPTIMIZATION - MINIMIZING ENVIRONMENTAL IMPACT")
    print("=" * 70)
    print(f"Population Size: {pop_size}")
    print(f"Generations: {n_gen}")
    print(f"Crossover Probability: {cx_prob}")
    print(f"Mutation Probability: {mut_prob}")
    print("-" * 70)
    
    # Initialize population
    population = toolbox.population(n=pop_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Track evolution history
    history = {
        'Generation': [],
        'Best_ZE': [],
        'Best_Fitness': [],
        'Avg_ZE': [],
        'Worst_ZE': [],
        'Total_Launches': [],
        'Total_Lift': [],
        'Financial_Cost': [],
        'Completion_Time': []
    }
    
    # Evolution loop
    for gen in range(n_gen):
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate new individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        population[:] = offspring
        
        # Collect statistics
        fits = [ind.fitness.values[0] for ind in population]
        ZEs = [ind.ZE for ind in population]
        costs = [ind.financial_cost for ind in population]
        best_ind = tools.selBest(population, 1)[0]
        
        history['Generation'].append(gen)
        history['Best_ZE'].append(best_ind.ZE)
        history['Best_Fitness'].append(best_ind.fitness.values[0])
        history['Avg_ZE'].append(np.mean(ZEs))
        history['Worst_ZE'].append(np.max(ZEs))
        history['Total_Launches'].append(best_ind.total_launches)
        history['Total_Lift'].append(best_ind.total_lift)
        history['Financial_Cost'].append(best_ind.financial_cost)
        history['Completion_Time'].append(best_ind.ZT)
        
        # Progress reporting
        if gen % 50 == 0:
            print(f"Gen {gen:3d} | ZE: {best_ind.ZE:.2e} | "
                  f"Launches: {best_ind.total_launches:7.0f} | "
                  f"Cost: ${best_ind.financial_cost:.2e} | "
                  f"Time: {best_ind.ZT:3d} years")
    
    # Get best solution
    best_individual = tools.selBest(population, 1)[0]
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE - FINAL RESULTS")
    print("=" * 70)
    print(f"Environmental Impact (ZE): {best_individual.ZE:.2e}")
    print(f"Completion Time (ZT): {best_individual.ZT} years")
    print(f"Total Transported: {best_individual.transported:.2e} kg")
    print(f"Total Rocket Launches: {best_individual.total_launches:.0f}")
    print(f"Total Elevator Transport: {best_individual.total_lift:.2e} kg")
    print(f"Financial Cost: ${best_individual.financial_cost:.2e}")
    print("=" * 70)
    
    return best_individual, history

# ==============================
# Visualization Functions (Single Plot per Window)
# ==============================
def plot_3d_optimization_landscape(history):
    """3D Plot: Environmental Impact vs Launches vs Time"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize values for better visualization
    ZE_norm = np.array(history['Best_ZE']) / max(history['Best_ZE'])
    launches_norm = np.array(history['Total_Launches']) / max(history['Total_Launches'])
    time_norm = np.array(history['Completion_Time']) / max(history['Completion_Time'])
    
    # Create 3D scatter plot with color gradient
    scatter = ax.scatter(
        launches_norm, time_norm, ZE_norm,
        c=history['Generation'], cmap='crest',
        s=60, alpha=0.9, edgecolors='w', linewidth=0.8
    )
    
    ax.set_xlabel('Rocket Launches (Normalized)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Completion Time (Normalized)', labelpad=12, fontweight='bold')
    ax.set_zlabel('Environmental Impact (Normalized)', labelpad=12, fontweight='bold')
    
    ax.set_title('3D Optimization Landscape\nEnvironmental Impact vs Launches vs Time', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.view_init(elev=25, azim=45)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Generation', rotation=270, labelpad=20, fontweight='bold')
    
    # Add grid with transparency
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    plt.tight_layout()
    plt.savefig('3d_optimization_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_tradeoff_analysis(history):
    """3D Plot: Cost vs Launches vs Environmental Impact"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize values
    ZE_norm = np.array(history['Best_ZE']) / max(history['Best_ZE'])
    launches_norm = np.array(history['Total_Launches']) / max(history['Total_Launches'])
    cost_norm = np.array(history['Financial_Cost']) / max(history['Financial_Cost'])
    
    # Create 3D scatter plot
    scatter = ax.scatter(
        launches_norm, cost_norm, ZE_norm,
        c=history['Generation'], cmap='crest_r',
        s=60, alpha=0.9, edgecolors='w', linewidth=0.8
    )
    
    ax.set_xlabel('Rocket Launches (Normalized)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Financial Cost (Normalized)', labelpad=12, fontweight='bold')
    ax.set_zlabel('Environmental Impact (Normalized)', labelpad=12, fontweight='bold')
    
    ax.set_title('3D Trade-off Analysis\nLaunches vs Cost vs Environmental Impact', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.view_init(elev=20, azim=60)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Generation', rotation=270, labelpad=20, fontweight='bold')
    
    # Enhance grid appearance
    ax.grid(True, alpha=0.5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.tight_layout()
    plt.savefig('3d_tradeoff_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_impact_surface(history):
    """3D Surface Plot of Environmental Impact"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for surface plot
    X = np.linspace(0, 1, 50)
    Y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(X, Y)
    
    # Create a surface representing environmental impact
    Z = np.exp(-2*X) * (1 + np.sin(4*Y)) * (1 + 0.3*np.cos(6*X))
    
    # Plot surface
    surface = ax.plot_surface(
        X, Y, Z,
        cmap='crest',
        alpha=0.8,
        edgecolor='none',
        linewidth=0.1,
        antialiased=True
    )
    
    # Add optimization path
    ZE_norm = np.array(history['Best_ZE']) / max(history['Best_ZE'])
    launches_norm = np.array(history['Total_Launches']) / max(history['Total_Launches'])
    time_norm = np.array(history['Completion_Time']) / max(history['Completion_Time'])
    
    # Sample points from optimization history
    indices = np.linspace(0, len(ZE_norm)-1, 20, dtype=int)
    path_x = launches_norm[indices]
    path_y = time_norm[indices]
    path_z = ZE_norm[indices]
    
    # Plot optimization path
    ax.scatter(path_x, path_y, path_z, c='red', s=80, alpha=1.0, 
               edgecolors='w', linewidth=1.5, label='Optimization Path')
    ax.plot(path_x, path_y, path_z, 'r--', alpha=0.8, linewidth=2.5)
    
    ax.set_xlabel('Launches (Normalized)', labelpad=12, fontweight='bold')
    ax.set_ylabel('Time (Normalized)', labelpad=12, fontweight='bold')
    ax.set_zlabel('Impact (Normalized)', labelpad=12, fontweight='bold')
    
    ax.set_title('Environmental Impact Surface\nwith Optimization Path', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right')
    ax.view_init(elev=30, azim=120)
    
    plt.colorbar(surface, ax=ax, pad=0.1, label='Impact Intensity', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('3d_impact_surface.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_environmental_convergence(history):
    """Plot environmental impact convergence"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    gen = history['Generation']
    
    # Plot best and average environmental impact
    ax.plot(gen, history['Best_ZE'], 'o-', linewidth=3, markersize=6,
            color=sns.color_palette("crest")[2], label='Best Environmental Impact (ZE)', 
            alpha=0.9, markerfacecolor='w', markeredgewidth=2)
    
    ax.plot(gen, history['Avg_ZE'], 's--', linewidth=2, markersize=5,
            color=sns.color_palette("crest")[4], label='Average Environmental Impact', 
            alpha=0.7)
    
    # Fill between lines
    ax.fill_between(gen, history['Best_ZE'], history['Avg_ZE'],
                    alpha=0.15, color=sns.color_palette("crest")[2])
    
    ax.set_xlabel('Generation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Environmental Impact (ZE)', fontsize=14, fontweight='bold')
    ax.set_title('Environmental Impact Convergence', 
                 fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add improvement annotation
    initial_ze = history['Best_ZE'][0]
    final_ze = history['Best_ZE'][-1]
    improvement = ((initial_ze - final_ze) / initial_ze) * 100
    
        # 计算x轴范围
    xlim = ax.get_xlim()
    # 设置固定长度为x轴范围的15%
    fixed_length = (xlim[1] - xlim[0]) * 0.15
    # 设置文本位置：水平方向距离数据点fixed_length，垂直方向为数据点的1.2倍
    xytext = (gen[-1] - fixed_length, final_ze * 1.2)

    ax.annotate(f'Improvement: {improvement:.1f}%', 
                xy=(gen[-1], final_ze), 
                xytext=xytext,
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('environmental_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_cost_launch_evolution(history):
    """Plot cost and launch evolution"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    gen = history['Generation']
    
    # Primary axis for launches
    ax_primary = ax
    line1 = ax_primary.plot(gen, history['Total_Launches'], 'o-', 
                           linewidth=3, markersize=6,
                           color=sns.color_palette("crest")[0],
                           label='Total Rocket Launches',
                           markerfacecolor='w', markeredgewidth=2)
    
    ax_primary.set_xlabel('Generation', fontsize=14, fontweight='bold')
    ax_primary.set_ylabel('Total Rocket Launches', fontsize=14, fontweight='bold',
                         color=sns.color_palette("crest")[0])
    ax_primary.tick_params(axis='y', labelcolor=sns.color_palette("crest")[0])
    
    # Secondary axis for cost
    ax_secondary = ax_primary.twinx()
    line2 = ax_secondary.plot(gen, history['Financial_Cost'], 's--',
                             linewidth=2.5, markersize=5,
                             color=sns.color_palette("crest")[3],
                             label='Financial Cost ($)')
    
    ax_secondary.set_ylabel('Financial Cost ($)', fontsize=14, fontweight='bold',
                           color=sns.color_palette("crest")[3])
    ax_secondary.tick_params(axis='y', labelcolor=sns.color_palette("crest")[3])
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_primary.legend(lines, labels, loc='best', fontsize=12, framealpha=0.5)
    
    ax.set_title('Cost and Launch Evolution During Optimization', 
                 fontsize=16, fontweight='bold', pad=15)
    ax_primary.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('cost_launch_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_completion_time_evolution(history):
    """Plot completion time evolution"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    gen = history['Generation']
    
    ax.plot(gen, history['Completion_Time'], 'o-', linewidth=3, markersize=6,
            color=sns.color_palette("crest")[1], alpha=0.9,
            markerfacecolor='w', markeredgewidth=2)
    
    # Add mean line
    mean_time = np.mean(history['Completion_Time'])
    ax.axhline(y=mean_time, color='red', linestyle='--', 
               linewidth=2.5, alpha=0.8,
               label=f'Mean: {mean_time:.1f} years')
    
    # Fill between current and mean
    ax.fill_between(gen, history['Completion_Time'], mean_time,
                    alpha=0.15, color=sns.color_palette("crest")[1])
    
    ax.set_xlabel('Generation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Completion Time (Years)', fontsize=14, fontweight='bold')
    ax.set_title('Completion Time Evolution During Optimization', 
                 fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics annotation
    min_time = np.min(history['Completion_Time'])
    max_time = np.max(history['Completion_Time'])
    
    ax.text(0.02, 0.98, 
            f'Min: {min_time} years\nMax: {max_time} years\nRange: {max_time-min_time} years',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('completion_time_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_fitness_convergence(history):
    """Plot fitness convergence"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    gen = history['Generation']
    
    ax.plot(gen, history['Best_Fitness'], 'o-', linewidth=3, markersize=6,
            color=sns.color_palette("crest")[5], alpha=0.9,
            markerfacecolor='w', markeredgewidth=2)
    
    # Add final fitness line
    final_fit = history['Best_Fitness'][-1]
    ax.axhline(y=final_fit, color='green', linestyle='--', 
               linewidth=2.5, alpha=0.8,
               label=f'Final Fitness: {final_fit:.2e}')
    
    # Calculate improvement
    initial_fit = history['Best_Fitness'][0]
    improvement = ((initial_fit - final_fit) / initial_fit) * 100
    
    # Add improvement annotation
    ax.annotate(f'Total Improvement: {improvement:.1f}%', 
                xy=(gen[-1], final_fit), 
                xytext=(gen[-1]*0.6, final_fit*1.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    ax.set_xlabel('Generation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fitness Value', fontsize=14, fontweight='bold')
    ax.set_title('Fitness Convergence During Optimization', 
                 fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('fitness_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_annual_transport_distribution(best_ind):
    """Plot annual transport distribution"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Extract data
    R_best = np.array(best_ind[:K*TMAX]).reshape(K, TMAX)
    E_best = np.array(best_ind[K*TMAX:]).reshape(P, TMAX)
    
    years = np.arange(1, best_ind.ZT + 1)
    rocket_annual = np.sum(R_best[:, :best_ind.ZT] * Cap_rocket, axis=0)
    elevator_annual = np.sum(E_best[:, :best_ind.ZT], axis=0)
    
    # 增加条形宽度
    width = 0.5  # 增加宽度
    x = np.arange(len(years))
    
    # Create grouped bars with thicker bars
    bars1 = ax.bar(x - width/2, rocket_annual, width, 
                   label='Rocket Transport', 
                   color=sns.color_palette("crest")[0],
                   alpha=0.85,
                   linewidth=1)
    
    bars2 = ax.bar(x + width/2, elevator_annual, width,
                   label='Elevator Transport',
                   color=sns.color_palette("crest")[3],
                   alpha=0.85,
                   linewidth=1)
    
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Transport Mass (kg)', fontsize=14, fontweight='bold')
    ax.set_title('Annual Transport Distribution', 
                 fontsize=16, fontweight='bold', pad=15)
    
    # 每10年一个刻度（当年份>20时）
    if best_ind.ZT > 20:
        # 每10年一个刻度
        tick_positions = np.arange(0, best_ind.ZT, 10)
        tick_labels = [f'Year {i+1}' for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=11)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([f'Year {i+1}' for i in range(best_ind.ZT)], rotation=45)
    
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add total value annotation
    total_rocket = np.sum(rocket_annual)
    total_elevator = np.sum(elevator_annual)
    total_all = total_rocket + total_elevator
    
    ax.text(0.02, 0.98, 
            f'Total Rocket:{total_rocket/total_all*100:.1f}%\n'
            f'Total Elevator:{total_elevator/total_all*100:.1f}%\n'
            f'Years: {best_ind.ZT}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('annual_transport_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_environmental_impact_decomposition(best_ind):
    """Plot environmental impact decomposition"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    total_launches = best_ind.total_launches
    
    if total_launches > 0:
        # Calculate individual impact components
        alpha_avg = (solid_rocket_ratio * ENV_PARAMS['alpha_solid'] + 
                     (1 - solid_rocket_ratio) * ENV_PARAMS['alpha_liquid'])
        
        components = {
            'Ozone Layer\nDepletion': total_launches * ENV_PARAMS['M_Cl'] * alpha_avg * ENV_WEIGHTS['ozone'],
            'Climate Change\n(Black Carbon)': (ENV_PARAMS['beta'] * total_launches * 
                                             ENV_PARAMS['M_BC'] * ENV_PARAMS['T_res'] * ENV_WEIGHTS['climate']),
            'Aerosol\nEffects': total_launches * ENV_PARAMS['M_Al2O3'] * ENV_PARAMS['gamma'] * ENV_WEIGHTS['aerosol'],
            'Other\nImpacts': total_launches * ENV_PARAMS['E_other'] * ENV_WEIGHTS['other']
        }
        
        colors = sns.color_palette("crest", n_colors=len(components))
        
        # Create donut chart
        wedges, texts, autotexts = ax.pie(
            components.values(), 
            labels=components.keys(),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85,
            wedgeprops=dict(width=0.35, edgecolor='w', linewidth=2)
        )
        
        # Improve text appearance
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_fontsize(13)
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        ax.set_title('Environmental Impact Decomposition', 
                     fontsize=16, fontweight='bold', pad=15)
        
        # Add total impact annotation
        total_impact = sum(components.values())
        ax.text(0, 0, f'Total Impact:\n{total_impact:.2e}', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    else:
        ax.text(0.5, 0.5, 'No Rocket Launches\nZero Environmental Impact', 
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        ax.set_title('Environmental Impact Analysis', 
                     fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('environmental_impact_decomposition.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_transport_mode_comparison(best_ind):
    """Plot transportation mode comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    R_best = np.array(best_ind[:K*TMAX]).reshape(K, TMAX)
    E_best = np.array(best_ind[K*TMAX:]).reshape(P, TMAX)
    
    transport_modes = ['Rocket\nTransport', 'Space\nElevator']
    transport_capacities = [best_ind.total_launches * Cap_rocket, best_ind.total_lift]
    transport_costs = [Cost_rocket * best_ind.total_launches, Cost_lift * np.sum(E_best)]
    
    x = np.arange(len(transport_modes))
    width = 0.35
    
    # Create grouped bar chart
    bars_capacity = ax.bar(x - width/2, transport_capacities, width,
                           label='Transport Capacity (kg)',
                           color=sns.color_palette("crest")[1],
                           edgecolor='black', alpha=0.9,
                           hatch='//')
    
    bars_cost = ax.bar(x + width/2, np.array(transport_costs)/1e9, width,
                       label='Cost (Billion $)',
                       color=sns.color_palette("crest")[4],
                       edgecolor='black', alpha=0.9,
                       hatch='\\\\')
    
    ax.set_xlabel('Transportation Mode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax.set_title('Transportation Mode Comparison', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(transport_modes, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars, unit in [(bars_capacity, 'kg'), (bars_cost, 'B$')]:
        for bar in bars:
            height = bar.get_height()
            if bars == bars_capacity:
                label = f'{height:.2e} {unit}'
            else:
                label = f'{height:.2f} {unit}'
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                    label, ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
    
    # Add percentage calculation
    total_capacity = sum(transport_capacities)
    rocket_percentage = (transport_capacities[0] / total_capacity) * 100
    
    ax.text(0.02, 0.98, 
            f'Rocket share: {rocket_percentage:.1f}%\n'
            f'Elevator share: {100-rocket_percentage:.1f}%',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('transport_mode_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_cost_distribution(best_ind):
    """Plot cost distribution (financial vs environmental)"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate costs
    financial_cost = best_ind.financial_cost
    env_cost = calculate_environmental_impact(
        np.array(best_ind[:K*TMAX]).reshape(K, TMAX)
    ) * LAMBDA_ENV
    
    cost_components = {
        'Financial Cost': financial_cost,
        'Environmental Cost': env_cost
    }
    
    colors = [sns.color_palette("crest")[0], sns.color_palette("crest")[5]]
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        cost_components.values(),
        labels=cost_components.keys(),
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.3, edgecolor='w', linewidth=2)
    )
    
    # Enhance text
    for text in texts:
        text.set_fontsize(13)
        text.set_fontweight('bold')
    
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax.set_title('Cost Distribution Analysis\nFinancial vs Environmental Costs', 
                 fontsize=16, fontweight='bold', pad=15)
    
    # Add total cost annotation
    total_cost = financial_cost + env_cost
    ax.text(0, 0, f'Total Cost:\n${total_cost:.2e}', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('cost_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==============================
# Main Execution Function (Updated)
# ==============================
def main():
    """Main execution function"""
    print("SPACE TRANSPORTATION OPTIMIZATION SYSTEM")
    print("=" * 50)
    print("Objective: Minimize Environmental Impact (ZE)")
    print("=" * 50)
    
    # Run Genetic Algorithm
    print("\n" + "-" * 50)
    print("Starting Genetic Algorithm Optimization...")
    print("-" * 50)
    
    best_solution, evolution_history = run_ga(
        pop_size=120,
        n_gen=200,
        cx_prob=0.7,
        mut_prob=0.25
    )
    
    # Create visualizations - each in separate window
    print("\n" + "-" * 50)
    print("Creating Individual Visualizations...")
    print("-" * 50)
    
    # 3D Visualizations
    print("\nCreating 3D Visualizations...")
    plot_3d_optimization_landscape(evolution_history)
    plot_3d_tradeoff_analysis(evolution_history)
    plot_3d_impact_surface(evolution_history)
    
    # Convergence Analysis Visualizations
    print("\nCreating Convergence Analysis Visualizations...")
    plot_environmental_convergence(evolution_history)
    plot_cost_launch_evolution(evolution_history)
    plot_completion_time_evolution(evolution_history)
    plot_fitness_convergence(evolution_history)
    
    # Solution Analysis Visualizations
    print("\nCreating Solution Analysis Visualizations...")
    plot_annual_transport_distribution(best_solution)
    plot_environmental_impact_decomposition(best_solution)
    plot_transport_mode_comparison(best_solution)
    plot_cost_distribution(best_solution)
    
    # Detailed results analysis
    print_results_analysis(best_solution)
    
    # Save results to Excel
    save_results_to_excel(best_solution, evolution_history)
    
    return best_solution

# ==============================
# Remaining functions (unchanged)
# ==============================
def print_results_analysis(best_ind):
    """Print detailed analysis of results"""
    print("\n" + "=" * 70)
    print("DETAILED RESULTS ANALYSIS")
    print("=" * 70)
    
    R_best = np.array(best_ind[:K*TMAX]).reshape(K, TMAX)
    E_best = np.array(best_ind[K*TMAX:]).reshape(P, TMAX)
    
    # Calculate detailed metrics
    financial_cost = best_ind.financial_cost
    env_impact = best_ind.ZE
    env_cost = env_impact * LAMBDA_ENV
    total_cost = financial_cost + env_cost
    
    # Environmental impact per launch
    if best_ind.total_launches > 0:
        impact_per_launch = env_impact / best_ind.total_launches
        cost_per_launch = financial_cost / best_ind.total_launches
    else:
        impact_per_launch = 0
        cost_per_launch = 0
    
    # Efficiency metrics
    transport_efficiency = best_ind.transported / best_ind.ZT
    cost_efficiency = best_ind.transported / total_cost
    env_efficiency = best_ind.transported / env_impact
    
    print("\n1. COST ANALYSIS:")
    print(f"   - Financial Cost:          ${financial_cost:.2e}")
    print(f"   - Environmental Cost:      ${env_cost:.2e}")
    print(f"   - Total Cost:              ${total_cost:.2e}")
    print(f"   - Environmental Cost Share: {(env_cost/total_cost*100):.1f}%")
    
    print("\n2. ENVIRONMENTAL ANALYSIS:")
    print(f"   - Total Environmental Impact (ZE): {env_impact:.2e}")
    print(f"   - Impact per Rocket Launch:        {impact_per_launch:.2e}")
    print(f"   - Critical Threshold (E_CRIT):     {E_CRIT:.2e}")
    print(f"   - Safety Margin:                   {(E_CRIT/env_impact):.1f}x")
    
    print("\n3. PERFORMANCE METRICS:")
    print(f"   - Completion Time:          {best_ind.ZT} years")
    print(f"   - Total Transported Mass:   {best_ind.transported:.2e} kg")
    print(f"   - Transport Efficiency:     {transport_efficiency:.2e} kg/year")
    print(f"   - Cost Efficiency:          {cost_efficiency:.2e} kg/$")
    print(f"   - Environmental Efficiency: {env_efficiency:.2e} kg/impact unit")
    
    print("\n4. TRANSPORTATION BREAKDOWN:")
    print(f"   - Total Rocket Launches:    {best_ind.total_launches:.0f}")
    print(f"   - Total Elevator Transport: {best_ind.total_lift:.2e} kg")
    print(f"   - Rocket Share of Mass:     {(best_ind.total_launches*Cap_rocket/best_ind.transported*100):.1f}%")
    print(f"   - Elevator Share of Mass:   {(best_ind.total_lift/best_ind.transported*100):.1f}%")
    
    # Check environmental thresholds
    print("\n5. ENVIRONMENTAL COMPLIANCE CHECK:")
    
    # Calculate annual environmental impact
    annual_launches = np.sum(R_best, axis=0)[:best_ind.ZT]
    avg_annual_launches = np.mean(annual_launches)
    
    # Critical threshold calculations
    alpha_avg = (solid_rocket_ratio * ENV_PARAMS['alpha_solid'] + 
                 (1 - solid_rocket_ratio) * ENV_PARAMS['alpha_liquid'])
    annual_env_threshold = E_CRIT / (ENV_PARAMS['M_Cl'] * alpha_avg * 100)  # Approximate
    
    if avg_annual_launches < annual_env_threshold:
        print(f"   ✅ PASS: Annual launches ({avg_annual_launches:.1f}) below threshold ({annual_env_threshold:.1f})")
        print(f"   Safety Factor: {annual_env_threshold/avg_annual_launches:.1f}x")
    else:
        print(f"   ⚠️  WARNING: Annual launches ({avg_annual_launches:.1f}) exceed threshold ({annual_env_threshold:.1f})")
        print(f"   Exceedance: {(avg_annual_launches/annual_env_threshold - 1)*100:.1f}%")
    
    print("\n" + "=" * 70)

def save_results_to_excel(best_ind, history):
    """Save all results to Excel file"""
    print("\nSaving results to Excel...")
    
    R_best = np.array(best_ind[:K*TMAX]).reshape(K, TMAX)
    E_best = np.array(best_ind[K*TMAX:]).reshape(P, TMAX)
    
    # Create comprehensive results workbook
    with pd.ExcelWriter('environmental_optimization_results_detailed.xlsx') as writer:
        # 1. Rocket launch schedule
        rocket_df = pd.DataFrame(R_best)
        rocket_df.columns = [f'Year_{i+1}' for i in range(TMAX)]
        rocket_df.index = [f'Rocket_Type_{i+1}' for i in range(K)]
        rocket_df.to_excel(writer, sheet_name='Rocket_Launch_Schedule')
        
        # 2. Elevator transport schedule
        elevator_df = pd.DataFrame(E_best)
        elevator_df.columns = [f'Year_{i+1}' for i in range(TMAX)]
        elevator_df.index = [f'Elevator_Channel_{i+1}' for i in range(P)]
        elevator_df.to_excel(writer, sheet_name='Elevator_Transport_Schedule')
        
        # 3. Annual summary
        annual_summary_data = {
            'Year': [f'Year_{i+1}' for i in range(best_ind.ZT)],
            'Rocket_Launches': np.sum(R_best[:, :best_ind.ZT], axis=0),
            'Rocket_Transport_kg': np.sum(R_best[:, :best_ind.ZT] * Cap_rocket, axis=0),
            'Elevator_Transport_kg': np.sum(E_best[:, :best_ind.ZT], axis=0),
            'Total_Transport_kg': np.sum(R_best[:, :best_ind.ZT] * Cap_rocket, axis=0) + np.sum(E_best[:, :best_ind.ZT], axis=0)
        }
        annual_df = pd.DataFrame(annual_summary_data)
        annual_df.to_excel(writer, sheet_name='Annual_Summary', index=False)
        
        # 4. Optimization history
        history_df = pd.DataFrame(history)
        history_df.to_excel(writer, sheet_name='Optimization_History', index=False)
        
        # 5. Final results summary
        summary_data = {
            'Metric': [
                'Total_Environmental_Impact_ZE',
                'Completion_Time_Years',
                'Total_Transported_Mass_kg',
                'Total_Rocket_Launches',
                'Total_Elevator_Transport_kg',
                'Financial_Cost_USD',
                'Environmental_Cost_USD',
                'Total_Cost_USD',
                'Environmental_Cost_Percentage',
                'Transport_Efficiency_kg_per_year',
                'Cost_Efficiency_kg_per_USD',
                'Environmental_Efficiency_kg_per_impact'
            ],
            'Value': [
                best_ind.ZE,
                best_ind.ZT,
                best_ind.transported,
                best_ind.total_launches,
                best_ind.total_lift,
                best_ind.financial_cost,
                best_ind.ZE * LAMBDA_ENV,
                best_ind.financial_cost + best_ind.ZE * LAMBDA_ENV,
                (best_ind.ZE * LAMBDA_ENV) / (best_ind.financial_cost + best_ind.ZE * LAMBDA_ENV) * 100,
                best_ind.transported / best_ind.ZT,
                best_ind.transported / (best_ind.financial_cost + best_ind.ZE * LAMBDA_ENV),
                best_ind.transported / best_ind.ZE if best_ind.ZE > 0 else 0
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Results_Summary', index=False)
        
        # 6. Environmental impact decomposition
        if best_ind.total_launches > 0:
            alpha_avg = (solid_rocket_ratio * ENV_PARAMS['alpha_solid'] + 
                         (1 - solid_rocket_ratio) * ENV_PARAMS['alpha_liquid'])
            
            env_decomp = {
                'Impact_Category': [
                    'Ozone_Layer_Depletion',
                    'Climate_Change_Black_Carbon',
                    'Aerosol_Effects',
                    'Other_Impacts'
                ],
                'Impact_Value': [
                    best_ind.total_launches * ENV_PARAMS['M_Cl'] * alpha_avg * ENV_WEIGHTS['ozone'],
                    ENV_PARAMS['beta'] * best_ind.total_launches * ENV_PARAMS['M_BC'] * ENV_PARAMS['T_res'] * ENV_WEIGHTS['climate'],
                    best_ind.total_launches * ENV_PARAMS['M_Al2O3'] * ENV_PARAMS['gamma'] * ENV_WEIGHTS['aerosol'],
                    best_ind.total_launches * ENV_PARAMS['E_other'] * ENV_WEIGHTS['other']
                ],
                'Percentage': [
                    ENV_WEIGHTS['ozone'] * 100,
                    ENV_WEIGHTS['climate'] * 100,
                    ENV_WEIGHTS['aerosol'] * 100,
                    ENV_WEIGHTS['other'] * 100
                ]
            }
            env_df = pd.DataFrame(env_decomp)
            env_df.to_excel(writer, sheet_name='Environmental_Decomposition', index=False)
    
    print("Results saved to: environmental_optimization_results_detailed.xlsx")


# ==============================
# Execute Main Program
# ==============================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ENVIRONMENTAL IMPACT OPTIMIZATION FOR SPACE TRANSPORTATION")
    print("=" * 70)
    print("\nInitializing optimization system...")
    
    optimal_solution = main()
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION PROCESS COMPLETE")
    print("=" * 70)
    print("\nAll visualizations have been saved as PNG files.")
    print("Detailed results have been exported to Excel.")
    print("\nThank you for using the Environmental Impact Optimization System!")
    print("=" * 70)