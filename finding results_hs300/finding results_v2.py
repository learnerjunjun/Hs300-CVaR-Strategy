# 导入包
import numpy as np
import pandas as pd
import datetime
import xlsxwriter
import warnings
import os
from scipy.optimize import minimize
import cvxpy as cp
from random import random, randint
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from scipy.optimize import Bounds
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, summation
import nlopt
from functools import partial
warnings.filterwarnings('ignore')

# hs300名单 2022.1.1之前
hs300 = pd.read_csv(r'../hs300_2005-2022/hs300_monthly.csv', dtype=object)
hs300 = hs300.drop('Unnamed: 0', axis=1)
hs300 = hs300.rename(columns={'stock': 'code'})
hs300['month'] = hs300['month'].astype(int)
#hs300收益率
Dataused_return=pd.read_excel(r'../data/国内指数月行情文件/IDX_Idxtrdmth.xlsx',dtype = object)
month_name_number = pd.DataFrame({'Month': hs300['month_details'].unique(), 'month':hs300['month'].unique()})
hs300_return_monthly = Dataused_return.loc[Dataused_return['Indexcd'] == '000300']
hs300_return = pd.merge(hs300_return_monthly, month_name_number, on='Month', how='right')
hs300_return = hs300_return.dropna()
hs300_return['month'] = hs300_return['month'].astype(int)
hs300_return['Idxrtn'] = hs300_return['Idxrtn'].astype(float)

## 定义参数类
# -- define a class including all parameters
class Para():
    method = 'LR'
    month_in_sample = range(82, 153 + 1)  # -- return 82~153 72 months
    month_test = range(154, 293 + 1)  # -- return 154~293 140 months
    percent_select = [0.3, 0.3]  # -- 30% positive samples, 30% negative samples
    percent_cv = 0.1  # -- percentage of cross validation samples 交叉验证的样本比例
    path_data = '../data/csv_02/'
    path_results = './results/'
    seed = 42  # -- random seed
    n_stock = 5166
    #以下都是需要调整的参数，包括数据量、轮动周期及选股数量
    # min=36,max=48,update=3,back=24，select=20的表现可以
    # train_start_test = 130  #测试集最早从第130个月开始 if min<45
    train_min_months = 69 #训练最少使用数据量
    train_start_month = 95 #训练开始时间 或者从94开始，避免回撤过大，年均收益率更高
    train_max_months = 48 #训练最多使用数据量
    train_update_months = 10 #训练轮动周期
    max_date = 284
    max_select = 46 #选股最多数量
    roll_back = 48 #决策集回溯地历史数据，从12个月开始调整，间隔周期为6个月，18个月长期收益表现可以
    CVaR_method = "minimize_cvar_scipy"
    # 优化算法
    # "differential_evolution", "genetic_algorithm", "simulated_annealing",
    # "bayesian_optimization", "particle_swarm_optimization"
    # "nlopt_optimization", "minimize_cvar_scipy"
    confidence_level = 0.95 #置信水平
    risk_free_rate = 0  # 无风险利率假设为0
    y_data = 'curr_return'  #'curr_return' ,'excess_return_curr','next_return','excess_return_next'
para = Para()

# 定义函数
# 基于信息系数大小的权重分配
# 与下一期收益率更高的因子有更高的权重
def assign_weight_by_coefficient(information_coefficient):
    # 计算IC值绝对值总和
    ic_sum = abs(information_coefficient['IC']).sum()
    # 更改阈值为0.03
    threshold = 0.03
    # 分配权重
    information_coefficient.loc[abs(information_coefficient['IC']) > threshold, 'Weight'] = 0.6 * (
            abs(information_coefficient['IC']) / ic_sum)
    information_coefficient.loc[abs(information_coefficient['IC']) <= threshold, 'Weight'] = 0.4 * (
            (threshold - abs(information_coefficient['IC'])) / (1 - ic_sum))
    return information_coefficient

# 优化器-1
def calculate_portfolio_weights(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    weights = cp.Variable(num_assets)
    # 定义目标函数
    objective = cp.Minimize(cp.quad_form(weights, cov_matrix))
    # 添加约束条件
    constraints = [weights >= 0, cp.sum(weights) == 1]
    # 创建问题并求解
    problem = cp.Problem(objective, constraints)
    problem.solve()
    # 获取最优权重
    result = weights.value
    return np.array(result)

# 优化器-2 非线性优化问题
def calculate_portfolio_returns(returns, weights):
    weights = np.array(weights)
    portfolio_returns = np.dot(weights.T, returns)
    return portfolio_returns
def calculate_cvar(portfolio_returns, confidence_level):
    var = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
    loss_function = -portfolio_returns
    cvar = var + (1 / (len(portfolio_returns) * (1 - confidence_level))) * np.sum(
        np.maximum(loss_function - var, 0))
    return cvar

def objective_function(weights, returns, confidence_level):
    portfolio_returns = calculate_portfolio_returns(returns, weights)
    cvar = calculate_cvar(portfolio_returns, confidence_level)
    return cvar
def constraint_function(weights):
    return np.sum(weights) - 1
def minimize_cvar_scipy(returns, confidence_level):
    n_stocks = returns.shape[0]
    initial_weights = np.random.rand(n_stocks)
    initial_weights = initial_weights / np.sum(initial_weights)
    bounds = [(0, 1) for _ in range(n_stocks)]
    constraints = [{'type': 'eq', 'fun': constraint_function}]
    result = minimize(objective_function, initial_weights, args=(returns, confidence_level),
                      bounds=bounds, constraints=constraints)
    min_cvar = result.fun
    optimal_weights = result.x
    return min_cvar, optimal_weights
# 差分进化算法优化
def differential_evolution(returns, confidence_level, population_size=50, max_generations=100,
                           F=0.5, CR=0.7):
    def clip_weights(weights):
        return np.clip(weights, 1e-9, 1.0 - 1e-9) / np.sum(np.clip(weights, 1e-9, 1.0 - 1e-9))
    n_stocks = returns.shape[0]
    population = np.random.rand(population_size, n_stocks)
    population = np.apply_along_axis(clip_weights, 1, population)  # 确保权重总和为1，且在（0，1）范围内
    for gen in range(max_generations):
        for i in range(population_size):
            target_vector = population[i]
            indices = list(range(population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant_vector = population[a] + F * (population[b] - population[c])
            crossover_mask = np.random.rand(n_stocks) < CR
            trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
            trial_vector = clip_weights(trial_vector)  # 确保权重总和为1，且在（0，1）范围内
            if objective_function(trial_vector, returns, confidence_level) < objective_function(target_vector, returns,
                                                                                                confidence_level):
                population[i] = trial_vector
    best_portfolio = min(population, key=lambda x: objective_function(x, returns, confidence_level))
    best_portfolio = clip_weights(best_portfolio)  # 确保权重总和为1，且在（0，1）范围内
    best_cvar = objective_function(best_portfolio, returns, confidence_level)
    return best_cvar, best_portfolio
# 定义遗传优化函数
def genetic_algorithm(returns, confidence_level, pop_size=50, generations=100, mutation_rate=0.1):
    def clip_weights(weights):
        clipped = np.clip(weights, 1e-9, 1.0 - 1e-9)  # 将权重限制在（0，1）范围内
        return clipped / np.sum(clipped)  # 确保权重总和为1
    def initialize_population(pop_size, n_stocks):
        population = []
        for _ in range(pop_size):
            weights = np.random.rand(n_stocks)
            weights /= np.sum(weights)  # 确保权重总和为1
            population.append(weights)
        return population
    def selection(population, scores, k=5):
        selected = []
        for _ in range(k):
            idx = randint(0, len(population) - 1)
            selected.append((population[idx], scores[idx]))
        selected.sort(key=lambda x: x[1])
        return [ind for ind, _ in selected[:k]]
    def crossover(parent1, parent2):
        split = randint(1, len(parent1) - 1)
        child = np.concatenate((parent1[:split], parent2[split:]), axis=0)
        return clip_weights(child)  # 使用裁剪函数确保权重范围和总和
    def mutation(child, mutation_rate=0.1):
        for i in range(len(child)):
            if random() < mutation_rate:
                child[i] = random()
        return clip_weights(child)  # 使用裁剪函数确保权重范围和总和
    n_stocks = returns.shape[0]
    population = initialize_population(pop_size, n_stocks)
    for gen in range(generations):
        scores = [objective_function(ind, returns, confidence_level) for ind in population]
        parents = selection(population, scores)
        next_gen = []
        while len(next_gen) < pop_size:
            idx1, idx2 = np.random.choice(len(parents), 2, replace=False)
            parent1, parent2 = parents[idx1], parents[idx2]
            offspring = crossover(parent1, parent2)
            offspring = mutation(offspring, mutation_rate)
            next_gen.append(offspring)
        population = next_gen
    best_portfolio = min(population, key=lambda x: objective_function(clip_weights(x), returns, confidence_level))
    best_portfolio = clip_weights(best_portfolio)  # 最终确认最佳投资组合的权重范围和总和
    best_cvar = objective_function(best_portfolio, returns, confidence_level)
    return best_cvar, best_portfolio
# 模拟退火算法
def simulated_annealing(returns, confidence_level, initial_temperature=100, final_temperature=1, max_iter=1000):
    def clip_weights(weights):
        clipped = np.clip(weights, 1e-9, 1.0 - 1e-9)  # 将权重限制在（0，1）范围内
        return clipped / np.sum(clipped)  # 确保权重总和为1
    n_stocks = returns.shape[0]
    current_state = np.random.rand(n_stocks)  # Initialize current state
    current_state /= np.sum(current_state) if np.sum(current_state) != 0 else 1  # Ensure sum is 1 (avoid division by 0)
    current_value = objective_function(current_state, returns, confidence_level)
    best_state = current_state
    best_value = current_value
    temperature = initial_temperature
    for i in range(max_iter):
        new_state = current_state + np.random.normal(0, 0.1, size=n_stocks)  # Generate new state
        new_state = clip_weights(new_state)  # 使用裁剪函数确保权重范围和总和
        new_value = objective_function(new_state, returns, confidence_level)
        delta = new_value - current_value
        if delta < 0 or np.random.rand() < np.exp(-delta / temperature):  # Accept new state
            current_state = new_state
            current_value = new_value
            if new_value < best_value:
                best_state = new_state
                best_value = new_value
        temperature = initial_temperature * (final_temperature / initial_temperature) ** (i / max_iter)  # Reduce temperature
    return best_value, best_state

# 贝叶斯优化函数
# Define the objective function for Bayesian optimization
def bayesian_optimization(returns, confidence_level):
    def clip_weights(weights):
        clipped = np.clip(weights, 1e-9, 1.0 - 1e-9)
        clipped /= np.sum(clipped)
        return clipped

    n_stocks = returns.shape[0]
    space = [Real(0.0, 1.0, name=f'w{i}') for i in range(n_stocks)]
    @use_named_args(space)
    def objective_function_wrapper(**params):
        weights = [params[f'w{i}'] for i in range(n_stocks)]
        weights = clip_weights(weights)
        portfolio_returns = calculate_portfolio_returns(returns, weights)
        return calculate_cvar(portfolio_returns, confidence_level)
    kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel)
    result = gp_minimize(
        objective_function_wrapper,
        dimensions=space,
        n_calls=20,
        random_state=para.seed,
        base_estimator=gp
    )
    best_cvar = result.fun  # Minimum CVaR value
    best_weights = clip_weights(result.x)
    return best_cvar, best_weights


def particle_swarm_optimization(returns, confidence_level, num_particles=50, max_iter=100, inertia=0.5, c1=2.0, c2=2.0):
    def clip_weights(weights):
        clipped = np.clip(weights, 1e-9, 1.0 - 1e-9)  # 将权重限制在（0，1）范围内
        return clipped / np.sum(clipped)  # 确保权重总和为1
    n_stocks = returns.shape[0]
    # Initialize particles and velocities
    particles = np.random.rand(num_particles, n_stocks)
    velocities = np.random.rand(num_particles, n_stocks)
    # Initialize the best-known positions and values for particles
    personal_best = particles.copy()
    personal_best_value = np.array([objective_function(particle, returns, confidence_level) for particle in particles])
    # Find global best position and value
    global_best_idx = np.argmin(personal_best_value)
    global_best = personal_best[global_best_idx]
    global_best_value = personal_best_value[global_best_idx]
    for _ in range(max_iter):
        for i in range(num_particles):
            # Update particle velocity
            velocities[i] = inertia * velocities[i] + c1 * np.random.rand() * (
                        personal_best[i] - particles[i]) + c2 * np.random.rand() * (global_best - particles[i])
            # Update particle position
            particles[i] = clip_weights(particles[i] + velocities[i])
            # Update personal best
            particle_value = objective_function(particles[i], returns, confidence_level)
            if particle_value < personal_best_value[i]:
                personal_best[i] = particles[i]
                personal_best_value[i] = particle_value
                # Update global best
                if particle_value < global_best_value:
                    global_best = particles[i]
                    global_best_value = particle_value
    return global_best_value, global_best


def nlopt_optimization(returns, confidence_level):
    def clip_weights(weights, n):
        clipped = np.clip(weights, 1e-9, 1.0 - 1e-9)  # 将权重限制在（0，1）范围内
        return clipped / np.sum(clipped)  # 确保权重总和为1
    n_stocks = returns.shape[0]
    # 定义目标函数
    def objective_function(weights, grad, returns=returns, confidence_level=confidence_level):
        portfolio_returns = calculate_portfolio_returns(returns, weights)
        cvar = calculate_cvar(portfolio_returns, confidence_level)
        return cvar
    opt = nlopt.opt(nlopt.LD_MMA, n_stocks)  # 选择优化算法
    opt.set_min_objective(objective_function)  # 设置最小化目标函数
    opt.set_lower_bounds(np.zeros(n_stocks))  # 设置下界为0
    opt.set_upper_bounds(np.ones(n_stocks))  # 设置上界为1
    opt.set_ftol_rel(1e-4)  # 设置相对容忍度
    initial_guess = np.random.rand(n_stocks)  # 初始化权重
    opt.set_initial_step(initial_guess * 0.1)  # 设置初始步长
    result = opt.optimize(initial_guess)  # 进行优化
    best_weights = clip_weights(result, n_stocks)  # 最终确认最佳权重的范围和总和
    best_value = objective_function(best_weights, None)  # 计算最优值
    return best_value, best_weights

# 只使用 scipy.optimize 进行优化
def minimize_cvar_scipy(returns, confidence_level):
    n_stocks = returns.shape[0]
    initial_weights = np.random.rand(n_stocks)
    initial_weights = initial_weights / np.sum(initial_weights)
    bounds = [(0, 1) for _ in range(n_stocks)]
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

    result = minimize(objective_function, initial_weights, args=(returns, confidence_level),
                      bounds=bounds, constraints=constraints)
    best_cvar = result.fun
    best_portfolio = result.x
    return best_cvar, best_portfolio
# minimize_cvar 函数
def minimize_cvar(returns, confidence_level):
    method = para.CVaR_method
    if method == "differential_evolution":
        min_cvar, optimal_weights = differential_evolution(returns, confidence_level)
    elif method == "genetic_algorithm":
        min_cvar, optimal_weights = genetic_algorithm(returns, confidence_level)
    elif method == "simulated_annealing":
        min_cvar, optimal_weights = simulated_annealing(returns, confidence_level)
    elif method == "bayesian_optimization":
        min_cvar, optimal_weights = bayesian_optimization(returns, confidence_level)
    elif method == "particle_swarm_optimization":
        min_cvar, optimal_weights = particle_swarm_optimization(returns, confidence_level)
    elif method == "nlopt_optimization":
        min_cvar, optimal_weights = nlopt_optimization(returns, confidence_level)
    elif method == "minimize_cvar_scipy":
        min_cvar, optimal_weights = minimize_cvar_scipy(returns, confidence_level)
    else:
        raise ValueError("Invalid CVaR_method specified in para")
    optimal_weights = np.array(optimal_weights)
    return min_cvar, optimal_weights

# 计算回撤
def calculate_drawdown(data):
    data['peak_value'] = data['compound_value'].cummax()
    data['drawdown'] = (data['peak_value'] - data['compound_value']) / data['peak_value']
    max_drawdown = data['drawdown'].max()
    return data, max_drawdown
# 计算年化收益率和波动率
def calculate_annualized_metrics(data):
    monthly_returns = data['return']  # 假设'return'列包含月度收益率
    # 计算均值
    mean_monthly_return = monthly_returns.mean()
    # 使用均值计算年化收益率
    annualized_return = (1 + mean_monthly_return) ** 12 - 1  # 年化收益率
    annualized_volatility = np.std(monthly_returns) * np.sqrt(12)  # 年化波动率
    return annualized_return, annualized_volatility
# 计算夏普比率（使用指数形式计算收益率）
def calculate_sharpe_ratio(annualized_return, annualized_volatility,risk_free_rate):
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
    return sharpe_ratio
# 计算信息比率（使用指数形式计算收益率）
def calculate_information_ratio(data_strategy, data_benchmark, risk_free_rate):
    annualized_return_strategy, annualized_volatility_strategy = calculate_annualized_metrics(data_strategy)
    annualized_return_benchmark, annualized_volatility_benchmark = calculate_annualized_metrics(data_benchmark)
    excess_returns = data_strategy['return'] - data_benchmark['return']
    annualized_volatility_excess = np.std(excess_returns) * np.sqrt(12)  # 月度超额收益波动率乘以根号12得到年化超额波动率
    information_ratio = (
                                annualized_return_strategy - annualized_return_benchmark - risk_free_rate
                        ) / annualized_volatility_excess if annualized_volatility_excess != 0 else 0
    return information_ratio
# 计算收益回撤比（使用指数形式计算收益率）
def calculate_return_drawdown_ratio(index_annual_return, max_drawdown):
    return index_annual_return / max_drawdown if max_drawdown != 0 else 0


# def rolling_train(train_data_min_months,train_update_months,start_date,roll_back,confidence_level):
# 训练部分
train_data_min_months = para.train_min_months  # 每次模型训练所用数据最少不低于
train_data_max_months = para.train_max_months  # 每次模型训练所用数据最大不超过
train_update_months = para.train_update_months  # 设置更新周期
start_date = para.train_start_month  # 第一次滚动训练开始日期
end_date = start_date + train_data_min_months - 1  # 第一次滚动训练结束日期

# 创建一个空的DataFrame
return_data_combined_coef = pd.DataFrame()
weights_data_combined_coef = pd.DataFrame()
return_data_combined_ic = pd.DataFrame()
weights_data_combined_ic = pd.DataFrame()
return_data_combined_corr = pd.DataFrame()
weights_data_combined_corr = pd.DataFrame()
return_data_combined_cvar_coef = pd.DataFrame()
weights_data_combined_cvar_coef = pd.DataFrame()
return_data_combined_cvar_ic = pd.DataFrame()
weights_data_combined_cvar_ic = pd.DataFrame()
return_data_combined_cvar_corr = pd.DataFrame()
weights_data_combined_cvar_corr = pd.DataFrame()
return_data_combined_hs300 = pd.DataFrame()

while end_date <= para.max_date:
    period_train = range(start_date, end_date + 1)
    ## 生成样本内数据集
    # -- generate in-sample data
    for i_month in period_train:
        # -- load csv
        file_name_curr = para.path_data + str(i_month) + '.csv'
        file_name_next = para.path_data + str(i_month+1) + '.csv'
        data_curr_month = pd.read_csv(file_name_curr, header=0)  # 设置表头
        data_next_month = pd.read_csv(file_name_next, header=0)  # 设置表头
        para.n_stock = data_curr_month.shape[0]
        # -- merge
        if i_month == period_train[0]:  # -- first month
            data_in_sample_all = data_curr_month
            data_in_sample_all_next = data_next_month
        else:
            data_in_sample_all = pd.concat((data_in_sample_all, data_curr_month), axis=0)
            data_in_sample_all_next = pd.concat((data_in_sample_all_next, data_next_month), axis=0)
        data_in_sample_all['code'] = data_in_sample_all['stock'].str.replace('[^\d]', '', regex=True)
        data_in_sample_all_next['code'] = data_in_sample_all['stock'].str.replace('[^\d]', '', regex=True)
        # 筛选出名单内的hs300
        data_in_sample = pd.merge(hs300, data_in_sample_all, on=['code', 'month'], how='inner')
        data_in_sample_next = pd.merge(hs300, data_in_sample_all_next, on=['code', 'month'], how='inner')
        # -- remove nan
        data_in_sample = data_in_sample.dropna(axis=0)
        data_in_sample_next = data_in_sample_next.dropna(axis=0)
    data_in_sample[para.y_data] = data_in_sample[para.y_data].astype(float)
    data_in_sample_next[para.y_data] = data_in_sample_next[para.y_data].astype(float)
    # 样本内数据集
    # -- generate in-sample data
    X_in_sample = data_in_sample.loc[:, 'EP':'bias']  # 提取数据
    # -- regression
    if para.method in ['LR']:
        y_in_sample = data_in_sample.loc[:, para.y_data]
    ## 划分训练集和验证集
    # -- generate train and cv data
    from sklearn.model_selection import train_test_split

    # 随机拆分数据为train set训练集和test set测试集
    # X:要划分的样本特征集（输入的信息）
    # y:需要划分的样本结果（输出结果）
    # test_size:样本占比，测试集在总数中的百分比（小数表示）
    # random_state:随机数种子，对于模型分割，必须用同一随机数种子，保证每次随机分割后数据集不变。
    if para.percent_cv > 0:
        X_train, X_cv, y_train, y_cv = train_test_split(X_in_sample, y_in_sample, test_size=para.percent_cv,
                                                        random_state=para.seed)
    else:
        X_train, y_train = X_in_sample.copy(), y_in_sample.copy()
    # 有监督，连续型数据
    # -- linear regression

    from sklearn import linear_model
    from sklearn.decomposition import PCA

    if para.method in ['LR']:
        model = linear_model.LinearRegression(fit_intercept=True)  # 计算偏置（截距）
        # -- regression
        model.fit(X_train, y_train)
        y_score_train = model.predict(X_train)
        if para.percent_cv > 0:
            y_score_cv = model.predict(X_cv)

    # 因子信息系数
    columns = ['code', 'month']
    factors_columns = X_in_sample.columns
    for i in factors_columns:
        columns.append(i)
    # 以 'month' 和 'code' 为基准合并两个 DataFrame，选择合适的 how 参数，根据实际需求选择合适的连接方式
    data_in_sample_next_y = data_in_sample_next[['month', 'code', para.y_data]]
    data_in_sample_x = data_in_sample.loc[:, columns]
    merged_data = pd.merge(data_in_sample_next_y, data_in_sample_x, on=['month', 'code'], how='inner')
    # 提取合并后的列
    X_in_sample_ic = merged_data.loc[:, 'EP':'bias']
    y_in_sample_ic = merged_data[para.y_data]
    ic_values = {}  # 存储每个因子的IC值
    for column in list(X_in_sample_ic.columns):
        factor_values = X_in_sample_ic[column]
        correlation = np.corrcoef(factor_values, y_in_sample_ic)[0, 1]
        ic_values[column] = correlation
    # 创建包含因子和IC值的DataFrame
    information_coefficient = pd.DataFrame(list(ic_values.items()), columns=['Factor', 'IC'])
    #基于信息系数
    weighted_factors_coefficient = assign_weight_by_coefficient(information_coefficient)
    factor_weights_ic = np.array(weighted_factors_coefficient['Weight'])

    # 基于回归系数
    # 获取回归系数
    coefficients = model.coef_
    # 计算因子权重
    factor_weights_coef = abs(coefficients) / sum(abs(coefficients))

    # 基于因子数据集的特征值，特征值大的权重高
    covariance_matrix = X_in_sample.cov()
    # 计算特征值和特征向量
    eigenvalues, _ = np.linalg.eig(covariance_matrix)
    # 使用特征值作为权重
    weights = eigenvalues / np.sum(eigenvalues)  # 归一化
    # 找到最高的两个因子的索引
    top_two_indices = np.argsort(weights)[-2:]
    # 计算最高的两个因子的权重总和
    sum_top_weights = np.sum(weights[top_two_indices])
    # 将最高的两个因子的权重设置为20%，按比例分配
    weights[top_two_indices] = 0.2 * (weights[top_two_indices] / sum_top_weights)
    # 计算除最高的两个因子外其他因子的总权重
    other_weights = np.sum(weights) - np.sum(weights[top_two_indices])
    # 将其他因子的权重按比例调整为80%
    weights[:-2] = weights[:-2] * (0.8 * other_weights / np.sum(weights[:-2]))
    # 将调整后的权重赋值给原始变量
    weights = weights  # 最终的变量名不变
    # 创建权重分配的 DataFrame
    factor_weights = pd.DataFrame({'Factor': X_in_sample.columns, 'Weight': weights})
    factor_weights_corr = np.array(factor_weights['Weight'])

    # 样本外预测
    test_date_start = end_date + 1
    test_date_end = end_date + para.train_update_months
    period_test = range(test_date_start, test_date_end + 1)
    combined_y_pred_return = pd.DataFrame()
    combined_y_curr_return = pd.DataFrame()
    for i_month in period_test:
        # -- load
        file_name = para.path_data + str(i_month) + '.csv'
        data_curr_month = pd.read_csv(file_name, header=0)
        # --hs300
        data_curr_month['code'] = data_curr_month['stock'].str.replace('[^\d]', '', regex=True)
        data_curr_month = pd.merge(hs300, data_curr_month, on=['code', 'month'], how='inner')
        # -- remove nan
        data_curr_month = data_curr_month.dropna(axis=0)
        # -- generate X
        X_curr_month = data_curr_month.loc[:, 'EP':'bias']
        # -- pca
        pca = PCA(n_components = 16)
        X_in_sample = pca.fit_transform(X_in_sample)
        X_curr_month = pca.fit_transform(X_curr_month)
        # -- pca_train
        model_pca = linear_model.LinearRegression(fit_intercept=True)
        model_pca.fit(X_in_sample, y_in_sample)
        # -- linear regression
        if para.method in ['LR']:
            y_score_curr_month = model_pca.predict(X_curr_month)
        y_pred_return = pd.DataFrame(
            {'month': data_curr_month['month'], 'code': data_curr_month['code'], 'pred_return': y_score_curr_month})
        combined_y_pred_return = pd.concat([combined_y_pred_return, y_pred_return], axis=0)
        # -- curr_return
        y_curr_return = pd.DataFrame({'month': data_curr_month['month'], 'code': data_curr_month['code'],
                                      'curr_return': data_curr_month[para.y_data]})
        combined_y_curr_return = pd.concat([combined_y_curr_return, y_curr_return], axis=0)
    combined_y_pred_return = pd.pivot_table(combined_y_pred_return, values='pred_return', index='code',
                                            columns=['month'])
    combined_y_curr_return = pd.pivot_table(combined_y_curr_return, values='curr_return', index='code',
                                            columns=['month'])
    if para.method in ['LR']:
        y_train.index = range(len(y_train))
        y_score_train = pd.Series(y_score_train)
        print('training set, cc = %.2f' % y_train.corr(y_score_train))
        if para.percent_cv > 0:
            y_cv.index = range(len(y_cv))
            y_score_cv = pd.Series(y_score_cv)
            print('cv set, cc = %.2f' % y_cv.corr(y_score_cv))
        for i_month in period_test:
            y_true_curr_month = pd.Series(combined_y_curr_return[i_month])
            y_score_curr_month = pd.Series(combined_y_pred_return[i_month])
            print('testing set, month %d, cc = %.2f' % (i_month, y_true_curr_month.corr(y_score_curr_month)))

    max_select = para.max_select  # 最长的数据长度
    # 创建一个空的DataFrame来存储最优投资组合权重
    portfolio_weights_df_coef = pd.DataFrame()
    portfolio_weights_df_ic = pd.DataFrame()
    portfolio_weights_df_corr = pd.DataFrame()
    portfolio_weights_df_cvar_coef = pd.DataFrame()
    portfolio_weights_df_cvar_ic = pd.DataFrame()
    portfolio_weights_df_cvar_corr = pd.DataFrame()
    # 创建一个空的DataFrame
    portfolio_return_data_coef = pd.DataFrame(columns=['month', 'return'])
    portfolio_return_data_ic = pd.DataFrame(columns=['month', 'return'])
    portfolio_return_data_corr = pd.DataFrame(columns=['month', 'return'])
    portfolio_return_data_cvar_coef = pd.DataFrame(columns=['month', 'return'])
    portfolio_return_data_cvar_ic = pd.DataFrame(columns=['month', 'return'])
    portfolio_return_data_cvar_corr = pd.DataFrame(columns=['month', 'return'])
    return_data_hs300 = pd.DataFrame(columns=['month', 'return'])
    for i_month_1 in period_test:
        # -- load csv
        file_name_1 = para.path_data + str(i_month_1 - 1) + '.csv'
        file_name_2 = para.path_data + str(i_month_1) + '.csv'
        data_last_month = pd.read_csv(file_name_1, header=0)  # 设置表头
        data_curr_month = pd.read_csv(file_name_2, header=0)  # 设置表头
        data_last_month['code'] = data_last_month['stock'].str.replace('[^\d]', '', regex=True)
        data_curr_month['code'] = data_curr_month['stock'].str.replace('[^\d]', '', regex=True)
        # --hs300
        data_last_month = pd.merge(hs300, data_last_month, on=['code', 'month'], how='inner')
        data_last_month[para.y_data] = data_last_month[para.y_data].astype(float)
        data_curr_month = pd.merge(hs300, data_curr_month, on=['code', 'month'], how='inner')
        data_curr_month[para.y_data] = data_curr_month[para.y_data].astype(float)
        # -- remove nan
        data_last_month = data_last_month.dropna(axis=0)
        data_curr_month = data_curr_month.dropna(axis=0)
        # 打分法筛选出股票
        data_for_score_x = data_last_month.copy()
        data_for_score_x = data_for_score_x.drop(labels=['month',para.y_data], axis=1)
        data_for_score_y = data_curr_month[['month', 'code', para.y_data]]
        data_for_score = pd.merge(data_for_score_y, data_for_score_x, on=['code'], how='inner')
        X_for_score = data_for_score.loc[:, 'EP':'bias']  # 提取数据
        y_curr_month = pd.DataFrame(
            {'month': data_for_score['month'], 'code': data_for_score['code'], 'curr_return': data_for_score[para.y_data]})
        y_curr_month.set_index('code', inplace=True)
        # 计算每只股票的打分
        scores_coef = X_for_score.dot(factor_weights_coef)
        scores_ic = X_for_score.dot(factor_weights_ic)
        scores_corr = X_for_score.dot(factor_weights_corr)
        n = max_select + 10

        selected_stocks_coef = scores_coef.nlargest(n)
        selected_stocks_ic = scores_ic.nlargest(n)
        selected_stocks_corr = scores_corr.nlargest(n)
        # 添加month和stock列
        selected_stocks_coef = pd.DataFrame(
            {'code': data_for_score.loc[selected_stocks_coef.index, 'code'], 'score': selected_stocks_coef.values})
        selected_stocks_coef.set_index('code', inplace=True)
        selected_stocks_ic = pd.DataFrame(
            {'code': data_for_score.loc[selected_stocks_ic.index, 'code'], 'score': selected_stocks_ic.values})
        selected_stocks_ic.set_index('code', inplace=True)
        selected_stocks_corr = pd.DataFrame(
            {'code': data_for_score.loc[selected_stocks_corr.index, 'code'], 'score': selected_stocks_corr.values})
        selected_stocks_corr.set_index('code', inplace=True)

        # 整合历史与预测数据
        period_select = range(test_date_start - para.roll_back, i_month_1 + 1)
        combined_y_curr_return_past = pd.DataFrame()
        for i_month_2 in period_select:
            # -- load
            file_name = para.path_data + str(i_month_2) + '.csv'
            data_curr_month = pd.read_csv(file_name, header=0)
            # --hs300
            data_curr_month['code'] = data_curr_month['stock'].str.replace('[^\d]', '', regex=True)
            data_curr_month = pd.merge(hs300, data_curr_month, on=['code', 'month'], how='inner')
            data_curr_month[para.y_data] = data_curr_month[para.y_data].astype(float)
            # -- remove nan
            data_curr_month = data_curr_month.dropna(axis=0)
            # -- curr_return
            y_curr_return_past = pd.DataFrame({'month': data_curr_month['month'], 'code': data_curr_month['code'],
                                               'curr_return': data_curr_month[para.y_data]})
            combined_y_curr_return_past = pd.concat([combined_y_curr_return_past, y_curr_return_past], axis=0)

        combined_y_curr_return_past = pd.pivot_table(combined_y_curr_return_past, values='curr_return', index='code',
                                                     columns=['month'])
        combined_return_data = combined_y_curr_return_past.copy()
        combined_return_data[i_month_1] = combined_y_pred_return[i_month_1]

        # 筛选出打分法得到的股票
        top_select_stocks_idx_coef = selected_stocks_coef.index[:max_select]
        top_select_stocks_return_coef = combined_return_data.loc[
            combined_return_data.index.intersection(top_select_stocks_idx_coef)]
        top_select_stocks_return_coef = top_select_stocks_return_coef.dropna()
        top_select_stocks_idx_ic = selected_stocks_ic.index[:max_select]
        top_select_stocks_return_ic = combined_return_data.loc[
            combined_return_data.index.intersection(top_select_stocks_idx_ic)]
        top_select_stocks_return_ic = top_select_stocks_return_ic.dropna()
        top_select_stocks_idx_corr = selected_stocks_corr.index[:max_select]
        top_select_stocks_return_corr = combined_return_data.loc[
            combined_return_data.index.intersection(top_select_stocks_idx_corr)]
        top_select_stocks_return_corr = top_select_stocks_return_corr.dropna()

        # 计算收益率的协方差矩阵
        cov_matrix_coef = top_select_stocks_return_coef.T.cov()
        cov_matrix_ic = top_select_stocks_return_ic.T.cov()
        cov_matrix_corr = top_select_stocks_return_corr.T.cov()
        # 获取 i_month_1 对应的预测收益和股票代码
        mean_returns_coef = top_select_stocks_return_coef.loc[:, i_month_1]
        stock_codes_coef = top_select_stocks_return_coef.index.tolist()
        mean_returns_ic = top_select_stocks_return_ic.loc[:, i_month_1]
        stock_codes_ic = top_select_stocks_return_ic.index.tolist()
        mean_returns_corr = top_select_stocks_return_corr.loc[:, i_month_1]
        stock_codes_corr = top_select_stocks_return_corr.index.tolist()

        # 使用均值-方差模型计算最优投资组合权重
        portfolio_weights_coef = calculate_portfolio_weights(mean_returns_coef, cov_matrix_coef)
        portfolio_weights_ic = calculate_portfolio_weights(mean_returns_ic, cov_matrix_ic)
        portfolio_weights_corr = calculate_portfolio_weights(mean_returns_corr, cov_matrix_corr)

        # 使用CVaR计算最优权重
        confidence_level = para.confidence_level
        returns_cvar_coef = top_select_stocks_return_coef
        returns_cvar_ic = top_select_stocks_return_ic
        returns_cvar_corr = top_select_stocks_return_corr
        cvar_coef, portfolio_weights_cvar_coef = minimize_cvar(returns_cvar_coef, confidence_level)
        cvar_ic, portfolio_weights_cvar_ic = minimize_cvar(returns_cvar_ic, confidence_level)
        cvar_corr, portfolio_weights_cvar_corr = minimize_cvar(returns_cvar_corr, confidence_level)

        # 计算组合收益
        y_curr_month_return_coef = y_curr_month.loc[y_curr_month.index.intersection(stock_codes_coef)]['curr_return']
        y_curr_month_return_ic = y_curr_month.loc[y_curr_month.index.intersection(stock_codes_ic)]['curr_return']
        y_curr_month_return_corr = y_curr_month.loc[y_curr_month.index.intersection(stock_codes_corr)]['curr_return']

        portfolio_return_coef = np.dot(portfolio_weights_coef.T, y_curr_month_return_coef)
        portfolio_return_ic = np.dot(portfolio_weights_ic.T, y_curr_month_return_ic)
        portfolio_return_corr = np.dot(portfolio_weights_corr.T, y_curr_month_return_corr)
        portfolio_return_cvar_coef = np.dot(portfolio_weights_cvar_coef.T, y_curr_month_return_coef)
        portfolio_return_cvar_ic = np.dot(portfolio_weights_cvar_ic.T, y_curr_month_return_ic)
        portfolio_return_cvar_corr = np.dot(portfolio_weights_cvar_corr.T, y_curr_month_return_corr)
        hs300_idxrtn = hs300_return.loc[hs300_return['month'] == i_month_1, 'Idxrtn'].item()

        # 将收益数据添加到DataFrame中
        row_coef = {'month': i_month_1, 'return': portfolio_return_coef,'Month':data_for_score['month_details'].unique()}
        portfolio_return_data_coef = pd.concat([portfolio_return_data_coef, pd.DataFrame(row_coef, index=[0])],
                                               ignore_index=True)
        row_ic = {'month': i_month_1, 'return': portfolio_return_ic,'Month':data_for_score['month_details'].unique()}
        portfolio_return_data_ic = pd.concat([portfolio_return_data_ic, pd.DataFrame(row_ic, index=[0])],
                                             ignore_index=True)
        row_corr = {'month': i_month_1, 'return': portfolio_return_corr,'Month':data_for_score['month_details'].unique()}
        portfolio_return_data_corr = pd.concat([portfolio_return_data_corr, pd.DataFrame(row_corr, index=[0])],
                                               ignore_index=True)
        row_cvar_coef = {'month': i_month_1, 'return': portfolio_return_cvar_coef,'Month':data_for_score['month_details'].unique()}
        portfolio_return_data_cvar_coef = pd.concat([portfolio_return_data_cvar_coef, pd.DataFrame(row_cvar_coef, index=[0])],
                                               ignore_index=True)
        row_cvar_ic = {'month': i_month_1, 'return': portfolio_return_cvar_ic,'Month':data_for_score['month_details'].unique()}
        portfolio_return_data_cvar_ic = pd.concat([portfolio_return_data_cvar_ic, pd.DataFrame(row_cvar_ic, index=[0])],
                                               ignore_index=True)
        row_cvar_corr = {'month': i_month_1, 'return': portfolio_return_cvar_corr,'Month':data_for_score['month_details'].unique()}
        portfolio_return_data_cvar_corr = pd.concat([portfolio_return_data_cvar_corr, pd.DataFrame(row_cvar_corr, index=[0])],
                                               ignore_index=True)
        row_hs300 = {'month': i_month_1, 'return': hs300_idxrtn,'Month':data_for_score['month_details'].unique()}
        return_data_hs300 = pd.concat([return_data_hs300, pd.DataFrame(row_hs300, index=[0])],
                                               ignore_index=True)

        # 使用NaN将数据补齐至长度为max_select
        stock_codes_coef = np.concatenate((stock_codes_coef, np.full(max_select - len(stock_codes_coef), np.nan)))
        stock_codes_ic = np.concatenate((stock_codes_ic, np.full(max_select - len(stock_codes_ic), np.nan)))
        stock_codes_corr = np.concatenate((stock_codes_corr, np.full(max_select - len(stock_codes_corr), np.nan)))
        portfolio_weights_coef = np.concatenate(
            (portfolio_weights_coef, np.full(max_select - len(portfolio_weights_coef), np.nan)))
        portfolio_weights_ic = np.concatenate(
            (portfolio_weights_ic, np.full(max_select - len(portfolio_weights_ic), np.nan)))
        portfolio_weights_corr = np.concatenate(
            (portfolio_weights_corr, np.full(max_select - len(portfolio_weights_corr), np.nan)))
        portfolio_weights_cvar_coef = np.concatenate(
            (portfolio_weights_cvar_coef, np.full(max_select - len(portfolio_weights_cvar_coef), np.nan)))
        portfolio_weights_cvar_ic = np.concatenate(
            (portfolio_weights_cvar_ic, np.full(max_select - len(portfolio_weights_cvar_ic), np.nan)))
        portfolio_weights_cvar_corr = np.concatenate(
            (portfolio_weights_cvar_corr, np.full(max_select - len(portfolio_weights_cvar_corr), np.nan)))

        # 将最优投资组合权重和股票代码添加到DataFrame中
        portfolio_weights_df_coef[str(i_month_1) + '_code'] = stock_codes_coef
        portfolio_weights_df_coef[str(i_month_1)] = portfolio_weights_coef
        portfolio_weights_df_ic[str(i_month_1) + '_code'] = stock_codes_ic
        portfolio_weights_df_ic[str(i_month_1)] = portfolio_weights_ic
        portfolio_weights_df_corr[str(i_month_1) + '_code'] = stock_codes_corr
        portfolio_weights_df_corr[str(i_month_1)] = portfolio_weights_corr
        portfolio_weights_df_cvar_coef[str(i_month_1) + '_code'] = stock_codes_coef
        portfolio_weights_df_cvar_coef[str(i_month_1)] = portfolio_weights_cvar_coef
        portfolio_weights_df_cvar_ic[str(i_month_1) + '_code'] = stock_codes_ic
        portfolio_weights_df_cvar_ic[str(i_month_1)] = portfolio_weights_cvar_ic
        portfolio_weights_df_cvar_corr[str(i_month_1) + '_code'] = stock_codes_corr
        portfolio_weights_df_cvar_corr[str(i_month_1)] = portfolio_weights_cvar_corr

    return_data_combined_coef = pd.concat([return_data_combined_coef, portfolio_return_data_coef], ignore_index=True)
    weights_data_combined_coef = pd.concat([weights_data_combined_coef, portfolio_weights_df_coef], ignore_index=True,axis=1)
    return_data_combined_ic = pd.concat([return_data_combined_ic, portfolio_return_data_ic], ignore_index=True)
    weights_data_combined_ic = pd.concat([weights_data_combined_ic, portfolio_weights_df_ic], ignore_index=True,axis=1)
    return_data_combined_corr = pd.concat([return_data_combined_corr, portfolio_return_data_corr], ignore_index=True)
    weights_data_combined_corr = pd.concat([weights_data_combined_corr, portfolio_weights_df_corr], ignore_index=True,axis=1)
    return_data_combined_cvar_coef = pd.concat([return_data_combined_cvar_coef, portfolio_return_data_cvar_coef], ignore_index=True)
    weights_data_combined_cvar_coef = pd.concat([weights_data_combined_cvar_coef, portfolio_weights_df_cvar_coef], ignore_index=True,axis=1)
    return_data_combined_cvar_ic = pd.concat([return_data_combined_cvar_ic, portfolio_return_data_cvar_ic], ignore_index=True)
    weights_data_combined_cvar_ic = pd.concat([weights_data_combined_cvar_ic, portfolio_weights_df_cvar_ic], ignore_index=True,axis=1)
    return_data_combined_cvar_corr = pd.concat([return_data_combined_cvar_corr, portfolio_return_data_cvar_corr], ignore_index=True)
    weights_data_combined_cvar_corr = pd.concat([weights_data_combined_cvar_corr, portfolio_weights_df_cvar_corr], ignore_index=True,axis=1)
    return_data_combined_hs300 = pd.concat([return_data_combined_hs300 , return_data_hs300],ignore_index=True)

    # -- evaluation
    # 计算年化收益率、波动率及夏普比率
    risk_free_rate = para.risk_free_rate  # 无风险利率假设为0
    ann_return_coef = (1 + np.mean(
        return_data_combined_coef[return_data_combined_coef['month'].isin(period_test)]['return'])) ** 12 - 1
    ann_vol_coef = np.std(
        return_data_combined_coef[return_data_combined_coef['month'].isin(period_test)]['return']) * np.sqrt(12)
    sharpe_ratio_coef = (ann_return_coef - risk_free_rate) / ann_vol_coef

    ann_return_ic = (1 + np.mean(
        return_data_combined_ic[return_data_combined_ic['month'].isin(period_test)]['return'])) ** 12 - 1
    ann_vol_ic = np.std(
        return_data_combined_ic[return_data_combined_ic['month'].isin(period_test)]['return']) * np.sqrt(12)
    sharpe_ratio_ic = (ann_return_ic - risk_free_rate) / ann_vol_ic

    ann_return_corr = (1 + np.mean(
        return_data_combined_corr[return_data_combined_corr['month'].isin(period_test)]['return'])) ** 12 - 1
    ann_vol_corr = np.std(
        return_data_combined_corr[return_data_combined_corr['month'].isin(period_test)]['return']) * np.sqrt(12)
    sharpe_ratio_corr = (ann_return_corr - risk_free_rate) / ann_vol_corr

    ann_return_cvar_coef = (1 + np.mean(
        return_data_combined_cvar_coef[return_data_combined_cvar_coef['month'].isin(period_test)]['return'])) ** 12 - 1
    ann_vol_cvar_coef = np.std(
        return_data_combined_cvar_coef[return_data_combined_cvar_coef['month'].isin(period_test)]['return']) * np.sqrt(
        12)
    sharpe_ratio_cvar_coef = (ann_return_cvar_coef - risk_free_rate) / ann_vol_cvar_coef

    ann_return_cvar_ic = (1 + np.mean(
        return_data_combined_cvar_ic[return_data_combined_cvar_ic['month'].isin(period_test)]['return'])) ** 12 - 1
    ann_vol_cvar_ic = np.std(
        return_data_combined_cvar_ic[return_data_combined_cvar_ic['month'].isin(period_test)]['return']) * np.sqrt(12)
    sharpe_ratio_cvar_ic = (ann_return_cvar_ic - risk_free_rate) / ann_vol_cvar_ic

    ann_return_cvar_corr = (1 + np.mean(
        return_data_combined_cvar_corr[return_data_combined_cvar_corr['month'].isin(period_test)]['return'])) ** 12 - 1
    ann_vol_cvar_corr = np.std(
        return_data_combined_cvar_corr[return_data_combined_cvar_corr['month'].isin(period_test)]['return']) * np.sqrt(
        12)
    sharpe_ratio_cvar_corr = (ann_return_cvar_corr - risk_free_rate) / ann_vol_cvar_corr

    ann_return_hs300 = (1 + np.mean(
        return_data_combined_hs300[return_data_combined_hs300['month'].isin(period_test)]['return'])) ** 12 - 1
    ann_vol_hs300 = np.std(
        return_data_combined_hs300[return_data_combined_hs300['month'].isin(period_test)]['return']) * np.sqrt(
        12)
    sharpe_ratio_hs300 = (ann_return_hs300 - risk_free_rate) / ann_vol_hs300

    print('回归系数')
    print('annual return = %.2f' % ann_return_coef)
    print('annual volatility = %.2f' % ann_vol_coef)
    print('sharpe ratio = %.2f' % sharpe_ratio_coef)
    print('CVaR_coef')
    print('annual return = %.2f' % ann_return_cvar_coef)
    print('annual volatility = %.2f' % ann_vol_cvar_coef)
    print('sharpe ratio = %.2f' % sharpe_ratio_cvar_coef)
    print('信息系数')
    print('annual return = %.2f' % ann_return_ic)
    print('annual volatility = %.2f' % ann_vol_ic)
    print('sharpe ratio = %.2f' % sharpe_ratio_ic)
    print('CVaR_ic')
    print('annual return = %.2f' % ann_return_cvar_ic, )
    print('annual volatility = %.2f' % ann_vol_cvar_ic)
    print('sharpe ratio = %.2f' % sharpe_ratio_cvar_ic)
    print('协方差矩阵')
    print('annual return = %.2f' % ann_return_corr)
    print('annual volatility = %.2f' % ann_vol_corr)
    print('sharpe ratio = %.2f' % sharpe_ratio_corr)
    print('CVaR_corr')
    print('annual return = %.2f' % ann_return_cvar_corr)
    print('annual volatility = %.2f' % ann_vol_cvar_corr)
    print('sharpe ratio = %.2f' % sharpe_ratio_cvar_corr)
    print('hs300')
    print('annual return = %.2f' % ann_return_hs300)
    print('annual volatility = %.2f' % ann_vol_hs300)
    print('sharpe ratio = %.2f' % sharpe_ratio_hs300)

    ##数据集滚动
    # 更新日期
    start_date += train_update_months
    # 限制训练数据的时间范围
    end_date = min(start_date + train_data_min_months - 1, para.max_date)
    # 若接近最大日期，则终止训练
    if end_date + train_update_months >= para.max_date:
        break

#计算累计收益
return_data_combined_coef['compound_value'] = (return_data_combined_coef['return']+1).cumprod()
return_data_combined_cvar_coef['compound_value'] = (return_data_combined_cvar_coef['return']+1).cumprod()
return_data_combined_ic['compound_value'] = (return_data_combined_ic['return']+1).cumprod()
return_data_combined_cvar_ic['compound_value'] = (return_data_combined_cvar_ic['return']+1).cumprod()
return_data_combined_corr['compound_value'] = (return_data_combined_corr['return']+1).cumprod()
return_data_combined_cvar_corr['compound_value'] = (return_data_combined_cvar_corr['return']+1).cumprod()
return_data_combined_hs300['compound_value'] = (return_data_combined_hs300['return']+1).cumprod()

# 储存所有数据列
data_columns = [
    'return_data_combined_coef',
    'return_data_combined_cvar_coef',
    'return_data_combined_ic',
    'return_data_combined_cvar_ic',
    'return_data_combined_corr',
    'return_data_combined_cvar_corr',
    'return_data_combined_hs300'
]
risk_free_rate = para.risk_free_rate  # 无风险利率假设为0
results = []
for column_name in data_columns:
    data_to_process = globals()[column_name]
    data_to_process, max_drawdown = calculate_drawdown(data_to_process)
    annualized_return, annualized_volatility = calculate_annualized_metrics(data_to_process)
    sharpe_ratio = calculate_sharpe_ratio(annualized_return,annualized_volatility,risk_free_rate)
    benchmark_data = return_data_combined_hs300  # 假设基准为return_data_combined_hs300
    information_ratio = calculate_information_ratio(data_to_process, benchmark_data, risk_free_rate)
    return_drawdown_ratio = calculate_return_drawdown_ratio(annualized_return, max_drawdown)
    result = {
        'Column': column_name,
        'Annual Return': annualized_return,  # 年化收益率
        'Max Drawdown': max_drawdown,
        'Return Drawdown Ratio': return_drawdown_ratio,  # 收益回撤比
        'Annual Volatility': annualized_volatility,  # 年化波动率
        'Sharpe Ratio': sharpe_ratio,
        'Information Ratio': information_ratio,
    }
    results.append(result)
# 将结果存储在 DataFrame 中
results_df = pd.DataFrame(results)
# 将DataFrame输出到Excel文件
results_df.to_excel(para.path_results + 'evaluation results.xlsx', index=False)
print(results_df)


# Rows to subtract: 2nd from 1st, 4th from 3rd, and 6th from 5th
rows_to_subtract = [1, 3, 5]
# Columns to exclude from subtraction
columns_to_exclude = ['Column']  # Replace 'Column' with the actual name of the first column
# Get columns except the ones to exclude
columns_for_subtraction = [col for col in results_df.columns if col not in columns_to_exclude]
# Perform row-wise subtraction excluding the specified columns
results_evaluation  = results_df[columns_for_subtraction].diff().iloc[rows_to_subtract]


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# 绘制回撤曲线并标注最大回撤
def plot_drawdown_with_max(data, column_name):
    data, max_drawdown = calculate_drawdown(data)
    plt.plot(data['Month'], data['drawdown'], label=column_name)
    max_drawdown_index = data['drawdown'].idxmax()
    plt.annotate(f'Max DD: {max_drawdown:.2f}',
                 xy=(data['Month'][max_drawdown_index], data['drawdown'][max_drawdown_index]),
                 xytext=(data['Month'][max_drawdown_index], data['drawdown'][max_drawdown_index] - 0.1),  # 调整y轴位置
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

# 设置风格和配色方案
sns.set_palette("pastel")
plt.rcParams['axes.facecolor'] = 'whitesmoke'  # 设置背景色
plt.figure(dpi=300)

# 自定义颜色列表
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 第一张图
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(return_data_combined_coef['Month'], return_data_combined_coef['return'], label='return_coef', color=colors[0])
plt.plot(return_data_combined_cvar_coef['Month'], return_data_combined_cvar_coef['return'], label='return_var_coef', color=colors[1])
plt.plot(return_data_combined_hs300['Month'], return_data_combined_hs300['return'], label='return_hs300', color=colors[2])
# 设置日期显示的间隔和格式
date_interval = 6  # 每隔10个月显示一个月份，至少显示一个月份
plt.xticks(ticks=plt.xticks()[0][::date_interval])
plt.gcf().autofmt_xdate()  # 自动格式化日期显示
plt.legend()
plt.xlabel('Month')
plt.ylabel('Return')
plt.title('Return Data Combined Coef')

plt.subplot(2, 1, 2)
plt.plot(return_data_combined_coef['Month'], return_data_combined_coef['compound_value'], label='compound_value_coef', color=colors[3])
plt.plot(return_data_combined_cvar_coef['Month'], return_data_combined_cvar_coef['compound_value'], label='compound_value_var_coef', color=colors[4])
plt.plot(return_data_combined_hs300['Month'], return_data_combined_hs300['compound_value'], label='compound_value_hs300', color=colors[5])
# 设置日期显示的间隔和格式
# 设置日期显示的间隔和格式
date_interval = 6 # 每隔10个月显示一个月份，至少显示一个月份
plt.xticks(ticks=plt.xticks()[0][::date_interval])
plt.gcf().autofmt_xdate()  # 自动格式化日期显示
plt.legend()
plt.xlabel('Month')
plt.ylabel('Compound Value')
filename = 'Coef.png'
filepath = os.path.join(para.path_results, filename)
plt.savefig(filepath, dpi=300)
plt.tight_layout()
plt.show()

# 绘制三条回撤曲线在同一张图上
plt.figure(figsize=(10, 4))
# 绘制回撤曲线
plot_drawdown_with_max(return_data_combined_coef, 'return_data_combined_coef')
plot_drawdown_with_max(return_data_combined_cvar_coef, 'return_data_combined_cvar_coef')
plot_drawdown_with_max(return_data_combined_hs300, 'return_data_combined_hs300')
# 限制y轴范围
plt.ylim(-0.01, 0.7)  # 调整y轴范围
# 设置日期显示的间隔和格式
date_interval = 6  # 每隔10个月显示一个月份，至少显示一个月份
plt.xticks(ticks=plt.xticks()[0][::date_interval])
plt.gcf().autofmt_xdate()  # 自动格式化日期显示
# 配置图例、标签和标题
plt.legend()
plt.xlabel('Month')
plt.ylabel('Drawdown')
plt.title('Drawdown Comparison')
filename = 'Drawdown Comparison Coef.png'
filepath = os.path.join(para.path_results, filename)
plt.savefig(filepath, dpi=300)
plt.show()

# 第二张图
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(return_data_combined_ic['Month'], return_data_combined_ic['return'], label='return_ic', color=colors[6])
plt.plot(return_data_combined_cvar_ic['Month'], return_data_combined_cvar_ic['return'], label='return_var_ic', color=colors[7])
plt.plot(return_data_combined_hs300['Month'], return_data_combined_hs300['return'], label='return_hs300', color=colors[8])
# 设置日期显示的间隔和格式
date_interval = 6  # 每隔10个月显示一个月份，至少显示一个月份
plt.xticks(ticks=plt.xticks()[0][::date_interval])
plt.gcf().autofmt_xdate()  # 自动格式化日期显示
plt.legend()
plt.xlabel('Month')
plt.ylabel('Return')
plt.title('Return Data Combined Ic')

plt.subplot(2, 1, 2)
plt.plot(return_data_combined_ic['Month'], return_data_combined_ic['compound_value'], label='compound_value_ic', color=colors[0])
plt.plot(return_data_combined_cvar_ic['Month'], return_data_combined_cvar_ic['compound_value'], label='compound_value_var_ic', color=colors[1])
plt.plot(return_data_combined_hs300['Month'], return_data_combined_hs300['compound_value'], label='compound_value_hs300', color=colors[2])
# 设置日期显示的间隔和格式
date_interval = 6  # 每隔10个月显示一个月份，至少显示一个月份
plt.xticks(ticks=plt.xticks()[0][::date_interval])
plt.gcf().autofmt_xdate()  # 自动格式化日期显示
plt.legend()
plt.xlabel('Month')
plt.ylabel('Compound Value')
filename = 'IC.png'
filepath = os.path.join(para.path_results, filename)
plt.savefig(filepath, dpi=300)
plt.tight_layout()
plt.show()

# 绘制三条回撤曲线在同一张图上
plt.figure(figsize=(10, 4))
# 绘制回撤曲线
plot_drawdown_with_max(return_data_combined_ic, 'return_data_combined_ic')
plot_drawdown_with_max(return_data_combined_cvar_ic, 'return_data_combined_cvar_ic')
plot_drawdown_with_max(return_data_combined_hs300, 'return_data_combined_hs300')
# 限制y轴范围
plt.ylim(-0.01, 0.7)  # 调整y轴范围
# 设置日期显示的间隔和格式
date_interval = 6  # 每隔10个月显示一个月份，至少显示一个月份
plt.xticks(ticks=plt.xticks()[0][::date_interval])
plt.gcf().autofmt_xdate()  # 自动格式化日期显示
# 配置图例、标签和标题
plt.legend()
plt.xlabel('Month')
plt.ylabel('Drawdown')
plt.title('Drawdown Comparison')
filename = 'Drawdown Comparison IC.png'
filepath = os.path.join(para.path_results, filename)
plt.savefig(filepath, dpi=300)
plt.show()

# 第三张图
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(return_data_combined_corr['Month'], return_data_combined_corr['return'], label='return_corr', color=colors[3])
plt.plot(return_data_combined_cvar_corr['Month'], return_data_combined_cvar_corr['return'], label='return_var_corr', color=colors[4])
plt.plot(return_data_combined_hs300['Month'], return_data_combined_hs300['return'], label='return_hs300', color=colors[5])
# 设置日期显示的间隔和格式
date_interval = 6 # 每隔10个月显示一个月份，至少显示一个月份
plt.xticks(ticks=plt.xticks()[0][::date_interval])
plt.gcf().autofmt_xdate()  # 自动格式化日期显示
plt.legend()
plt.xlabel('Month')
plt.ylabel('Return')
plt.title('Return Data Combined Corr')

plt.subplot(2, 1, 2)
plt.plot(return_data_combined_corr['Month'], return_data_combined_corr['compound_value'], label='compound_value_corr', color=colors[6])
plt.plot(return_data_combined_cvar_corr['Month'], return_data_combined_cvar_corr['compound_value'], label='compound_value_var_corr', color=colors[7])
plt.plot(return_data_combined_hs300['Month'], return_data_combined_hs300['compound_value'], label='compound_value_hs300', color=colors[8])
# 设置日期显示的间隔和格式
date_interval = 6  # 每隔10个月显示一个月份，至少显示一个月份
plt.xticks(ticks=plt.xticks()[0][::date_interval])
plt.gcf().autofmt_xdate()  # 自动格式化日期显示
plt.legend()
plt.xlabel('Month')
plt.ylabel('Compound Value')
filename = 'Corr.png'
filepath = os.path.join(para.path_results, filename)
plt.savefig(filepath, dpi=300)
plt.tight_layout()
plt.show()

# 绘制三条回撤曲线在同一张图上
plt.figure(figsize=(10, 4))
# 绘制回撤曲线
plot_drawdown_with_max(return_data_combined_corr, 'return_data_combined_corr')
plot_drawdown_with_max(return_data_combined_cvar_corr, 'return_data_combined_cvar_corr')
plot_drawdown_with_max(return_data_combined_hs300, 'return_data_combined_hs300')
# 限制y轴范围
plt.ylim(-0.01, 0.7)  # 调整y轴范围
# 设置日期显示的间隔和格式
date_interval = 6  # 每隔10个月显示一个月份，至少显示一个月份
plt.xticks(ticks=plt.xticks()[0][::date_interval])
plt.gcf().autofmt_xdate()  # 自动格式化日期显示
# 配置图例、标签和标题
plt.legend()
plt.xlabel('Month')
plt.ylabel('Drawdown')
plt.title('Drawdown Comparison')
filename = 'Drawdown Comparison Corr.png'
filepath = os.path.join(para.path_results, filename)
plt.savefig(filepath, dpi=300)
plt.show()

print('回归系数下因子权重收益:', return_data_combined_coef['return'].mean())
print('回归系数下因子权重累计价值:', return_data_combined_coef['compound_value'].mean())
print('CVaR_coef下因子权重收益:', return_data_combined_cvar_coef['return'].mean())
print('CVaR_coef下因子权重累计价值:', return_data_combined_cvar_coef['compound_value'].mean())
print('信息系数下因子权重收益:', return_data_combined_ic['return'].mean())
print('信息系数下因子权重累计价值:', return_data_combined_ic['compound_value'].mean())
print('CVaR_ic下因子权重收益:', return_data_combined_cvar_ic['return'].mean())
print('CVaR_ic下因子权重累计价值:', return_data_combined_cvar_ic['compound_value'].mean())
print('协方差矩阵下因子权重收益:', return_data_combined_corr['return'].mean())
print('协方差矩阵下因子权重累计价值:', return_data_combined_corr['compound_value'].mean())
print('CVaR_corr下因子权重收益:', return_data_combined_cvar_corr['return'].mean())
print('CVaR_corr下因子权重累计价值:', return_data_combined_cvar_corr['compound_value'].mean())
print('hs300收益:', return_data_combined_hs300['return'].mean())
print('hs300累计价值:', return_data_combined_hs300['compound_value'].mean())

# 创建字典保存结果
results_dict = {
    'Factor': ['回归系数下因子权重', '回归系数下因子权重累计价值', 'CVaR_coef下因子权重', 'CVaR_coef下因子权重累计价值',
               '信息系数下因子权重', '信息系数下因子权重累计价值', 'CVaR_ic下因子权重', 'CVaR_ic下因子权重累计价值',
               '协方差矩阵下因子权重', '协方差矩阵下因子权重累计价值', 'CVaR_corr下因子权重', 'CVaR_corr下因子权重累计价值',
               'hs300收益', 'hs300累计价值'],
    'Value': [
        return_data_combined_coef['return'].mean(),
        return_data_combined_coef['compound_value'].mean(),
        return_data_combined_cvar_coef['return'].mean(),
        return_data_combined_cvar_coef['compound_value'].mean(),
        return_data_combined_ic['return'].mean(),
        return_data_combined_ic['compound_value'].mean(),
        return_data_combined_cvar_ic['return'].mean(),
        return_data_combined_cvar_ic['compound_value'].mean(),
        return_data_combined_corr['return'].mean(),
        return_data_combined_corr['compound_value'].mean(),
        return_data_combined_cvar_corr['return'].mean(),
        return_data_combined_cvar_corr['compound_value'].mean(),
        return_data_combined_hs300['return'].mean(),
        return_data_combined_hs300['compound_value'].mean()
    ]
}
# 转换为 DataFrame
results_df = pd.DataFrame(results_dict)
# 将结果保存到 Excel 文件中
results_df.to_excel(para.path_results + 'results summary.xlsx', index=False)

# 导出最优选股及权重
# 创建一个Excel写入对象
filename = 'portfolio data.xlsx'
filepath = os.path.join(para.path_results, filename)
writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
# 将DataFrame写入Excel文件的不同sheet页
return_data_combined_coef.to_excel(writer, sheet_name='return_data_combined_coef', index=False)
weights_data_combined_coef.to_excel(writer, sheet_name='weights_data_combined_coef', index=False)
return_data_combined_ic.to_excel(writer, sheet_name='return_data_combined_ic', index=False)
weights_data_combined_ic.to_excel(writer, sheet_name='weights_data_combined_ic', index=False)
return_data_combined_corr.to_excel(writer, sheet_name='return_data_combined_corr', index=False)
weights_data_combined_corr.to_excel(writer, sheet_name='weights_data_combined_corr', index=False)
return_data_combined_cvar_coef.to_excel(writer, sheet_name='return_data_combined_cvar_coef', index=False)
weights_data_combined_cvar_coef.to_excel(writer, sheet_name='weights_data_combined_cvar_coef', index=False)
return_data_combined_cvar_ic.to_excel(writer, sheet_name='return_data_combined_cvar_ic', index=False)
weights_data_combined_cvar_ic.to_excel(writer, sheet_name='weights_data_combined_cvar_ic', index=False)
return_data_combined_cvar_corr.to_excel(writer, sheet_name='return_data_combined_cvar_corr', index=False)
weights_data_combined_cvar_corr.to_excel(writer, sheet_name='weights_data_combined_cvar_corr', index=False)
return_data_combined_hs300.to_excel(writer, sheet_name='return_data_combined_hs300', index=False)
# 保存Excel文件
writer.close()