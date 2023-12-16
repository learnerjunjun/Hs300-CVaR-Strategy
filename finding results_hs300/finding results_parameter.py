# 导入包
import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize
import cvxpy as cp
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
    train_min_months = 60 #训练最少使用数据量
    train_start_month = 85 #训练开始时间 或者从94开始，避免回撤过大，年均收益率更高
    train_max_months = 48 #训练最多使用数据量
    train_update_months = 3 #训练轮动周期
    max_date = 284
    max_select = 20 #选股最多数量
    roll_back = 24 #决策集回溯地历史数据，从12个月开始调整，间隔周期为6个月，18个月长期收益表现可以
    CVaR_method = "minimize_cvar_scipy"
    # 优化算法
    # "differential_evolution", "genetic_algorithm", "simulated_annealing",
    # "bayesian_optimization", "particle_swarm_optimization"
    # "nlopt_optimization", "minimize_cvar_scipy"
    confidence_level = 0.99 #置信水平
    risk_free_rate = 0  # 无风险利率假设为0
    y_data = 'curr_return'  #'curr_return' ,'excess_return_curr','next_return','excess_return_next'
para = Para()

def rolling_train(parameters):
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
    def objective_function(weights, returns, confidence_level):
        portfolio_returns = calculate_portfolio_returns(returns, weights)
        cvar = calculate_cvar(portfolio_returns, confidence_level)
        return cvar
    def constraint_function(weights):
        return np.sum(weights) - 1
    def minimize_cvar(returns, confidence_level):
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
    def calculate_sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate):
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
    # 训练部分
    train_data_min_months = parameters.train_min_months  # 每次模型训练所用数据最少不低于
    train_data_max_months = parameters.train_max_months  # 每次模型训练所用数据最大不超过
    train_update_months = parameters.train_update_months  # 设置更新周期
    start_date = parameters.train_start_month  # 第一次滚动训练开始日期
    end_date = start_date + train_data_min_months - 1  # 第一次滚动训练结束日期

    # 创建一个空的DataFrame
    return_data_combined_coef = pd.DataFrame()
    return_data_combined_ic = pd.DataFrame()
    return_data_combined_corr = pd.DataFrame()
    return_data_combined_cvar_coef = pd.DataFrame()
    return_data_combined_cvar_ic = pd.DataFrame()
    return_data_combined_cvar_corr = pd.DataFrame()
    return_data_combined_hs300 = pd.DataFrame()
    while end_date <= parameters.max_date:
        period_train = range(start_date, end_date + 1)
        ## 生成样本内数据集
        # -- generate in-sample data
        for i_month in period_train:
            # -- load csv
            file_name_curr = parameters.path_data + str(i_month) + '.csv'
            file_name_next = parameters.path_data + str(i_month+1) + '.csv'
            data_curr_month = pd.read_csv(file_name_curr, header=0)  # 设置表头
            data_next_month = pd.read_csv(file_name_next, header=0)  # 设置表头
            parameters.n_stock = data_curr_month.shape[0]
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
        data_in_sample[parameters.y_data] = data_in_sample[parameters.y_data].astype(float)
        data_in_sample_next[parameters.y_data] = data_in_sample_next[parameters.y_data].astype(float)
        # 样本内数据集
        # -- generate in-sample data
        X_in_sample = data_in_sample.loc[:, 'EP':'bias']  # 提取数据
        # -- regression
        if parameters.method in ['LR']:
            y_in_sample = data_in_sample.loc[:, parameters.y_data]
        from sklearn import linear_model
        from sklearn.decomposition import PCA
        if parameters.method in ['LR']:
            model = linear_model.LinearRegression(fit_intercept=True)  # 计算偏置（截距）
            # -- regression
            model.fit(X_in_sample, y_in_sample)
        # 因子信息系数
        columns = ['code', 'month']
        factors_columns = X_in_sample.columns
        for i in factors_columns:
            columns.append(i)
        # 以 'month' 和 'code' 为基准合并两个 DataFrame，选择合适的 how 参数，根据实际需求选择合适的连接方式
        data_in_sample_next_y = data_in_sample_next[['month', 'code', parameters.y_data]]
        data_in_sample_x = data_in_sample.loc[:, columns]
        merged_data = pd.merge(data_in_sample_next_y, data_in_sample_x, on=['month', 'code'], how='inner')
        # 提取合并后的列
        X_in_sample_ic = merged_data.loc[:, 'EP':'bias']
        y_in_sample_ic = merged_data[parameters.y_data]
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
        test_date_end = end_date + parameters.train_update_months
        period_test = range(test_date_start, test_date_end + 1)
        combined_y_pred_return = pd.DataFrame()
        combined_y_curr_return = pd.DataFrame()
        for i_month in period_test:
            # -- load
            file_name = parameters.path_data + str(i_month) + '.csv'
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
            if parameters.method in ['LR']:
                y_score_curr_month = model_pca.predict(X_curr_month)
            y_pred_return = pd.DataFrame(
                {'month': data_curr_month['month'], 'code': data_curr_month['code'], 'pred_return': y_score_curr_month})
            combined_y_pred_return = pd.concat([combined_y_pred_return, y_pred_return], axis=0)
            # -- curr_return
            y_curr_return = pd.DataFrame({'month': data_curr_month['month'], 'code': data_curr_month['code'],
                                          'curr_return': data_curr_month[parameters.y_data]})
            combined_y_curr_return = pd.concat([combined_y_curr_return, y_curr_return], axis=0)
        combined_y_pred_return = pd.pivot_table(combined_y_pred_return, values='pred_return', index='code',
                                                columns=['month'])
        combined_y_curr_return = pd.pivot_table(combined_y_curr_return, values='curr_return', index='code',
                                                columns=['month'])

        max_select = parameters.max_select  # 最长的数据长度
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
            file_name_1 = parameters.path_data + str(i_month_1 - 1) + '.csv'
            file_name_2 = parameters.path_data + str(i_month_1) + '.csv'
            data_last_month = pd.read_csv(file_name_1, header=0)  # 设置表头
            data_curr_month = pd.read_csv(file_name_2, header=0)  # 设置表头
            data_last_month['code'] = data_last_month['stock'].str.replace('[^\d]', '', regex=True)
            data_curr_month['code'] = data_curr_month['stock'].str.replace('[^\d]', '', regex=True)
            # --hs300
            data_last_month = pd.merge(hs300, data_last_month, on=['code', 'month'], how='inner')
            data_last_month[parameters.y_data] = data_last_month[parameters.y_data].astype(float)
            data_curr_month = pd.merge(hs300, data_curr_month, on=['code', 'month'], how='inner')
            data_curr_month[parameters.y_data] = data_curr_month[parameters.y_data].astype(float)
            # -- remove nan
            data_last_month = data_last_month.dropna(axis=0)
            data_curr_month = data_curr_month.dropna(axis=0)
            # 打分法筛选出股票
            data_for_score_x = data_last_month.copy()
            data_for_score_x = data_for_score_x.drop(labels=['month',parameters.y_data], axis=1)
            data_for_score_y = data_curr_month[['month', 'code', parameters.y_data]]
            data_for_score = pd.merge(data_for_score_y, data_for_score_x, on=['code'], how='inner')
            X_for_score = data_for_score.loc[:, 'EP':'bias']  # 提取数据
            y_curr_month = pd.DataFrame(
                {'month': data_for_score['month'], 'code': data_for_score['code'], 'curr_return': data_for_score[parameters.y_data]})
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
            period_select = range(test_date_start - parameters.roll_back, i_month_1 + 1)
            combined_y_curr_return_past = pd.DataFrame()
            for i_month_2 in period_select:
                # -- load
                file_name = parameters.path_data + str(i_month_2) + '.csv'
                data_curr_month = pd.read_csv(file_name, header=0)
                # --hs300
                data_curr_month['code'] = data_curr_month['stock'].str.replace('[^\d]', '', regex=True)
                data_curr_month = pd.merge(hs300, data_curr_month, on=['code', 'month'], how='inner')
                data_curr_month[parameters.y_data] = data_curr_month[parameters.y_data].astype(float)
                # -- remove nan
                data_curr_month = data_curr_month.dropna(axis=0)
                # -- curr_return
                y_curr_return_past = pd.DataFrame({'month': data_curr_month['month'], 'code': data_curr_month['code'],
                                                   'curr_return': data_curr_month[parameters.y_data]})
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
            confidence_level = parameters.confidence_level
            returns_cvar_coef = top_select_stocks_return_coef
            returns_cvar_ic = top_select_stocks_return_ic
            returns_cvar_corr = top_select_stocks_return_corr
            cvar_coef, portfolio_weights_cvar_coef = minimize_cvar(returns_cvar_coef, confidence_level)
            cvar_ic, portfolio_weights_cvar_ic = minimize_cvar(returns_cvar_ic, confidence_level)
            cvar_corr, portfolio_weights_cvar_corr = minimize_cvar(returns_cvar_corr, confidence_level)

            # 计算组合收益
            y_curr_month_return_coef = y_curr_month.loc[y_curr_month.index.intersection(stock_codes_coef)][
                'curr_return']
            y_curr_month_return_ic = y_curr_month.loc[y_curr_month.index.intersection(stock_codes_ic)]['curr_return']
            y_curr_month_return_corr = y_curr_month.loc[y_curr_month.index.intersection(stock_codes_corr)][
                'curr_return']

            portfolio_return_coef = np.dot(portfolio_weights_coef.T, y_curr_month_return_coef)
            portfolio_return_ic = np.dot(portfolio_weights_ic.T, y_curr_month_return_ic)
            portfolio_return_corr = np.dot(portfolio_weights_corr.T, y_curr_month_return_corr)
            portfolio_return_cvar_coef = np.dot(portfolio_weights_cvar_coef.T, y_curr_month_return_coef)
            portfolio_return_cvar_ic = np.dot(portfolio_weights_cvar_ic.T, y_curr_month_return_ic)
            portfolio_return_cvar_corr = np.dot(portfolio_weights_cvar_corr.T, y_curr_month_return_corr)
            hs300_idxrtn = hs300_return.loc[hs300_return['month'] == i_month_1, 'Idxrtn'].item()

            # 将收益数据添加到DataFrame中
            row_coef = {'month': i_month_1, 'return': portfolio_return_coef,
                        'Month': data_for_score['month_details'].unique()}
            portfolio_return_data_coef = pd.concat([portfolio_return_data_coef, pd.DataFrame(row_coef, index=[0])],
                                                   ignore_index=True)
            row_ic = {'month': i_month_1, 'return': portfolio_return_ic,
                      'Month': data_for_score['month_details'].unique()}
            portfolio_return_data_ic = pd.concat([portfolio_return_data_ic, pd.DataFrame(row_ic, index=[0])],
                                                 ignore_index=True)
            row_corr = {'month': i_month_1, 'return': portfolio_return_corr,
                        'Month': data_for_score['month_details'].unique()}
            portfolio_return_data_corr = pd.concat([portfolio_return_data_corr, pd.DataFrame(row_corr, index=[0])],
                                                   ignore_index=True)
            row_cvar_coef = {'month': i_month_1, 'return': portfolio_return_cvar_coef,
                             'Month': data_for_score['month_details'].unique()}
            portfolio_return_data_cvar_coef = pd.concat(
                [portfolio_return_data_cvar_coef, pd.DataFrame(row_cvar_coef, index=[0])],
                ignore_index=True)
            row_cvar_ic = {'month': i_month_1, 'return': portfolio_return_cvar_ic,
                           'Month': data_for_score['month_details'].unique()}
            portfolio_return_data_cvar_ic = pd.concat(
                [portfolio_return_data_cvar_ic, pd.DataFrame(row_cvar_ic, index=[0])],
                ignore_index=True)
            row_cvar_corr = {'month': i_month_1, 'return': portfolio_return_cvar_corr,
                             'Month': data_for_score['month_details'].unique()}
            portfolio_return_data_cvar_corr = pd.concat(
                [portfolio_return_data_cvar_corr, pd.DataFrame(row_cvar_corr, index=[0])],
                ignore_index=True)
            row_hs300 = {'month': i_month_1, 'return': hs300_idxrtn, 'Month': data_for_score['month_details'].unique()}
            return_data_hs300 = pd.concat([return_data_hs300, pd.DataFrame(row_hs300, index=[0])],
                                          ignore_index=True)

        return_data_combined_coef = pd.concat([return_data_combined_coef, portfolio_return_data_coef],
                                              ignore_index=True)
        return_data_combined_ic = pd.concat([return_data_combined_ic, portfolio_return_data_ic], ignore_index=True)
        return_data_combined_corr = pd.concat([return_data_combined_corr, portfolio_return_data_corr],
                                              ignore_index=True)
        return_data_combined_cvar_coef = pd.concat([return_data_combined_cvar_coef, portfolio_return_data_cvar_coef],
                                                   ignore_index=True)
        return_data_combined_cvar_ic = pd.concat([return_data_combined_cvar_ic, portfolio_return_data_cvar_ic],
                                                 ignore_index=True)
        return_data_combined_cvar_corr = pd.concat([return_data_combined_cvar_corr, portfolio_return_data_cvar_corr],
                                                   ignore_index=True)
        return_data_combined_hs300 = pd.concat([return_data_combined_hs300, return_data_hs300], ignore_index=True)

        ##数据集滚动
        # 更新日期
        start_date += train_update_months
        # 限制训练数据的时间范围
        end_date = min(start_date + train_data_min_months - 1, parameters.max_date)
        # 若接近最大日期，则终止训练
        if end_date + train_update_months >= parameters.max_date:
            break

    #计算累计收益
    return_data_combined_coef['compound_value'] = (return_data_combined_coef['return']+1).cumprod()
    return_data_combined_cvar_coef['compound_value'] = (return_data_combined_cvar_coef['return']+1).cumprod()
    return_data_combined_ic['compound_value'] = (return_data_combined_ic['return']+1).cumprod()
    return_data_combined_cvar_ic['compound_value'] = (return_data_combined_cvar_ic['return']+1).cumprod()
    return_data_combined_corr['compound_value'] = (return_data_combined_corr['return']+1).cumprod()
    return_data_combined_cvar_corr['compound_value'] = (return_data_combined_cvar_corr['return']+1).cumprod()
    return_data_combined_hs300['compound_value'] = (return_data_combined_hs300['return']+1).cumprod()

    # 假设数据以字典形式存储
    data = {
        'return_data_combined_coef': return_data_combined_coef,
        'return_data_combined_cvar_coef': return_data_combined_cvar_coef,
        'return_data_combined_ic': return_data_combined_ic,
        'return_data_combined_cvar_ic': return_data_combined_cvar_ic,
        'return_data_combined_corr': return_data_combined_corr,
        'return_data_combined_cvar_corr': return_data_combined_cvar_corr,
        'return_data_combined_hs300': return_data_combined_hs300
    }
    # 储存计算结果
    results = []
    risk_free_rate = parameters.risk_free_rate  # 假设无风险利率为0
    for column_name, data_to_process in data.items():
        # 假设这些函数是你已经定义的
        data_to_process, max_drawdown = calculate_drawdown(data_to_process)
        annualized_return, annualized_volatility = calculate_annualized_metrics(data_to_process)
        sharpe_ratio = calculate_sharpe_ratio(annualized_return, annualized_volatility, risk_free_rate)
        benchmark_data = data['return_data_combined_hs300']  # 假设基准数据存在于字典中
        information_ratio = calculate_information_ratio(data_to_process, benchmark_data, risk_free_rate)
        return_drawdown_ratio = calculate_return_drawdown_ratio(annualized_return, max_drawdown)
        result = {
            'Column': column_name,
            'Annual Return': annualized_return,
            'Max Drawdown': max_drawdown,
            'Return Drawdown Ratio': return_drawdown_ratio,
            'Annual Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Information Ratio': information_ratio,
        }
        results.append(result)
    results_df = pd.DataFrame(results)

    # Rows to subtract: 2nd from 1st, 4th from 3rd, and 6th from 5th
    rows_to_subtract = [1, 3, 5]
    # Columns to exclude from subtraction
    columns_to_exclude = ['Column']  # Replace 'Column' with the actual name of the first column
    # Get columns except the ones to exclude
    columns_for_subtraction = [col for col in results_df.columns if col not in columns_to_exclude]
    # Perform row-wise subtraction excluding the specified columns
    results_evaluation  = results_df[columns_for_subtraction].diff().iloc[rows_to_subtract]

    return results_evaluation


from bayes_opt import BayesianOptimization
# 定义目标函数
def objective_function(train_min_months, train_update_months, roll_back, max_select): #, confidence_level):
    try:
        # 创建 Para 类的实例并设置参数值
        para = Para()
        para.train_min_months = int(round(train_min_months))
        para.train_update_months = int(round(train_update_months))
        para.roll_back = int(round(roll_back))
        para.max_select = int(round(max_select))
        # para.confidence_level = round(confidence_level, 2)  # 保留两位小数
        # 执行滚动训练
        results_evaluation = rolling_train(para)
        # 计算第一列值的总和
        sum_of_first_column = results_evaluation.iloc[:, 0].sum()
        # 总和最大化
        target_value = sum_of_first_column
        return target_value
    except Exception as e:
        print(f"Exception occurred: {e}")
        return -1e6  # 返回一个极端小的目标函数值，让优化算法跳过这组参数
# 定义参数范围
pbounds = {
    'train_min_months': (24, 84),
    'train_update_months': (3, 12),
    'roll_back': (12, 72),
    'max_select': (10, 50),
    #'confidence_level': (0.95, 0.99)
}
# 初始化贝叶斯优化器
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=42
)
# 开始优化
optimizer.maximize(init_points=5, n_iter=20)
# 输出最佳的参数组合
best_params = optimizer.max['params']
best_params['train_min_months'] = int(round(best_params['train_min_months']))
best_params['train_update_months'] = int(round(best_params['train_update_months']))
best_params['roll_back'] = int(round(best_params['roll_back']))
best_params['max_select'] = int(round(best_params['max_select']))
best_params['confidence_level'] = round(best_params['confidence_level'], 2)
print(f"最佳参数: {best_params}")
print(f"最佳结果: {optimizer.max['target']}")