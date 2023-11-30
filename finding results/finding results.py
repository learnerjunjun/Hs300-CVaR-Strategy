# 导入包
import numpy as np
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')

# hs300名单 2022.1.1之前
hs300 = pd.read_csv(r'../hs300_2005-2022/hs300_monthly.csv', dtype=object)
hs300 = hs300.drop('Unnamed: 0', axis=1)
hs300 = hs300.rename(columns={'stock': 'code'})
hs300['month'] = hs300['month'].astype(int)


## 定义参数类
# -- define a class including all parameters
class Para():
    method = 'LR'
    month_in_sample = range(82, 153 + 1)  # -- return 82~153 72 months
    month_test = range(154, 293 + 1)  # -- return 154~293 140 months

    percent_select = [0.3, 0.3]  # -- 30% positive samples, 30% negative samples
    percent_cv = 0.1  # -- percentage of cross validation samples 交叉验证的样本比例
    path_data = '../data/csv_01/'
    path_results = './results/'
    seed = 42  # -- random seed
    n_stock = 5166


para = Para()

train_data_min_months = 72  # 每次模型训练所用数据最少不低于
train_data_max_months = 108  # 每次模型训练所用数据最大不超过
train_update_months = 6  # 设置更新周期
start_date = 82  # 第一次滚动训练开始日期
end_date = start_date + train_data_min_months  # 第一次滚动训练结束日期

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

while end_date <= 284:
    period_train = range(start_date, end_date + 1)
    ## 生成样本内数据集
    # -- generate in-sample data
    a1 = pd.DataFrame([np.nan] * np.zeros((para.n_stock, period_train[-1])))
    for i_month in period_train:
        # -- load csv
        file_name = para.path_data + str(i_month) + '.csv'
        data_curr_month = pd.read_csv(file_name, header=0)  # 设置表头
        para.n_stock = data_curr_month.shape[0]
        # -- remove nan
        data_curr_month = data_curr_month.dropna(axis=0)
        # return data merge
        a1.iloc[data_curr_month.index, i_month - 1] = data_curr_month['return'][data_curr_month.index]
        # -- merge
        if i_month == period_train[0]:  # -- first month
            data_in_sample_all = data_curr_month
        else:
            data_in_sample_all = pd.concat((data_in_sample_all, data_curr_month), axis=0)
        data_in_sample_all['code'] = data_in_sample_all['stock'].str.replace('[^\d]', '', regex=True)
    # 筛选出名单内的hs300
    data_in_sample = pd.merge(hs300, data_in_sample_all, on=['code', 'month'], how='inner')
    # 样本内数据集
    # -- generate in-sample data
    X_in_sample = data_in_sample.loc[:, 'EP':'bias']  # 提取数据
    # -- regression
    if para.method in ['LR']:
        y_in_sample = data_in_sample.loc[:, 'return']
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
    factors_columns = data_in_sample.columns[7:-12]
    for i in factors_columns:
        columns.append(i)
    X_in_sample_ic = data_in_sample.loc[:, columns]  # 提取数据
    y_in_sample_ic = data_in_sample[['code', 'month', 'return']]
    # 提取code和month列
    code_col = X_in_sample_ic['code']
    month_col = X_in_sample_ic['month']
    # 获取下一期的收益率数据
    next_month = y_in_sample_ic['month'] + 1
    y_in_sample_next = y_in_sample_ic['code']
    y_in_sample_next = pd.DataFrame(y_in_sample_next)
    y_in_sample_next['month'] = next_month
    y_in_sample_next = y_in_sample_ic.merge(y_in_sample_next, on=['code', 'month'])
    y_in_sample_next['month'] = y_in_sample_next['month'] - 1
    # 合并X和y数据
    data_merged = y_in_sample_next.merge(X_in_sample_ic, on=['code', 'month'])
    ic_values = {}  # 存储每个因子的IC值
    for column in data_merged.columns[3:]:  # 从第四列开始计算IC，假设第一列是code，第二列是month，第三列是下一期的收益率
        factor_values = data_merged[column]
        y_values = data_merged.iloc[:, 2]  # 下一期的收益率值
        correlation = np.corrcoef(factor_values, y_values)[0, 1]
        ic_values[column] = correlation
    # 创建包含因子和IC值的DataFrame
    information_coefficient = pd.DataFrame(list(ic_values.items()), columns=['Factor', 'IC'])

    # 基于回归系数
    # 获取回归系数
    coefficients = model.coef_
    # 计算因子权重
    factor_weights_coef = abs(coefficients) / sum(abs(coefficients))


    # 基于信息系数大小的权重分配
    def assign_weight_by_coefficient(information_coefficient):
        # 计算绝对值大于0.02的IC值的总和
        ic_sum = abs(information_coefficient.loc[information_coefficient['IC'] > 0.02, 'IC']).sum()
        # 分配权重
        information_coefficient.loc[information_coefficient['IC'] > 0.02, 'Weight'] = 0.5 * (
                abs(information_coefficient['IC']) / ic_sum)
        information_coefficient.loc[information_coefficient['IC'] <= 0.02, 'Weight'] = 0.5 * (
                (0.02 - abs(information_coefficient['IC'])) / (1 - ic_sum))
        return information_coefficient


    weighted_factors_coefficient = assign_weight_by_coefficient(information_coefficient)
    factor_weights_ic = np.array(weighted_factors_coefficient['Weight'])

    # 基于协方差矩阵
    covariance_matrix = X_in_sample.cov()
    # 计算协方差矩阵的逆
    inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
    # 计算每个因子的相关性得分
    correlation_scores = np.sum(inverse_covariance_matrix, axis=0)
    # 将相关性得分归一化为权重
    weights = correlation_scores / np.sum(correlation_scores)
    # 创建权重分配的DataFrame，并按权重降序排序
    factor_weights = pd.DataFrame({'Factor': X_in_sample.columns, 'Weight': weights})
    factor_weights_sorted = factor_weights.sort_values(by='Weight', ascending=False)
    factor_weights_corr = np.array(factor_weights_sorted['Weight'])

    # 样本外预测
    test_date_start = end_date + 1
    test_date_end = end_date + 6
    period_test = range(test_date_start, test_date_end + 1)
    combined_y_pred_return = pd.DataFrame()
    combined_y_curr_return = pd.DataFrame()
    for i_month in period_test:
        # -- load
        file_name = para.path_data + str(i_month) + '.csv'
        data_curr_month = pd.read_csv(file_name, header=0)
        # -- remove nan
        data_curr_month = data_curr_month.dropna(axis=0)
        # --hs300
        data_curr_month['code'] = data_curr_month['stock'].str.replace('[^\d]', '', regex=True)
        data_curr_month = pd.merge(hs300, data_curr_month, on=['code', 'month'], how='inner')
        # -- generate X
        X_curr_month = data_curr_month.loc[:, 'EP':'bias']
        # -- pca
        pca = PCA(n_components=16)
        X_curr_month = pca.fit_transform(X_curr_month)
        # -- pca_train
        X_train_pca = pca.fit_transform(X_train)
        model_pca = linear_model.LinearRegression(fit_intercept=True)
        model_pca.fit(X_train_pca, y_train)
        # -- linear regression
        if para.method in ['LR']:
            y_score_curr_month = model_pca.predict(X_curr_month)
        y_pred_return = pd.DataFrame(
            {'month': data_curr_month['month'], 'code': data_curr_month['code'], 'pred_return': y_score_curr_month})
        combined_y_pred_return = pd.concat([combined_y_pred_return, y_pred_return], axis=0)
        # -- curr_return
        y_curr_return = pd.DataFrame({'month': data_curr_month['month'], 'code': data_curr_month['code'],
                                      'curr_return': data_curr_month['return']})
        combined_y_curr_return = pd.concat([combined_y_curr_return, y_curr_return], axis=0)
    combined_y_pred_return = pd.pivot_table(combined_y_pred_return, values='pred_return', index='code',
                                            columns=['month'])
    combined_y_curr_return = pd.pivot_table(combined_y_curr_return, values='curr_return', index='code',
                                            columns=['month'])
    if para.method in ['LR']:
        y_train.index = range(len(y_train))
        y_score_train = pd.Series(y_score_train)
        print('training set, ic = %.2f' % y_train.corr(y_score_train))
        if para.percent_cv > 0:
            y_cv.index = range(len(y_cv))
            y_score_cv = pd.Series(y_score_cv)
            print('cv set, ic = %.2f' % y_cv.corr(y_score_cv))
        for i_month in period_test:
            y_true_curr_month = pd.Series(combined_y_curr_return[i_month])
            y_score_curr_month = pd.Series(combined_y_pred_return[i_month])
            print('testing set, month %d, ic = %.2f' % (i_month, y_true_curr_month.corr(y_score_curr_month)))
    # 优化器-1
    from scipy.optimize import minimize
    import cvxpy as cp

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


    # 优化器-2
    def calculate_portfolio_returns(returns, weights):
        portfolio_returns = np.dot(weights.T, returns)
        return portfolio_returns


    def calculate_var(portfolio_returns, confidence_level=0.99):
        var = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        return var


    def calculate_cvar(portfolio_returns, confidence_level=0.99):
        var = calculate_var(portfolio_returns, confidence_level)
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


    def minimize_cvar(returns, confidence_level=0.99):
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


    max_select = 20  # 最长的数据长度
    # 创建一个空的DataFrame来存储最优投资组合权重
    portfolio_weights_df_coef = pd.DataFrame()
    portfolio_weights_df_ic = pd.DataFrame()
    portfolio_weights_df_corr = pd.DataFrame()
    portfolio_weights_df_cvar_coef = pd.DataFrame()
    portfolio_weights_df_cvar_ic = pd.DataFrame()
    portfolio_weights_df_cvar_corr = pd.DataFrame()
    # 创建一个空的DataFrame
    portfolio_return_data_coef = pd.DataFrame(columns=['month', 'return', 'compound_value'])
    portfolio_return_data_ic = pd.DataFrame(columns=['month', 'return', 'compound_value'])
    portfolio_return_data_corr = pd.DataFrame(columns=['month', 'return', 'compound_value'])
    portfolio_return_data_cvar_coef = pd.DataFrame(columns=['month', 'return', 'compound_value'])
    portfolio_return_data_cvar_ic = pd.DataFrame(columns=['month', 'return', 'compound_value'])
    portfolio_return_data_cvar_corr = pd.DataFrame(columns=['month', 'return', 'compound_value'])
    for i_month_1 in period_test:
        # -- load
        file_name = para.path_data + str(i_month_1) + '.csv'
        data_curr_month = pd.read_csv(file_name, header=0)
        # -- remove nan
        data_curr_month = data_curr_month.dropna(axis=0)
        # --hs300
        data_curr_month['code'] = data_curr_month['stock'].str.replace('[^\d]', '', regex=True)
        data_curr_month = pd.merge(hs300, data_curr_month, on=['code', 'month'], how='inner')
        # 打分法筛选出股票
        data_for_score = data_curr_month[data_curr_month['month'] == i_month_1]
        X_for_score = data_for_score.loc[:, 'EP':'bias']  # 提取数据
        y_curr_month = pd.DataFrame(
            {'month': data_for_score['month'], 'code': data_for_score['code'], 'curr_return': data_for_score['return']})
        y_curr_month.set_index('code', inplace=True)
        # 计算每只股票的打分
        scores_coef = X_for_score.dot(factor_weights_coef)
        scores_ic = X_for_score.dot(factor_weights_ic)
        scores_corr = X_for_score.dot(factor_weights_corr)
        n = 30
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
        period_select = range(test_date_start - 6, i_month_1)
        combined_y_curr_return_past = pd.DataFrame()
        for i_month_2 in period_select:
            # -- load
            file_name = para.path_data + str(i_month_2) + '.csv'
            data_curr_month = pd.read_csv(file_name, header=0)
            # -- remove nan
            data_curr_month = data_curr_month.dropna(axis=0)
            # --hs300
            data_curr_month['code'] = data_curr_month['stock'].str.replace('[^\d]', '', regex=True)
            data_curr_month = pd.merge(hs300, data_curr_month, on=['code', 'month'], how='inner')
            # -- curr_return
            y_curr_return_past = pd.DataFrame({'month': data_curr_month['month'], 'code': data_curr_month['code'],
                                               'curr_return': data_curr_month['return']})
            combined_y_curr_return_past = pd.concat([combined_y_curr_return_past, y_curr_return_past], axis=0)

        combined_y_curr_return_past = pd.pivot_table(combined_y_curr_return_past, values='curr_return', index='code',
                                                     columns=['month'])
        combined_return_data = combined_y_curr_return_past.copy()
        combined_return_data[i_month_1] = combined_y_pred_return[i_month_1]

        # 筛选出打分法得到的股票
        top_20_stocks_idx_coef = selected_stocks_coef.index[:20]
        top_20_stocks_return_coef = combined_return_data.loc[
            combined_return_data.index.intersection(top_20_stocks_idx_coef)]
        top_20_stocks_return_coef = top_20_stocks_return_coef.dropna()
        top_20_stocks_idx_ic = selected_stocks_ic.index[:20]
        top_20_stocks_return_ic = combined_return_data.loc[
            combined_return_data.index.intersection(top_20_stocks_idx_ic)]
        top_20_stocks_return_ic = top_20_stocks_return_ic.dropna()
        top_20_stocks_idx_corr = selected_stocks_corr.index[:20]
        top_20_stocks_return_corr = combined_return_data.loc[
            combined_return_data.index.intersection(top_20_stocks_idx_corr)]
        top_20_stocks_return_corr = top_20_stocks_return_corr.dropna()

        # 计算收益率的协方差矩阵
        cov_matrix_coef = top_20_stocks_return_coef.T.cov()
        cov_matrix_ic = top_20_stocks_return_ic.T.cov()
        cov_matrix_corr = top_20_stocks_return_corr.T.cov()
        # 获取 i_month_1 对应的收益均值和股票代码
        mean_returns_coef = top_20_stocks_return_coef.loc[:, i_month_1]
        stock_codes_coef = top_20_stocks_return_coef.index.tolist()
        mean_returns_ic = top_20_stocks_return_ic.loc[:, i_month_1]
        stock_codes_ic = top_20_stocks_return_ic.index.tolist()
        mean_returns_corr = top_20_stocks_return_corr.loc[:, i_month_1]
        stock_codes_corr = top_20_stocks_return_corr.index.tolist()

        # 使用均值-方差模型计算最优投资组合权重
        portfolio_weights_coef = calculate_portfolio_weights(mean_returns_coef, cov_matrix_coef)
        portfolio_weights_ic = calculate_portfolio_weights(mean_returns_ic, cov_matrix_ic)
        portfolio_weights_corr = calculate_portfolio_weights(mean_returns_corr, cov_matrix_corr)
        
        # 使用CVaR计算最优方差
        confidence_level = 0.99
        returns_cvar_coef = top_20_stocks_return_coef
        returns_cvar_ic = top_20_stocks_return_ic
        returns_cvar_corr = top_20_stocks_return_corr
        cvar_coef, portfolio_weights_cvar_coef = minimize_cvar(returns_cvar_coef, confidence_level)
        cvar_ic, portfolio_weights_cvar_ic = minimize_cvar(returns_cvar_ic, confidence_level)
        cvar_corr, portfolio_weights_cvar_corr = minimize_cvar(returns_cvar_corr, confidence_level)

        # 计算组合收益
        y_curr_month_return_coef = y_curr_month.loc[y_curr_month.index.intersection(stock_codes_coef)]['curr_return']
        portfolio_return_coef = np.dot(portfolio_weights_coef.T, y_curr_month_return_coef)
        y_curr_month_return_ic = y_curr_month.loc[y_curr_month.index.intersection(stock_codes_ic)]['curr_return']
        portfolio_return_ic = np.dot(portfolio_weights_ic.T, y_curr_month_return_ic)
        y_curr_month_return_corr = y_curr_month.loc[y_curr_month.index.intersection(stock_codes_corr)]['curr_return']
        portfolio_return_corr = np.dot(portfolio_weights_corr.T, y_curr_month_return_corr)
        portfolio_return_cvar_coef = np.dot(portfolio_weights_cvar_coef.T, y_curr_month_return_coef)
        portfolio_return_cvar_ic = np.dot(portfolio_weights_cvar_ic.T, y_curr_month_return_ic)
        portfolio_return_cvar_corr = np.dot(portfolio_weights_cvar_corr.T, y_curr_month_return_corr)

        # 计算累计值
        if portfolio_return_data_coef.empty:
            cumulative_value_coef = 1 + portfolio_return_coef
        else:
            previous_cumulative_value = portfolio_return_data_coef['compound_value'].iloc[-1]
            cumulative_value_coef = previous_cumulative_value * (1 + portfolio_return_coef)
        if portfolio_return_data_ic.empty:
            cumulative_value_ic = 1 + portfolio_return_ic
        else:
            previous_cumulative_value = portfolio_return_data_ic['compound_value'].iloc[-1]
            cumulative_value_ic = previous_cumulative_value * (1 + portfolio_return_ic)
        if portfolio_return_data_corr.empty:
            cumulative_value_corr = 1 + portfolio_return_corr
        else:
            previous_cumulative_value = portfolio_return_data_corr['compound_value'].iloc[-1]
            cumulative_value_corr = previous_cumulative_value * (1 + portfolio_return_corr)
        if portfolio_return_data_cvar_coef.empty:
            cumulative_value_cvar_coef = 1 + portfolio_return_cvar_coef
        else:
            previous_cumulative_value = portfolio_return_data_cvar_coef['compound_value'].iloc[-1]
            cumulative_value_cvar_coef = previous_cumulative_value * (1 + portfolio_return_cvar_coef)
        if portfolio_return_data_cvar_ic.empty:
            cumulative_value_cvar_ic = 1 + portfolio_return_cvar_ic
        else:
            previous_cumulative_value = portfolio_return_data_cvar_ic['compound_value'].iloc[-1]
            cumulative_value_cvar_ic = previous_cumulative_value * (1 + portfolio_return_cvar_ic)
        if portfolio_return_data_cvar_corr.empty:
            cumulative_value_cvar_corr = 1 + portfolio_return_cvar_corr
        else:
            previous_cumulative_value = portfolio_return_data_cvar_corr['compound_value'].iloc[-1]
            cumulative_value_cvar_corr = previous_cumulative_value * (1 + portfolio_return_cvar_corr)

        # 将收益数据添加到DataFrame中
        row_coef = {'month': i_month_1, 'return': portfolio_return_coef, 'compound_value': cumulative_value_coef}
        portfolio_return_data_coef = pd.concat([portfolio_return_data_coef, pd.DataFrame(row_coef, index=[0])],
                                               ignore_index=True)
        row_ic = {'month': i_month_1, 'return': portfolio_return_ic, 'compound_value': cumulative_value_ic}
        portfolio_return_data_ic = pd.concat([portfolio_return_data_ic, pd.DataFrame(row_ic, index=[0])],
                                             ignore_index=True)
        row_corr = {'month': i_month_1, 'return': portfolio_return_corr, 'compound_value': cumulative_value_corr}
        portfolio_return_data_corr = pd.concat([portfolio_return_data_corr, pd.DataFrame(row_corr, index=[0])],
                                               ignore_index=True)
        row_cvar_coef = {'month': i_month_1, 'return': portfolio_return_cvar_coef, 'compound_value': cumulative_value_cvar_coef}
        portfolio_return_data_cvar_coef = pd.concat([portfolio_return_data_cvar_coef, pd.DataFrame(row_cvar_coef, index=[0])],
                                               ignore_index=True)
        row_cvar_ic = {'month': i_month_1, 'return': portfolio_return_cvar_ic, 'compound_value': cumulative_value_cvar_ic}
        portfolio_return_data_cvar_ic = pd.concat([portfolio_return_data_cvar_ic, pd.DataFrame(row_cvar_ic, index=[0])],
                                               ignore_index=True)
        row_cvar_corr = {'month': i_month_1, 'return': portfolio_return_cvar_corr, 'compound_value': cumulative_value_cvar_corr}
        portfolio_return_data_cvar_corr = pd.concat([portfolio_return_data_cvar_corr, pd.DataFrame(row_cvar_corr, index=[0])],
                                               ignore_index=True)

        # 使用NaN将数据补齐至长度为20
        portfolio_weights_coef = np.concatenate(
            (portfolio_weights_coef, np.full(max_select - len(portfolio_weights_coef), np.nan)))
        stock_codes_coef = np.concatenate((stock_codes_coef, np.full(max_select - len(stock_codes_coef), np.nan)))
        portfolio_weights_ic = np.concatenate(
            (portfolio_weights_ic, np.full(max_select - len(portfolio_weights_ic), np.nan)))
        stock_codes_ic = np.concatenate((stock_codes_ic, np.full(max_select - len(stock_codes_ic), np.nan)))
        portfolio_weights_corr = np.concatenate(
            (portfolio_weights_corr, np.full(max_select - len(portfolio_weights_corr), np.nan)))
        stock_codes_corr = np.concatenate((stock_codes_corr, np.full(max_select - len(stock_codes_corr), np.nan)))
        portfolio_weights_cvar_coef = np.concatenate(
            (portfolio_weights_cvar_coef, np.full(max_select - len(portfolio_weights_cvar_coef), np.nan)))
        stock_codes_corr = np.concatenate((stock_codes_coef, np.full(max_select - len(stock_codes_coef), np.nan)))
        portfolio_weights_cvar_ic = np.concatenate(
            (portfolio_weights_cvar_ic, np.full(max_select - len(portfolio_weights_cvar_ic), np.nan)))
        stock_codes_corr = np.concatenate((stock_codes_ic, np.full(max_select - len(stock_codes_ic), np.nan)))
        portfolio_weights_cvar_corr = np.concatenate(
            (portfolio_weights_cvar_corr, np.full(max_select - len(portfolio_weights_cvar_corr), np.nan)))
        stock_codes_corr = np.concatenate((stock_codes_corr, np.full(max_select - len(stock_codes_corr), np.nan)))
  

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
    weights_data_combined_coef = pd.concat([weights_data_combined_coef, portfolio_weights_df_coef], ignore_index=True)
    return_data_combined_ic = pd.concat([return_data_combined_ic, portfolio_return_data_ic], ignore_index=True)
    weights_data_combined_ic = pd.concat([weights_data_combined_ic, portfolio_weights_df_ic], ignore_index=True)
    return_data_combined_corr = pd.concat([return_data_combined_corr, portfolio_return_data_corr], ignore_index=True)
    weights_data_combined_corr = pd.concat([weights_data_combined_corr, portfolio_weights_df_corr], ignore_index=True)
    return_data_combined_cvar_coef = pd.concat([return_data_combined_cvar_coef, portfolio_return_data_cvar_coef], ignore_index=True)
    weights_data_combined_cvar_coef = pd.concat([weights_data_combined_cvar_coef, portfolio_weights_df_cvar_coef], ignore_index=True)
    return_data_combined_cvar_ic = pd.concat([return_data_combined_cvar_ic, portfolio_return_data_cvar_ic], ignore_index=True)
    weights_data_combined_cvar_ic = pd.concat([weights_data_combined_cvar_ic, portfolio_weights_df_cvar_ic], ignore_index=True)
    return_data_combined_cvar_corr = pd.concat([return_data_combined_cvar_corr, portfolio_return_data_cvar_corr], ignore_index=True)
    weights_data_combined_cvar_corr = pd.concat([weights_data_combined_cvar_corr, portfolio_weights_df_cvar_corr], ignore_index=True)

    # -- evaluation
    ann_excess_return_coef = np.mean(
        return_data_combined_coef[return_data_combined_coef['month'].isin(period_test)]['return']) * 12
    ann_excess_vol_coef = np.std(
        return_data_combined_coef[return_data_combined_coef['month'].isin(period_test)]['return']) * np.sqrt(12)
    info_ratio_coef = ann_excess_return_coef / ann_excess_vol_coef

    ann_excess_return_ic = np.mean(
        return_data_combined_ic[return_data_combined_ic['month'].isin(period_test)]['return']) * 12
    ann_excess_vol_ic = np.std(
        return_data_combined_ic[return_data_combined_ic['month'].isin(period_test)]['return']) * np.sqrt(12)
    info_ratio_ic = ann_excess_return_ic / ann_excess_vol_ic

    ann_excess_return_corr = np.mean(
        return_data_combined_corr[return_data_combined_corr['month'].isin(period_test)]['return']) * 12
    ann_excess_vol_corr = np.std(
        return_data_combined_corr[return_data_combined_corr['month'].isin(period_test)]['return']) * np.sqrt(12)
    info_ratio_corr = ann_excess_return_corr / ann_excess_vol_corr

    ann_excess_return_cvar_coef = np.mean(
        return_data_combined_cvar_coef[return_data_combined_cvar_coef['month'].isin(period_test)]['return']) * 12
    ann_excess_vol_cvar_coef = np.std(
        return_data_combined_cvar_coef[return_data_combined_cvar_coef['month'].isin(period_test)]['return']) * np.sqrt(12)
    info_ratio_cvar_coef = ann_excess_return_cvar_coef / ann_excess_vol_cvar_coef
    ann_excess_return_cvar_ic = np.mean(
        return_data_combined_cvar_ic[return_data_combined_cvar_ic['month'].isin(period_test)]['return']) * 12
    ann_excess_vol_cvar_ic = np.std(
        return_data_combined_cvar_ic[return_data_combined_cvar_ic['month'].isin(period_test)]['return']) * np.sqrt(12)
    info_ratio_cvar_ic = ann_excess_return_cvar_ic / ann_excess_vol_cvar_ic
    ann_excess_return_cvar_corr = np.mean(
        return_data_combined_cvar_corr[return_data_combined_cvar_corr['month'].isin(period_test)]['return']) * 12
    ann_excess_vol_cvar_corr = np.std(
        return_data_combined_cvar_corr[return_data_combined_cvar_corr['month'].isin(period_test)]['return']) * np.sqrt(12)
    info_ratio_cvar_corr = ann_excess_return_cvar_corr / ann_excess_vol_cvar_corr

    print('回归系数')
    print('annual excess return = %.2f' % ann_excess_return_coef)
    print('annual excess volatility = %.2f' % ann_excess_vol_coef)
    print('information ratio = %.2f' % info_ratio_coef)
    print('CVaR_coef')
    print('annual excess return = %.2f' % ann_excess_return_cvar_coef)
    print('annual excess volatility = %.2f' % ann_excess_vol_cvar_coef)
    print('information ratio = %.2f' % info_ratio_cvar_coef)
    print('信息系数')
    print('annual excess return = %.2f' % ann_excess_return_ic)
    print('annual excess volatility = %.2f' % ann_excess_vol_ic)
    print('information ratio = %.2f' % info_ratio_ic)
    print('CVaR_ic')
    print('annual excess return = %.2f' % ann_excess_return_cvar_ic)
    print('annual excess volatility = %.2f' % ann_excess_vol_cvar_ic)
    print('information ratio = %.2f' % info_ratio_cvar_ic)
    print('协方差矩阵')
    print('annual excess return = %.2f' % ann_excess_return_corr)
    print('annual excess volatility = %.2f' % ann_excess_vol_corr)
    print('information ratio = %.2f' % info_ratio_corr)
    print('CVaR_corr')
    print('annual excess return = %.2f' % ann_excess_return_cvar_corr)
    print('annual excess volatility = %.2f' % ann_excess_vol_cvar_corr)
    print('information ratio = %.2f' % info_ratio_cvar_corr)

    ##数据集滚动
    end_date += train_update_months
    if train_data_max_months <= end_date - start_date:
        start_date = end_date - train_data_max_months
    else:
        start_date = start_date
    if end_date + 6 >= 284:
        break

import matplotlib.pyplot as plt

# 绘制曲线图1
plt.plot(return_data_combined_coef['month'], return_data_combined_coef['return'], label='return_1')
plt.plot(return_data_combined_coef['month'], return_data_combined_coef['compound_value'], label='compound_value_1')
# 添加图例和标签
plt.legend()
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Return Data Combined 1')
# 显示图像
plt.show()

# 创建新的图像窗口
plt.figure()
# 绘制曲线图2
plt.plot(return_data_combined_cvar_coef['month'], return_data_combined_cvar_coef['return'], label='return_var_coef')
plt.plot(return_data_combined_cvar_coef['month'], return_data_combined_cvar_coef['compound_value'], label='compound_value_var_coef')
# 添加图例和标签
plt.legend()
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Return Data VaR_Coef')
# 显示图像
plt.show()

# 创建新的图像窗口
plt.figure()
# 绘制曲线图3
plt.plot(return_data_combined_ic['month'], return_data_combined_ic['return'], label='return_2')
plt.plot(return_data_combined_ic['month'], return_data_combined_ic['compound_value'], label='compound_value_2')
# 添加图例和标签
plt.legend()
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Return Data Combined 2')
# 显示图像
plt.show()

# 创建新的图像窗口
plt.figure()
# 绘制曲线图4
plt.plot(return_data_combined_cvar_ic['month'], return_data_combined_cvar_ic['return'], label='return_var_ic')
plt.plot(return_data_combined_cvar_ic['month'], return_data_combined_cvar_ic['compound_value'], label='compound_value_var_ic')
# 添加图例和标签
plt.legend()
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Return Data VaR_ic')
# 显示图像
plt.show()

# 创建新的图像窗口
plt.figure()
# 绘制曲线图5
plt.plot(return_data_combined_corr['month'], return_data_combined_corr['return'], label='return_3')
plt.plot(return_data_combined_corr['month'], return_data_combined_corr['compound_value'], label='compound_value_3')
# 添加图例和标签
plt.legend()
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Return Data Combined 3')
# 显示图像
plt.show()

# 创建新的图像窗口
plt.figure()
# 绘制曲线图6
plt.plot(return_data_combined_cvar_corr['month'], return_data_combined_cvar_corr['return'], label='return_var_corr')
plt.plot(return_data_combined_cvar_corr['month'], return_data_combined_cvar_corr['compound_value'], label='compound_value_var_corr')
# 添加图例和标签
plt.legend()
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Return Data VaR_corr')
# 显示图像
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