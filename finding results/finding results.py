#导入包
import numpy as np
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')

#hs300名单 2022.1.1之前
hs300=pd.read_csv(r'../hs300_2005-2022/hs300_monthly.csv',dtype = object)
hs300=hs300.drop('Unnamed: 0',axis=1)
hs300=hs300.rename(columns={'stock': 'code'})
hs300['month']=hs300['month'].astype(int)

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
return_data_combined_1 = pd.DataFrame()
weights_data_combined_1 = pd.DataFrame()
return_data_combined_2 = pd.DataFrame()
weights_data_combined_2 = pd.DataFrame()
return_data_combined_3 = pd.DataFrame()
weights_data_combined_3 = pd.DataFrame()

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

    #因子信息系数
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
    factor_weights_2 = np.array(weighted_factors_coefficient['Weight'])

    #基于回归系数
    # 获取回归系数
    coefficients = model.coef_
    # 计算因子权重
    factor_weights_1 = abs(coefficients) / sum(abs(coefficients))

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
    factor_weights_3 = np.array(factor_weights_sorted['Weight'])

    #样本外预测
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
    #优化器
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

    max_select = 20  # 最长的数据长度
    # 创建一个空的DataFrame来存储最优投资组合权重
    portfolio_weights_df_1 = pd.DataFrame()
    portfolio_weights_df_2 = pd.DataFrame()
    portfolio_weights_df_3 = pd.DataFrame()

    # 创建一个空的DataFrame
    portfolio_return_data_1 = pd.DataFrame(columns=['month', 'return', 'compound_value'])
    portfolio_return_data_2 = pd.DataFrame(columns=['month', 'return', 'compound_value'])
    portfolio_return_data_3 = pd.DataFrame(columns=['month', 'return', 'compound_value'])
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
        scores_1 = X_for_score.dot(factor_weights_1)
        scores_2 = X_for_score.dot(factor_weights_2)
        scores_3 = X_for_score.dot(factor_weights_3)
        n = 30
        selected_stocks_1 = scores_1.nlargest(n)
        selected_stocks_2 = scores_2.nlargest(n)
        selected_stocks_3 = scores_3.nlargest(n)
        # 添加month和stock列
        selected_stocks_1 = pd.DataFrame(
            {'code': data_for_score.loc[selected_stocks_1.index, 'code'], 'score': selected_stocks_1.values})
        selected_stocks_1.set_index('code', inplace=True)
        selected_stocks_2 = pd.DataFrame(
            {'code': data_for_score.loc[selected_stocks_2.index, 'code'], 'score': selected_stocks_2.values})
        selected_stocks_2.set_index('code', inplace=True)
        selected_stocks_3 = pd.DataFrame(
            {'code': data_for_score.loc[selected_stocks_3.index, 'code'], 'score': selected_stocks_3.values})
        selected_stocks_3.set_index('code', inplace=True)

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
        top_20_stocks_idx_1 = selected_stocks_1.index[:20]
        top_20_stocks_return_1 = combined_return_data.loc[combined_return_data.index.intersection(top_20_stocks_idx_1)]
        top_20_stocks_return_1 = top_20_stocks_return_1.dropna()
        top_20_stocks_idx_2 = selected_stocks_2.index[:20]
        top_20_stocks_return_2 = combined_return_data.loc[combined_return_data.index.intersection(top_20_stocks_idx_2)]
        top_20_stocks_return_2 = top_20_stocks_return_2.dropna()
        top_20_stocks_idx_3 = selected_stocks_3.index[:20]
        top_20_stocks_return_3 = combined_return_data.loc[combined_return_data.index.intersection(top_20_stocks_idx_3)]
        top_20_stocks_return_3 = top_20_stocks_return_3.dropna()

        # 计算收益率的协方差矩阵
        cov_matrix_1 = top_20_stocks_return_1.T.cov()
        cov_matrix_2 = top_20_stocks_return_2.T.cov()
        cov_matrix_3 = top_20_stocks_return_3.T.cov()
        # 获取 i_month_1 对应的收益均值和股票代码
        mean_returns_1 = top_20_stocks_return_1.loc[:, i_month_1]
        stock_codes_1 = top_20_stocks_return_1.index.tolist()
        mean_returns_2 = top_20_stocks_return_2.loc[:, i_month_1]
        stock_codes_2 = top_20_stocks_return_2.index.tolist()
        mean_returns_3 = top_20_stocks_return_3.loc[:, i_month_1]
        stock_codes_3 = top_20_stocks_return_3.index.tolist()

        # 使用均值-方差模型计算最优投资组合权重
        portfolio_weights_1 = calculate_portfolio_weights(mean_returns_1, cov_matrix_1)
        portfolio_weights_2 = calculate_portfolio_weights(mean_returns_2, cov_matrix_2)
        portfolio_weights_3 = calculate_portfolio_weights(mean_returns_3, cov_matrix_3)

        # 计算组合收益
        y_curr_month_return_1 = y_curr_month.loc[y_curr_month.index.intersection(stock_codes_1)]['curr_return']
        portfolio_return_1 = np.dot(portfolio_weights_1.T, y_curr_month_return_1)
        y_curr_month_return_2 = y_curr_month.loc[y_curr_month.index.intersection(stock_codes_2)]['curr_return']
        portfolio_return_2 = np.dot(portfolio_weights_2.T, y_curr_month_return_2)
        y_curr_month_return_3 = y_curr_month.loc[y_curr_month.index.intersection(stock_codes_3)]['curr_return']
        portfolio_return_3 = np.dot(portfolio_weights_3.T, y_curr_month_return_3)

        # 计算累计值
        if portfolio_return_data_1.empty:
            cumulative_value_1 = 1 + portfolio_return_1
        else:
            previous_cumulative_value = portfolio_return_data_1['compound_value'].iloc[-1]
            cumulative_value_1 = previous_cumulative_value * (1 + portfolio_return_1)
        if portfolio_return_data_2.empty:
            cumulative_value_2 = 1 + portfolio_return_2
        else:
            previous_cumulative_value = portfolio_return_data_2['compound_value'].iloc[-1]
            cumulative_value_2 = previous_cumulative_value * (1 + portfolio_return_2)
        if portfolio_return_data_3.empty:
            cumulative_value_3 = 1 + portfolio_return_3
        else:
            previous_cumulative_value = portfolio_return_data_3['compound_value'].iloc[-1]
            cumulative_value_3 = previous_cumulative_value * (1 + portfolio_return_3)


        # 将收益数据添加到DataFrame中
        row_1 = {'month': i_month_1, 'return': portfolio_return_1, 'compound_value': cumulative_value_1}
        portfolio_return_data_1 = pd.concat([portfolio_return_data_1, pd.DataFrame(row_1, index=[0])], ignore_index=True)
        row_2 = {'month': i_month_1, 'return': portfolio_return_2, 'compound_value': cumulative_value_2}
        portfolio_return_data_2 = pd.concat([portfolio_return_data_2, pd.DataFrame(row_2, index=[0])], ignore_index=True)
        row_3 = {'month': i_month_1, 'return': portfolio_return_3, 'compound_value': cumulative_value_3}
        portfolio_return_data_3 = pd.concat([portfolio_return_data_3, pd.DataFrame(row_3, index=[0])], ignore_index=True)

        # 使用NaN将数据补齐至长度为20
        portfolio_weights_1 = np.concatenate((portfolio_weights_1, np.full(max_select - len(portfolio_weights_1), np.nan)))
        stock_codes_1 = np.concatenate((stock_codes_1, np.full(max_select - len(stock_codes_1), np.nan)))
        portfolio_weights_2 = np.concatenate((portfolio_weights_2, np.full(max_select - len(portfolio_weights_2), np.nan)))
        stock_codes_2 = np.concatenate((stock_codes_2, np.full(max_select - len(stock_codes_2), np.nan)))
        portfolio_weights_3 = np.concatenate((portfolio_weights_3, np.full(max_select - len(portfolio_weights_3), np.nan)))
        stock_codes_3 = np.concatenate((stock_codes_3, np.full(max_select - len(stock_codes_3), np.nan)))

        # 将最优投资组合权重和股票代码添加到DataFrame中
        portfolio_weights_df_1[str(i_month_1) + '_code'] = stock_codes_1
        portfolio_weights_df_1[str(i_month_1)] = portfolio_weights_1
        portfolio_weights_df_2[str(i_month_1) + '_code'] = stock_codes_2
        portfolio_weights_df_2[str(i_month_1)] = portfolio_weights_2
        portfolio_weights_df_3[str(i_month_1) + '_code'] = stock_codes_3
        portfolio_weights_df_3[str(i_month_1)] = portfolio_weights_3
    return_data_combined_1 = pd.concat([return_data_combined_1, portfolio_return_data_1], ignore_index=True)
    weights_data_combined_1 = pd.concat([weights_data_combined_1, portfolio_weights_df_1], ignore_index=True)
    return_data_combined_2 = pd.concat([return_data_combined_2, portfolio_return_data_2], ignore_index=True)
    weights_data_combined_2 = pd.concat([weights_data_combined_2, portfolio_weights_df_2], ignore_index=True)
    return_data_combined_3 = pd.concat([return_data_combined_3, portfolio_return_data_3], ignore_index=True)
    weights_data_combined_3 = pd.concat([weights_data_combined_3, portfolio_weights_df_3], ignore_index=True)

    # -- evaluation
    ann_excess_return_1 = np.mean(return_data_combined_1[return_data_combined_1['month'].isin(period_test)]['return']) * 12
    ann_excess_vol_1 = np.std(return_data_combined_1[return_data_combined_1['month'].isin(period_test)]['return']) * np.sqrt(12)
    info_ratio_1 = ann_excess_return_1 / ann_excess_vol_1

    ann_excess_return_2 = np.mean(return_data_combined_2[return_data_combined_2['month'].isin(period_test)]['return']) * 12
    ann_excess_vol_2 = np.std(return_data_combined_2[return_data_combined_2['month'].isin(period_test)]['return']) * np.sqrt(12)
    info_ratio_2 = ann_excess_return_2 / ann_excess_vol_2

    ann_excess_return_3 = np.mean(return_data_combined_3[return_data_combined_3['month'].isin(period_test)]['return']) * 12
    ann_excess_vol_3 = np.std(return_data_combined_3[return_data_combined_3['month'].isin(period_test)]['return']) * np.sqrt(12)
    info_ratio_3 = ann_excess_return_3 / ann_excess_vol_3
    print('回归系数')
    print('annual excess return = %.2f' % ann_excess_return_1)
    print('annual excess volatility = %.2f' % ann_excess_vol_1)
    print('information ratio = %.2f' % info_ratio_1)
    print('信息系数')
    print('annual excess return = %.2f' % ann_excess_return_2)
    print('annual excess volatility = %.2f' % ann_excess_vol_2)
    print('information ratio = %.2f' % info_ratio_2)
    print('协方差矩阵')
    print('annual excess return = %.2f' % ann_excess_return_3)
    print('annual excess volatility = %.2f' % ann_excess_vol_3)
    print('information ratio = %.2f' % info_ratio_3)

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
plt.plot(return_data_combined_1['month'], return_data_combined_1['return'], label='return_1')
plt.plot(return_data_combined_1['month'], return_data_combined_1['compound_value'], label='compound_value_1')
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
plt.plot(return_data_combined_2['month'], return_data_combined_2['return'], label='return_2')
plt.plot(return_data_combined_2['month'], return_data_combined_2['compound_value'], label='compound_value_2')
# 添加图例和标签
plt.legend()
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Return Data Combined 2')
# 显示图像
plt.show()
# 创建新的图像窗口
plt.figure()
# 绘制曲线图3
plt.plot(return_data_combined_3['month'], return_data_combined_3['return'], label='return_3')
plt.plot(return_data_combined_3['month'], return_data_combined_3['compound_value'], label='compound_value_3')
# 添加图例和标签
plt.legend()
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Return Data Combined 3')
# 显示图像
plt.show()

print('回归系数下因子权重收益',return_data_combined_1['return'].mean)
print('回归系数下因子权重累计价值',return_data_combined_1['compound_value'].mean)
print('信息系数下因子权重收益',return_data_combined_2['return'].mean)
print('信息系数下因子权重累计价值',return_data_combined_2['compound_value'].mean)
print('协方差矩阵下因子权重收益',return_data_combined_3['return'].mean)
print('协方差矩阵下因子权重累计价值',return_data_combined_3['compound_value'].mean)






