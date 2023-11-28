#TODO: 1.筛选有效因子 2.均值方差投资策略 3.VaR 及 CVaR计算；

#导入包
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

## 定义参数类
# -- define a class including all parameters
class Para():
    method = 'LR'
    month_in_sample = range(82, 153 + 1)  # -- return 82~153 72 months
    month_test = range(154, 293 + 1)  # -- return 154~293 140 months

    percent_select = [0.3, 0.3]  # -- 30% positive samples, 30% negative samples
    percent_cv = 0.1  # -- percentage of cross validation samples 交叉验证的样本比例
    path_data = './data/csv_01/'
    path_results = './results/'
    seed = 42  # -- random seed
    n_stock = 5166
para = Para()

train_data_min_months = 72  # 每次模型训练所用数据最少不低于
train_data_max_months = 108  # 每次模型训练所用数据最大不超过
train_update_months = 6  # 设置更新周期
start_date = 82  # 第一次滚动训练开始日期
end_date = start_date + train_data_min_months  # 第一次滚动训练结束日期

## 生成样本内数据集
# -- generate in-sample data
period_train = range(start_date, end_date + 1)
a1 = pd.DataFrame([np.nan] * np.zeros((para.n_stock, period_train[-1] )))
for i_month in period_train:
    # -- load csv
    file_name = para.path_data + str(i_month) + '.csv'
    data_curr_month = pd.read_csv(file_name, header=0)  # 设置表头
    para.n_stock = data_curr_month.shape[0]
    # -- remove nan
    data_curr_month = data_curr_month.dropna(axis=0)
    a1.iloc[data_curr_month.index, i_month-1] = data_curr_month['return'][data_curr_month.index]
    # -- merge
    if i_month == period_train[0]:  # -- first month
        data_in_sample = data_curr_month
    else:
        data_in_sample = pd.concat((data_in_sample, data_curr_month), axis=0)

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
if para.method in ['LR']:
    model = linear_model.LinearRegression(fit_intercept=True)  # 计算偏置（截距）
    # -- regression
    model.fit(X_train, y_train)
    y_score_train = model.predict(X_train)
    if para.percent_cv > 0:
        y_score_cv = model.predict(X_cv)

## 样本外预测
# -- predict
d = pd.DataFrame([np.nan] * np.zeros((para.n_stock, end_date)))
y_true_test = d
y_pred_test = d
y_score_test = d
test_date_start = end_date + 1
test_date_end = end_date + 6
period_test = range(test_date_start, test_date_end + 1)
a = pd.DataFrame([np.nan] * np.zeros((para.n_stock, period_test[-1] + 1)))
b = pd.DataFrame([np.nan] * np.zeros((para.n_stock, period_test[-1] + 1)))
c = pd.DataFrame([np.nan] * np.zeros((para.n_stock, period_test[-1] + 1)))
for i_month in period_test:
    # -- load
    file_name = para.path_data + str(i_month) + '.csv'
    data_curr_month = pd.read_csv(file_name, header=0)
    # -- remove nan
    data_curr_month = data_curr_month.dropna(axis=0)
    # -- generate X
    X_curr_month = data_curr_month.loc[:, 'EP':'bias']
    # -- pca
    #X_curr_month = pca.transform(X_curr_month)
    # -- predict and get predicted probability
    # -- linear regression
    if para.method in ['LR']:
        y_score_curr_month = model.predict(X_curr_month)
    # -- store real and predicted return
    a.iloc[data_curr_month.index, i_month - 1] = data_curr_month['return'][data_curr_month.index]
    c.iloc[data_curr_month.index, i_month - 1] = y_score_curr_month
# 合并运行生成的数据
y_true_test = pd.concat([y_true_test, a.iloc[:, -7:-1]], axis=1)
y_pred_test = pd.concat([y_pred_test, b.iloc[:, -7:-1]], axis=1)
y_score_test = pd.concat([y_score_test, c.iloc[:, -7:-1]], axis=1)

if para.method in ['LR']:
    y_train.index = range(len(y_train))
    y_score_train = pd.Series(y_score_train)
    print('training set, ic = %.2f' % y_train.corr(y_score_train))
    if para.percent_cv > 0:
        y_cv.index = range(len(y_cv))
        y_score_cv = pd.Series(y_score_cv)
        print('cv set, ic = %.2f' % y_cv.corr(y_score_cv))
    for i_month in period_test:
        y_true_curr_month = y_true_test.iloc[:, i_month - 1]
        y_score_curr_month = y_score_test.iloc[:, i_month - 1]
        print('testing set, month %d, ic = %.2f' % (i_month, y_true_curr_month.corr(y_score_curr_month)))

#合并过去6个月的数据，与预测的未来三个月的数据
y_true_pred=pd.concat([a1, c.iloc[:, -7:-1]], axis=1)

# 优化器_1
from scipy.optimize import minimize
def mean_variance_optimization(returns, n_stock_select):
    n_assets = returns.shape[0]
    # 定义目标函数
    def objective(weights):
        mean_return = np.mean(returns, axis=1)
        cov_matrix = np.cov(returns)
        portfolio_return = np.dot(weights, mean_return)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        return portfolio_variance
    # 定义约束条件
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    # 设置初始权重猜测值
    init_weights = np.ones(n_assets) / n_assets
    # 执行优化
    result = minimize(objective, init_weights, constraints=constraints)
    # 返回优化后的权重
    return result.x
    # 获取前n_stock_select个股票的索引
    selected_indexes = np.argpartition(optimized_weights, -n_stock_select)[-n_stock_select:]
    # 计算选取股票的权重
    selected_weights = optimized_weights[selected_indexes]
    # 返回选取的股票权重和索引
    return selected_weights, selected_indexes
def optimize_weights(returns, selected_indexes):
    n_assets = len(selected_indexes)
    selected_returns = returns[selected_indexes]
    # 定义目标函数
    def objective(weights):
        mean_return = np.mean(selected_returns, axis=1)
        cov_matrix = np.cov(selected_returns)
        portfolio_return = np.dot(weights, mean_return)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        return portfolio_variance
    # 定义约束条件
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    # 使用第一次优化的结果作为初始权重猜测值
    init_weights = np.ones(n_assets) / n_assets
    # 执行第二次优化
    result = minimize(objective, init_weights, constraints=constraints)
    # 返回最优权重
    return result.x

# 初始化策略DataFrame
strategy = pd.DataFrame({'return': [0] * (period_test[-1] + 1), 'value': [1] * (period_test[-1] + 1)})
# 遍历测试期间执行策略
for i_month in period_test:
    # 获取当前月份的真实股票数据
    y_true_curr_month = y_true_test.iloc[:, i_month - 1]
    # 过去6个月与预测6个月的数据
    y_comb_dec = y_true_pred.iloc[:, -12:]
    # 去空缺值
    decision_matrix = y_comb_dec.dropna(axis=0)
    # 执行均值-方差优化以确定权重
    optimization_result = mean_variance_optimization(decision_matrix.values, 10)
    weights = optimization_result[0]
    selected_indexes = optimization_result[1]# 选择10支股票
    optimized_weights = optimize_weights(decision_matrix.values, selected_indexes.tolist())
    # 根据权重大于0的股票确定选取的股票索引
    selected_indexes = decision_matrix.index[weights > 0]
    # 将weights和selected_indexes转换为DataFrame
    weights_df = pd.DataFrame(optimized_weights, columns=['Weights'])
    selected_indexes_df = pd.DataFrame(selected_indexes, columns=['Selected Indexes'])
    # 生成文件名
    filename1 = str(i_month) + 'weights.csv'
    filename2 = str(i_month) + 'index.csv'
    # 将DataFrame保存为CSV文件
    weights_df.to_csv(filename1, index=False)
    selected_indexes_df.to_csv(filename2, index=False)
    # 计算当月的真实收益
    #returns_curr_month = y_true_curr_month[selected_indexes].mean()
    #portfolio_returns_curr_month = np.dot(weights[weights > 0], np.array([returns_curr_month]))
    # 更新策略DataFrame
    #strategy.loc[i_month - 1, 'stocks'] = ', '.join(map(str, selected_indexes))
    #strategy.loc[i_month - 1, 'return'] = portfolio_returns_curr_month
# 计算总收益
#total_return = strategy['return'].sum()
#print(strategy)