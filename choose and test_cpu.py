#导入包
import numpy as np
import pandas as pd
import torch

## 定义参数类
# -- define a class including all parameters

class Para():
    method = 'XGBOOST-C'  # 'LOGI' 'XGBOOST-C' 'LR'
    month_in_sample = range(82, 153 + 1)  # -- return 82~153 72 months
    month_test = range(154, 293 + 1)  # -- return 154~293 140 months

    percent_select = [0.3, 0.3]  # -- 30% positive samples, 30% negative samples
    percent_cv = 0.1  # -- percentage of cross validation samples 交叉验证的样本比例
    path_data = './data/csv_01/'
    path_results = './results/'
    seed = 42  # -- random seed
    n_stock = 5166

    logi_c = 0.0006  # -- logistic regression parameter
    xgbc_n_estimators = 100  # -- xgboost classifier parameter
    xgbc_learning_rate = 0.2 # 0.1 # -- xgboost classifier parameter
    xgbc_subsample_C = 0.95  # -- xgboost classifier parameter
    xgbc_max_depth = 4 #3  # -- xgboost classifier parameter
para = Para()

# 将输入数据转移到与XGBoost模型所在设备一致的设备上
#device = torch.device('cuda:0')  # 指定设备为cuda:0，可以根据实际情况进行修改
# 将需要转移的数据转移到指定设备上
# 示例代码中的path_data和path_results是字符串变量，不需要转移
# 如果有其他需要转移的数据，可以使用类似的方式进行转移
#para.month_in_sample = torch.tensor(list(range(82, 153 + 1))).to(device)
#para.month_test = torch.tensor(list(range(154, 293 + 1))).to(device)
#para.percent_select = torch.tensor([0.3, 0.3]).to(device)
#para.percent_cv = torch.tensor(0.1).to(device)

## 生成二分类标签函数
# -- function, label data
def label_data(data):
    # -- label data
    data['return_bin'] = np.nan

    # -- sort by return
    data = data.sort_values(by='return', ascending=False)

    # -- decide the amount of stocks selected
    n_stock_select = np.multiply(para.percent_select, data.shape[0])  # 计算矩阵的内积
    #n_stock_select = torch.mul(para.percent_select, data.shape[0])
    n_stock_select = np.around(n_stock_select).astype(int)  # 结果为整数

    # -- assign 1 or 0
    data.iloc[0:n_stock_select[0], -1] = 1
    data.iloc[-n_stock_select[1]:, -1] = 0

    # -- remove other stocks
    data = data.dropna(axis=0)

    return data

# 设置滚动训练时间
train_data_min_months = 72  # 每次模型训练所用数据最少不低于
train_data_max_months = 108  # 每次模型训练所用数据最大不超过
train_update_months = 6  # 设置更新周期
start_date = 82  # 第一次滚动训练开始日期
end_date = start_date + train_data_min_months  # 第一次滚动训练结束日期

i = 0

## 样本外预测
# -- predict

d = pd.DataFrame([np.nan] * np.zeros((para.n_stock, end_date)))
y_true_test = d
y_pred_test = d
y_score_test = d

while end_date <= 293:
    period_train = range(start_date, end_date + 1)

    ## 生成样本内数据集
    # -- generate in-sample data
    for i_month in period_train:
        # -- load csv
        file_name = para.path_data + str(i_month) + '.csv'
        data_curr_month = pd.read_csv(file_name, header=0)  # 设置表头
        para.n_stock = data_curr_month.shape[0]

        # -- remove nan
        data_curr_month = data_curr_month.dropna(axis=0)

        # -- label data
        data_curr_month = label_data(data_curr_month)  # 调用函数，

        # -- merge
        if i_month == period_train[0]:  # -- first month
            data_in_sample = data_curr_month
        else:
            data_in_sample = pd.concat((data_in_sample, data_curr_month), axis=0)

    # 样本内数据集
    # -- generate in-sample data
    X_in_sample = data_in_sample.loc[:, 'EP':'bias']  # 提取数据

    # -- classification
    if para.method in ['LOGI', 'XGBOOST-C']:
        y_in_sample = data_in_sample.loc[:, 'return_bin']

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

    ## 设置模型
    # -- set model
    # 有监督，输出是分类型数据，逻辑回归
    # -- logistic regression
    if para.method == 'LOGI':
        from sklearn import linear_model
        model = linear_model.LogisticRegression(C=para.logi_c)

    # 集成算法
    # -- XGBoost Classifier 分类
    if para.method == 'XGBOOST-C':
        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=para.seed,
                              n_estimators=para.xgbc_n_estimators,  # 迭代次数
                              learning_rate=para.xgbc_learning_rate,  # 减小每一步的权重，提高robust
                              subsample=para.xgbc_subsample_C,  # 随机采样比例
                              max_depth=para.xgbc_max_depth)# 控制深度，避免过拟合
                              #tree_method="hist", device="cuda")
    # 有监督，连续型数据
    # -- linear regression
    if para.method == 'LR':
        from sklearn import linear_model
        model = linear_model.LinearRegression(fit_intercept=True)  # 计算偏置
    ## 训练模型，交叉验证
    # -- train model, and perform cross validation
    # -- classification
    if para.method in ['LOGI', 'XGBOOST-C']:
        model.fit(X_train, y_train)  # 投入训练数据，fit()函数
        # -- y_pred: binary format; y_score: continious format
        y_pred_train = model.predict(X_train)  # 预测输出结果，得到预测类别结果
        y_score_train = model.predict_proba(X_train)[:, 1]  # 得到预测概率
        if para.percent_cv > 0:  # 0.1比例的验证集
            y_pred_cv = model.predict(X_cv)
            y_score_cv = model.predict_proba(X_cv)[:, 1]
            # 返回预测属于某标签的概率

    # -- regression
    if para.method in ['LR']:
        model.fit(X_train, y_train)
        y_score_train = model.predict(X_train)

        if para.percent_cv > 0:
            y_score_cv = model.predict(X_cv)

    ##样本外预测集
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
        # X_curr_month = pca.transform(X_curr_month)
        # -- predict and get predicted probability
        # -- classification
        if para.method in ['LOGI', 'XGBOOST-C']:
            y_pred_curr_month = model.predict(X_curr_month)
            y_score_curr_month = model.predict_proba(X_curr_month)[:, 1]
        # -- linear regression
        if para.method in ['LR', ]:
            y_score_curr_month = model.predict(X_curr_month)
        # -- store real and predicted return
        a.iloc[data_curr_month.index, i_month - 1] = data_curr_month['return'][data_curr_month.index]
        if para.method in ['LOGI', 'XGBOOST-C']:
            b.iloc[data_curr_month.index, i_month - 1] = y_pred_curr_month
        c.iloc[data_curr_month.index, i_month - 1] = y_score_curr_month
    # 合并运行生成的数据
    y_true_test = pd.concat([y_true_test, a.iloc[:, -7:-1]], axis=1)
    y_pred_test = pd.concat([y_pred_test, b.iloc[:, -7:-1]], axis=1)
    y_score_test = pd.concat([y_score_test, c.iloc[:, -7:-1]], axis=1)

    ## 模型评价
    # -- evaluate
    if para.method in ['LOGI', 'XGBOOST-C']:
        from sklearn import metrics
        print('training set, accuracy = %.2f' % metrics.accuracy_score(y_train, y_pred_train))
        print('training set, AUC = %.2f' % metrics.roc_auc_score(y_train, y_score_train))
        if para.percent_cv > 0:
            print('cv set, accuracy = %.2f' % metrics.accuracy_score(y_cv, y_pred_cv))
            print('cv set, AUC = %.2f' % metrics.roc_auc_score(y_cv, y_score_cv))
        for i_month in period_test:
            # -- 4 types of y
            # -- y_true_*: true continious
            # -- y_*: true binary
            # -- y_pred_*: predicted binary
            # -- y_score_*: predicted continious
            y_true_curr_month = pd.DataFrame({'return': y_true_test.iloc[:, i_month - 1]})
            y_pred_curr_month = y_pred_test.iloc[:, i_month - 1]
            y_score_curr_month = y_score_test.iloc[:, i_month - 1]
            # -- remove nan
            y_true_curr_month = y_true_curr_month.dropna(axis=0)
            # -- label data
            y_curr_month = label_data(y_true_curr_month)['return_bin']
            # -- only select best and worst 30% data
            y_pred_curr_month = y_pred_curr_month[y_curr_month.index]
            y_score_curr_month = y_score_curr_month[y_curr_month.index]
            print('testing set, month %d, accuracy = %.2f' % (
            i_month, metrics.accuracy_score(y_curr_month, y_pred_curr_month)))
            print('testing set, month %d, AUC = %.2f' % (
            i_month, metrics.roc_auc_score(y_curr_month, y_score_curr_month)))
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

    ##数据集滚动
    end_date += train_update_months
    if train_data_max_months <= end_date - start_date:
        start_date = end_date - train_data_max_months
    else:
        start_date = start_date
    if end_date + 6 >= 293:
        break

## 简易回测
# -- simple strategy, select 50 stocks every month, equally weighted
para.n_stock_select = 50
period_test = range(155, 293)
strategy = pd.DataFrame({'return': [0] * (period_test[-1] + 1), 'value': [1] * (period_test[-1] + 1)})

for i_month in period_test:
    # -- get real and predicted return
    y_true_curr_month = y_true_test.iloc[:, i_month - 1]
    y_score_curr_month = y_score_test.iloc[:, i_month - 1]

    # -- sort predicted return, and choose the best 50
    y_score_curr_month = y_score_curr_month.sort_values(ascending=False)
    index_select = y_score_curr_month[0:para.n_stock_select].index

    # -- take the average return as the return of next month
    strategy.loc[i_month - 1, 'return'] = np.mean(y_true_curr_month[index_select])

# -- compute the compund value of the strategy
strategy['value'] = (strategy['return'] + 1).cumprod()

# -- plot the value
import matplotlib.pyplot as plt

plt.plot(range(155, 292 + 1), strategy.loc[range(155, 292 + 1), 'value'], 'r-')
plt.show()

# -- evaluation
ann_excess_return = np.mean(strategy.loc[period_test, 'return']) * 12
ann_excess_vol = np.std(strategy.loc[period_test, 'return']) * np.sqrt(12)
info_ratio = ann_excess_return / ann_excess_vol

print('annual excess return = %.2f' % ann_excess_return)
print('annual excess volatility = %.2f' % ann_excess_vol)
print('information ratio = %.2f' % info_ratio)

## 训练模型，交叉验证
# -- train model, and perform cross validation
# -- classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

n_folds = 20  # 设置交叉检验的次数
if para.method in ['LOGI', 'XGBOOST-C']:
    cv_score_list = []  # 交叉检验结果列表
    model.fit(X_train, y_train)
    # -- y_pred: binary format; y_score: continious format
    y_pred_train = model.predict(X_train)
    y_score_train = model.predict_proba(X_train)[:, 1]

    if para.percent_cv > 0:
        # 交叉验证
        y_pred_cv = model.predict(X_cv)
        y_score_cv = model.predict_proba(X_cv)[:, 1]
        scores = cross_val_score(model, X_cv, y_cv, cv=n_folds, scoring='r2')
        cv_score_list.append(scores)
        # 调参
        learning_rate = [0.01, 0.1, 0.2, 0.3]  # 学习率
        gamma = [1, 0.1, 0.01]
        max_depth = [2, 3, 4]

        param_grid = dict(learning_rate=learning_rate, gamma=gamma, max_depth=max_depth)  # 转化为字典格式，网络搜索要求
        model_gs = GridSearchCV(model, param_grid, scoring='neg_log_loss', n_jobs=-1, cv=5)
        model_gs.fit(X_train, y_train)
        print(cv_score_list)
        print('Best score is:', model_gs.best_score_)
        print('Best parameter is:', model_gs.best_params_)

    # -- regression
if para.method in ['LR']:
    model.fit(X_train, y_train)
    y_score_train = model.predict(X_train)

    if para.percent_cv > 0:
        y_score_cv = model.predict(X_cv)
        scores = cross_val_score(model, X_cv, y_cv, cv=n_folds, scoring='r2')
        cv_score_list.append(scores)