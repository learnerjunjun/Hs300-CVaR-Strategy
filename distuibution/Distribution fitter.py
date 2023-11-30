#导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
#数据导入
path_data = '../data/csv_01/'
start_date = 82
end_date = 294
period_train=range(start_date,end_date+1)
for i_month in period_train:
    #-- load csv
    file_name = path_data + str(i_month) + '.csv'
    data_curr_month = pd.read_csv(file_name, header = 0)#设置表头
    n_stock = data_curr_month.shape[0]
    #-- remove nan
    data_curr_month = data_curr_month.dropna(axis=0)
    #-- merge
    if i_month == period_train[0]: #-- first month
        data_in_sample = data_curr_month
    else:
        data_in_sample = pd.concat((data_in_sample,data_curr_month), axis=0)
#历史收益数据截取
return_in_sample = data_in_sample.loc[:,'month':'return']
return_in_sample = return_in_sample.sort_values(['stock','month'])
from fitter import Fitter
# 假设df是包含股票数据的DataFrame，具有'month'、'stock'、'status'和'return'列
df = return_in_sample
# 获取唯一的股票名称
unique_stocks = df['stock'].unique()
# 从中随机选择五个股票
selected_stocks = pd.Series(unique_stocks).sample(n=10, random_state=0).tolist()
# 定义要拟合的分布列表
distributions = ['expon', 'gamma','dgamma', 'norm','lognorm','skewnorm','laplace_asymmetric','laplace','loglaplace','genhyperbolic','tukeylambda','johnsonsu']
# 创建一个空的DataFrame用于存储拟合结果
fit_results = pd.DataFrame(columns=['Stock', 'Best Fit Distribution', 'Best Fit Parameters','Alpha','Quantile'])
# 遍历选定的股票
for stock in selected_stocks:
    stock_data = df[df['stock'] == stock]
    returns = stock_data['return']
    # 使用Fitter库拟合分布
    f = Fitter(returns,distributions=distributions,bins=50)
    #f = Fitter(returns, timeout=100)
    f.fit()
    f.summary()  # 返回排序好的分布拟合质量（拟合效果从好到坏）,并绘制数据分布和Nbest分布
    #f.df_errors  # 返回这些分布的拟合质量（均方根误差的和）
    #f.fitted_param  # 返回拟合分布的参数
    #f.fitted_pdf  # 使用最适合数据分布的分布参数生成的概率密度
    # 获取最佳拟合分布和参数
    best_summary = f.get_best(method='sumsquare_error')
    best_fit_name = list(best_summary.keys())[0]
    best_fit_params = list(best_summary.values())[0]
    # 计算分位数 / VaR
    alpha=1 #置信水平下的分位数
    returns_np=np.array(returns)
    quantile=np.percentile(returns_np,alpha)
    print(quantile)
    # 将拟合结果存储到DataFrame中
    fit_results.loc[len(fit_results)] = [stock, best_fit_name, best_fit_params,alpha,quantile]
    #绘图
    #f.hist()  # 绘制组数=bins的标准化直方图
    #f.plot_pdf(names=None, Nbest=3, lw=2)  # 绘制分布的概率密度函数
    plt.xlabel('Values')
    plt.ylabel('PDF or Frequence')
    plt.title('Data Distribution')
    filename = stock + ".png"
    plt.savefig(filename, dpi=300)
    plt.show()
print(fit_results.to_string())
fit_results.to_excel("distribution and quantile.xlsx", index=False)

# Summary: 从4848只股票中随机选取10个股票的收益分布并非正态分布
# Ressults:
# 1.{'laplace_asymmetric': {'kappa': 0.804808257327414, 'loc': -0.028237365914254044, 'scale': 0.07258159259853167}}
# 2.{'genhyperbolic': {'p': 9.415813694597688, 'a': 0.0008811038514943892, 'b': 4.79864127069472e-05, 'loc': -0.01637876097454328, 'scale': 1.6828948682258248e-05}}
# 3.{'genhyperbolic': {'p': -0.19367876319304736, 'a': 0.34996556155164094, 'b': 0.10195329562265809, 'loc': -0.02292908464378212, 'scale': 0.057134202860838555}}
# 4.{'laplace_asymmetric': {'kappa': 0.8567977791270114, 'loc': -0.035844882806033845, 'scale': 0.09289283989008722}}
# 5.{'dgamma': {'a': 0.6500770716900943, 'loc': 0.006246290000000001, 'scale': 0.08537341704153789}}
# 6.{'loglaplace': {'c': 5.7663364523688205, 'loc': -0.5170124094757588, 'scale': 0.5125570338522736}}
# 7.{'genhyperbolic': {'p': 0.6491938935454755, 'a': 2.8599456624473387e-10, 'b': 2.1215505907123596e-10, 'loc': -0.07644990000797401, 'scale': 9.304192611400532e-12}}
# 8.{'skewnorm': {'a': 2.759384057353138, 'loc': -0.12462884934077859, 'scale': 0.1749000774381379}}
# 9.{'tukeylambda': {'lam': 1.197044768918491, 'loc': 0.009560623113440121, 'scale': 0.2917082908878366}}
# 10.{'johnsonsu': {'a': -0.43284121875181014, 'b': 1.1016306805556269, 'loc': -0.03199115272939734, 'scale': 0.06427318561469116}}