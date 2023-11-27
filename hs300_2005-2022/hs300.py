#导入包
import numpy as np
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')

# 沪深300名单导入
hs300 = pd.read_excel(r"../data/沪深300指数成分股历年调整名单+更新至2022年1月1日.xlsx",dtype = object)
hs300_used = hs300.copy()

# 假设hs300_used是您的DataFrame，包含沪深300名单的信息
# 创建一个日期范围，包括从2005年4月开始到2021年12月结束的所有月份
date_range = pd.date_range(start='2005-04-01', end='2021-12-31', freq='M')
# 将日期范围转换为周期数据类型
months = date_range.to_period('M')
# 创建一个包含月份、股票代码和股票名称的DataFrame
monthly_stock_data = pd.DataFrame(columns=['month_details', 'stock', 'stock_name'])
# 遍历每个月份
for month in months:
    # 使用条件筛选操作选择满足条件的数据
    filtered_stock_list = hs300_used[(hs300_used['开始日期_BegDt'].dt.to_period('M') <= month) &
                                     ((hs300_used['结束日期_EndDt'].dt.to_period('M') > month) | (hs300_used['结束日期_EndDt'].isnull()))]
    # 将开始日期和结束日期转换为月份
    filtered_stock_list['开始日期_BegDt'] = filtered_stock_list['开始日期_BegDt'].dt.to_period('M')
    filtered_stock_list['结束日期_EndDt'] = filtered_stock_list['结束日期_EndDt'].dt.to_period('M')
    # 将成分股代码和成分股名称转换为字符串类型
    filtered_stock_list['成分股代码_CompoStkCd'] = filtered_stock_list['成分股代码_CompoStkCd'].astype(str)
    filtered_stock_list['成分股名称_CompoStkNm'] = filtered_stock_list['成分股名称_CompoStkNm'].astype(str)
    # 遍历每个月份的筛选结果
    for index, row in filtered_stock_list.iterrows():
        start_date = row['开始日期_BegDt']
        end_date = row['结束日期_EndDt']
        codes = row['成分股代码_CompoStkCd'].split(', ')
        stocks = row['成分股名称_CompoStkNm'].split(', ')
        data = {'month_details': [month] * len(codes), 'stock': codes, 'stock_name': stocks}
        monthly_stock_data = pd.concat([monthly_stock_data, pd.DataFrame(data)])

# 月份赋值
month_name=monthly_stock_data['month_details'].unique()
month_number=range(85,286)
# 创建包含每个月份和相应数字的DataFrame
month_name_number = pd.DataFrame({'month_details': month_name, 'month':month_number})
# 合并dataframe
hc300_monthly = pd.merge(monthly_stock_data, month_name_number, on='month_details', how='left')
hc300_monthly.to_csv('hs300_monthly.csv')
df=hc300_monthly.copy()
# 按照"month"进行分组，并将"stock"和"name"合并为列表
grouped_df = df.groupby("month").agg({"stock": list, "stock_name": list}).reset_index()
from ast import literal_eval
grouped_df ['stock'] = grouped_df ['stock'].apply(literal_eval)
grouped_df ['stock_name'] = grouped_df ['stock_name'].apply(literal_eval)
grouped_df['stock'] = grouped_df['stock'].apply(lambda x: list(set(x)))
grouped_df['stock_name'] = grouped_df['stock_name'].apply(lambda x: list(set(x)))
# 打印整合后的结果
grouped_df.to_csv('hs300.csv')
grouped_df
