# %% 加载需要的包
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

# %% 设置需要参数
snapdate = datetime.datetime(2022, 12, 14)
path_actual = 'repay_actual_data_new.csv'
path_plan = 'repay_plan_data_new.csv'
col_actual_1 = ['order_id', 'order_sub_id', 'phase', 'actual_repay_date', 'actual_repay_prin']
col_plan_1 = ['order_id', 'order_sub_id', 'phase','expected_date', 'create_time', 'loan_amt', 'irr_rate', 'schedule_repay_prin']

# %% 读取数据并做初步探索
# 提取实际还款表并初步识别
df_actual = pd.read_csv(path_actual)
df_actual['snapshot_date'] = pd.to_datetime(df_actual['snapshot_date'].astype('str'))
df_actual = df_actual[df_actual['snapshot_date']==snapdate]
print(f'Actual repay tables has shape {df_actual.shape} with sample table\n{df_actual.head()}')
print(f'Actual repay table has {df_actual['order_id'].nunique()} unique orders')
print(f'The phase for each orders have following hist picture')
sns.histplot(df_actual.groupby('order_id')['phase'].count(), bins=4)
plt.title("Distribution of actual repayment periods")
plt.xlabel("Phase")
plt.ylabel("Count")
plt.savefig('Distribution of actual repayment periods.png')
plt.show()

# 提取预期还款表并初步识别
df_plan = pd.read_csv(path_plan)
df_plan['snapshot_date'] = pd.to_datetime(df_plan['snapshot_date'].astype('str'))
df_plan = df_plan[df_plan['snapshot_date']==snapdate]
print(f'plan repay tables has shape {df_plan.shape} with sample table\n{df_plan.head()}')
print(f'plan repay table has {df_plan['user_id'].nunique()} unique users with {df_plan['order_id'].nunique()} unique orders')
print(f'The phase for each orders have following hist picture')
sns.histplot(df_plan.groupby('order_id')['phase'].count(), bins=4)
plt.title("Distribution of plan repayment periods")
plt.xlabel("Phase")
plt.ylabel("Count")
plt.savefig('Distribution of plan repayment periods.png')
plt.show()

# 选取有用特征进行提取，对列进行初步处理并生成相关信息
df_actual = df_actual[col_actual_1]
df_plan = df_plan[col_plan_1]
df_actual['actual_repay_date'] = pd.to_datetime(df_actual['actual_repay_date'].astype('str'))
df_plan['expected_date'] = pd.to_datetime(df_plan['expected_date'].astype('str'))
df_plan['create_time'] = pd.to_datetime(df_plan['create_time'])
df_plan['loan_month'] = df_plan['create_time'].apply(lambda x: datetime.datetime(x.year, x.month, 1))
df_plan['observe_date'] = df_plan['expected_date'].apply(lambda x: datetime.datetime(x.year, x.month, int(x.days_in_month)))

# %%数据检查
print(f'预期还款表每列的空置为\n{df_plan.isna().sum()}')
print(f'预期还款表子单重复数为{df_plan.duplicated(subset='order_sub_id').sum()}')
print(f'预期还款表的异常单数为{len(df_plan[(df_plan['irr_rate']<=0) | (df_plan['schedule_repay_prin']<=0)])}')

print(f'实际还款表每列的空置为\n{df_actual.isna().sum()}')
print(f'实际还款表子单重复数为{df_actual.duplicated(subset='order_sub_id').sum()}')
print(f'实际还款表的异常单数为{len(df_actual[(df_actual['actual_repay_prin']<=0) ])}')

# %%
