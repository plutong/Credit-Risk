# %%%%%%%%%%%%%%%%%%%%加载需要的包%%%%%%%%%%%%%%%%%%%%
import pandas as pd
pd.options.display.max_columns = 50
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

# %%%%%%%%%%%%%%%%%%%%设置需要参数%%%%%%%%%%%%%%%%%%%%
snapdate = datetime.datetime(2022, 12, 14)
path_actual = 'repay_actual_data_new.csv'
path_plan = 'repay_plan_data_new.csv'
col_actual_1 = ['order_id', 'order_sub_id', 'actual_repay_date', 'actual_repay_prin']
col_plan_1 = ['order_id', 'order_sub_id', 'phase','expected_date', 'create_time', 'loan_amt', 'irr_rate', 'schedule_repay_prin']

# %%%%%%%%%%%%%%%%%%%%读取数据并做初步探索%%%%%%%%%%%%%%%%%%%%
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
df_actual.sort_values(['order_id', 'order_sub_id'], ascending=[True, True], inplace=True)
df_actual['actual_repay_date'] = pd.to_datetime(df_actual['actual_repay_date'].astype('str'))
df_plan.sort_values(['order_id', 'order_sub_id'], ascending=[True, True], inplace=True)
df_plan['expected_date'] = pd.to_datetime(df_plan['expected_date'].astype('str'))
df_plan['create_time'] = pd.to_datetime(df_plan['create_time'])
df_plan['loan_month'] = df_plan['create_time'].apply(lambda x: datetime.datetime(x.year, x.month, 1))
df_plan['observe_date'] = df_plan['expected_date'].apply(lambda x: datetime.datetime(x.year, x.month, int(x.days_in_month)))

# %%%%%%%%%%%%%%%%%%%%数据检查%%%%%%%%%%%%%%%%%%%%
print(f'预期还款表每列的空置为\n{df_plan.isna().sum()}')
print(f'预期还款表子单重复数为{df_plan.duplicated(subset='order_sub_id').sum()}')
print(f'预期还款表的异常单数为{len(df_plan[(df_plan['irr_rate']<=0) | (df_plan['schedule_repay_prin']<=0)])}')

print(f'实际还款表每列的空置为\n{df_actual.isna().sum()}')
print(f'实际还款表子单重复数为{df_actual.duplicated(subset='order_sub_id').sum()}')
print(f'实际还款表的异常单数为{len(df_actual[(df_actual['actual_repay_prin']<=0) ])}')

# %%%%%%%%%%%%%%%%%%%%合并预期还款表以及实际还款表%%%%%%%%%%%%%%%%%%%%
df = pd.merge(df_plan, df_actual, how='left', on=['order_id', 'order_sub_id'], indicator=True)
# 选取应还款日在快照日期之前的单
df.sort_values(['order_id', 'phase'], ascending=[True, True])
df['actual_repay_date_pre'] = df.groupby('order_id')['actual_repay_date'].shift(1)
df = df[df['expected_date']<=snapdate]
print(f'两表的合并情况如下\n{df['_merge'].value_counts()}')

# %%%%%%%%%%%%%%%%%%%%统计每单每期逾期标识与金额%%%%%%%%%%%%%%%%%%%%
# 统计本期以及上期未还本金
df['paid_amt'] = df.groupby('order_id')['actual_repay_prin'].cumsum()
df['balance'] = df['loan_amt'] - df['paid_amt']
df['balance_pre'] = df.groupby('order_id')['balance'].shift(1)
# 根据归还行为以及日期分类
# 归还标签: 有归还行为且在月末观察日之前归还/ 未归还: 其他
df['observe_flag'] = 0
df.loc[(df['actual_repay_date'].isna()) | 
       (df['actual_repay_date'] > df['observe_date']), 
       'observe_flag'] = 1
df['observe_flag_pre'] = df.groupby('order_id')['observe_flag'].shift(1)
# 归还类型
# 1. 良好归还
df['observe_type'] = 0
# 2.初犯: 本期逾期，同时上期归还或者归还在本期月末观察日之前
df.loc[(df['observe_flag']==1) & 
       (df['observe_flag_pre']==0),
       'observe_type'] = 1
df.loc[(df['observe_flag']==1) & 
       (df['observe_flag_pre']==1) &
       (df['actual_repay_date_pre']<= df['observe_date']),
       'observe_type'] = 1
# 3. 再犯: 本期逾期，上期未实际还款或者实际还款日在本月的观察日之后
df.loc[(df['observe_flag']==1) & 
       (df['observe_flag_pre']==1) &
       (df['actual_repay_date_pre']).isna(),
       'observe_type'] = 2
df.loc[(df['observe_flag']==1) & 
       (df['observe_flag_pre']==1) &
       (df['actual_repay_date_pre'] > df['observe_date']),
       'observe_type'] = 2

# %%%%%%%%%%%%%%%%%%%%根据逾期标识加工逾期天数和金额%%%%%%%%%%%%%%%%%%%%
# 逾期天数
# 良好归还账单逾期0天,如果第一期就未还款，前列判断首次逾期的逻辑失效，需要手动判断并设0
df.loc[(df['actual_repay_date'].notna()) & 
       (df['actual_repay_date'] <= df['observe_date']), 'overdue_days'] = 0
# 初次逾期账单逾期天数为月末观察日与预期归还日期差
df.loc[df['observe_type']==1, 'overdue_days'] = (df['observe_date'] - df['expected_date']).dt.days
# 再次逾期账单的实际应还款日应往前追溯到首次逾期单
df['expected_date_origin'] = df['expected_date']
df.loc[df['observe_type']==2, 'expected_date_origin'] = np.nan
df['expected_date_origin'] = df.groupby('order_id')['expected_date_origin'].transform(lambda x:x.ffill())
df.loc[df['observe_type']==2, 'overdue_days'] = (df['observe_date'] - df['expected_date_origin']).dt.days
# 更新剩余本金
df.loc[df['observe_type']==1,'balance'] = df['balance_pre']
df.loc[(df['observe_type']==2),'balance'] = np.nan
df['balance'] = df.groupby('order_id')['balance'].transform(lambda x:x.ffill())

# %%%%%%%%%%%%%%%%%%%%根据逾期天数加工逾期期数%%%%%%%%%%%%%%%%%%%%
df.loc[df['overdue_days']<1, 'overdue_status'] = 'M0'
df.loc[(1<=df['overdue_days'])&(df['overdue_days']<=30), 'overdue_status'] = 'M1'
df.loc[(30<df['overdue_days'])&(df['overdue_days']<=60), 'overdue_status'] = 'M2'
df.loc[(60<df['overdue_days'])&(df['overdue_days']<=90), 'overdue_status'] = 'M3'
df.loc[(90<df['overdue_days'])&(df['overdue_days']<=120), 'overdue_status'] = 'M4'
df.loc[(120<df['overdue_days'])&(df['overdue_days']<=150), 'overdue_status'] = 'M5'
df.loc[(150<df['overdue_days'])&(df['overdue_days']<=180), 'overdue_status'] = 'M6'
df.loc[(180<df['overdue_days'])&(df['overdue_days']<=210), 'overdue_status'] = 'M7'
df.loc[(210<df['overdue_days'])&(df['overdue_days']<=240), 'overdue_status'] = 'M8'
df.loc[(240<df['overdue_days']), 'overdue_status'] = 'M9+'
print(f'现阶段我们观察到各类型的逾期期数如下\n{df['overdue_status'].value_counts()}')

# %%%%%%%%%%%%%%%%%%%%生成迁移率表格%%%%%%%%%%%%%%%%%%%%
df['observe_month'] = df['observe_date'].apply(lambda x: datetime.datetime(x.year, x.month, 1))
df_migration = pd.pivot_table(df, index='overdue_status', columns='observe_month', values='balance', aggfunc='sum').sort_index()
df_migration = df_migration.shift(-1,axis=1).shift(-1,axis=0)/df_migration
df_migration = df_migration.iloc[:-1, 1:-1]
df_migration.index = ['MO->M1','M1->M2','M2->M3','M3->M4','M4->M5','M5->M6','M6->M7','M7->M8','M8->M9+']

# %%%%%%%%%%%%%%%%%%%%可视乎迁移率表格%%%%%%%%%%%%%%%%%%%%
# df_migration.columns = df_migration.columns.strftime('%Y%m')
plt.figure(figsize=(12,9))
plt.title('Migration Rate Statement')
sns.heatmap(data=df_migration.iloc[:-1,:-1],annot = True,fmt = '.2%',cmap='YlOrRd')
plt.savefig('Migration Rate Statement.png')
plt.show()
