#%% Package Area
import pandas as pd
pd.set_option('display.max_columns', None)
import datetime as dt
import numpy as np
import sys
sys.path.append(r"D:\Personal\Credit-Risk")
from common import extract
import seaborn as sns
import matplotlib.pyplot as plt

#%% 设置参数
debit_path = r'D:\Personal\Credit-Risk\Statement\Vintage\repay_actual_data_new.csv'
plan_path=r'D:\Personal\Credit-Risk\Statement\Vintage\repay_plan_data_new.csv'
snap_date=20221214
snap_col = 'snapshot_date'
now = dt.date(2022,12,31)

#%% 取数
debit_df = extract(debit_path)
plan_df = extract(plan_path)

#%% 在增量数据中取快照数据
debit_time_df = debit_df[debit_df[snap_col]==snap_date]
plan_time_df = plan_df[plan_df[snap_col]==snap_date]

# %% 链接两表得到各期还款数据
df = pd.merge(debit_time_df[['order_id', 'order_sub_id', 'actual_repay_date', 'actual_repay_amt', 'actual_repay_prin', 'actual_repay_int']],
                plan_time_df, 
                on=['order_id', 'order_sub_id'],
                how='right')
col_sel = ['user_id','order_id','phase_total','phase','loan_amt','expected_date',
        'actual_repay_date','actual_repay_amt','obser_month_end',
        'balance','balance_pre','odu_flag','odu_flag_pre','actual_repay_date_pre',
        'odu_type']
# 筛选预期还款日在快照日期之前的数据
df = df[df['expected_date'] <= snap_date]

# %% 加工时间数据
for col in ['actual_repay_date', 'expected_date']:
    df[col] = pd.to_datetime(df[col].astype(int, errors='ignore'), format='%Y%m%d').dt.date
df['create_time'] = pd.to_datetime(df['create_time']).dt.date
# 观测应还日期在观测点之前的数据
df = df[df['expected_date'] <= now]
# 得到放款月
df['create_month'] = pd.to_datetime(df['create_time']).dt.strftime('%Y%m')
# 在每个应还款月月末进行观察
df['observe_date'] = df.loc[df['expected_date'].notna(), 'expected_date'].apply(lambda x: dt.date(x.year, x.month, pd.to_datetime(x).days_in_month))
df.sort_values(['order_id', 'phase'], ascending=[True, True], inplace=True)

# %% 对每单进行金额统计
df['actual_repay_prin_cumsum'] = df.groupby('order_id')['actual_repay_prin'].transform('cumsum')
df['balance'] = df['loan_amt'] - df['actual_repay_prin_cumsum']
df['balance_pre'] = df.groupby('order_id')['balance'].transform(lambda x:x.shift(1))
# %% 加工每月逾期标识
# 未逾期月份标为0
df['overdue_flag'] = 0
# 加工逾期标签: 实际还款日位于月末观察日之后
df.loc[(df['actual_repay_date']>df['observe_date']) | 
        (df['actual_repay_date'].isna()), 'overdue_flag'] = 1
# %% 迁移上期数据
df['overdue_flag_pre'] = df.groupby('order_id')['overdue_flag'].transform(lambda x:x.shift(1))
df['actual_repay_date_pre'] = df.groupby('order_id')['actual_repay_date'].transform(lambda x:x.shift(1))
# %% 进一步加工每月逾期标识
# 1. 有良好还款记录的还款单
df['overdue_type'] = 0
# 2. 第一期便逾期的单，逾期金额记为原借款金额
df.loc[(df['overdue_flag_pre'].isna())&
        (df['overdue_flag']==1), 'overdue_type'] = 1
df.loc[(df['overdue_flag_pre'].isna())&
        (df['overdue_flag']==1), 'balance'] = df['loan_amt']
# 3. 前一期未逾期但是本期逾期
df.loc[(df['overdue_flag_pre']==0)&
        (df['overdue_flag']==1), 'overdue_type'] = 2
df.loc[(df['overdue_flag_pre']==0)&
        (df['overdue_flag']==1), 'balance'] = df['balance_pre']
# 4. 上一期逾期，本期逾期，但是上期还款日在观察日之后
df.loc[(df['overdue_flag_pre']==1)&
        (df['overdue_flag']==1) & 
        (df['actual_repay_date_pre'] <= df['observe_date']), 'overdue_type'] = 3
df.loc[(df['overdue_flag_pre']==0)&
        (df['overdue_flag']==1) & 
        (df['actual_repay_date_pre'] <= df['observe_date']), 'balance'] = df['balance_pre']
# 5. 上一期逾期，本期继续逾期且没有实际还款
df.loc[(df['overdue_flag_pre']==1)&
        (df['overdue_flag']==1) &
        (df['actual_repay_date'].isna()),'overdue_type'] = 4
df.loc[(df['overdue_flag_pre']==1)&
        (df['overdue_flag']==1) &
        (df['actual_repay_date_pre'] > df['observe_date']),'overdue_type'] = 4
df.loc[(df['overdue_type']==4),'balance'] = np.nan
# 6. 对每笔借款单的拖欠金额进行向下补齐
df['balance'] = df.groupby('order_id')['balance'].transform(lambda x:x.ffill())
# %%加工逾期金额
# 1. 加工逾期日期
df.loc[(df['actual_repay_date'].notna())&(df['actual_repay_date']<=df['observe_date']),'overdue_days'] = 0
df.loc[(df['overdue_type'].isin([1,2,3])),'overdue_days'] = (pd.to_datetime(df['observe_date'])-pd.to_datetime(df['expected_date'])).dt.days
df['expected_date_2'] = df['expected_date']
df.loc[(df['overdue_type'].isin([4])),'expected_date_2'] = np.nan
df['expected_date_2'] = df.groupby('order_id')['expected_date_2'].transform(lambda x:x.ffill())
df.loc[(df['overdue_type'].isin([4])),'overdue_days'] = (pd.to_datetime(df['observe_date']) - pd.to_datetime(df['expected_date_2'])).dt.days

# 如果定义逾期超过30为 “坏客户”，则：
cond = df['overdue_days']>30
df['overdue_amt'] = df['balance'].where(cond, other=0)
col_sel.extend(['odu_days','expected_date_2','overdue_amt'])

# %% 生成Vintage报表
# 截取今天之前的数据
df = df[df['observe_date'] <= now]
df_new = df.drop_duplicates(subset=['order_id'])
df_mob = pd.pivot_table(df, index='create_month', columns='phase', values='overdue_amt', aggfunc='sum')
# 放款月的放款本金求和
df_loan = df_new.groupby(['create_month'])['loan_amt'].sum()
# 拼接计算金额dpd30+逾期率
df_mob_all = pd.concat([df_loan.to_frame(name='loan_amt'),df_mob],axis=1)
df_mob_rate = df_mob_all.loc[:,1:].divide(df_mob_all['loan_amt'],axis=0)
# %% 可视化Vintage报表
plt.figure(figsize=(15, 8))
plt.title('cohort vintage DPD30+')
sns.heatmap(data=df_mob_rate, annot = True, fmt = '.4%', vmin = 0.0, vmax = 0.005,cmap="YlGnBu")
plt.savefig('heatmap.png')
plt.show()

# %% 可视化Vintange坏账折现
palette = sns.color_palette("husl", 9)
plt.figure(figsize=(15, 8))
plt.title('lineplot vintage DPD30+')
plt.ylim(0,0.04)
plt.xticks(df_mob_rate.loc[:,1:].columns.tolist())
sns.lineplot(data=df_mob_rate.loc[:,1:].T, markers=True, dashes=True, lw=3, palette=palette)
plt.savefig('lineplot.png')
plt.show()
# %%
