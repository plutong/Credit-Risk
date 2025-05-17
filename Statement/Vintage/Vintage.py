#%% Package Area
import pandas as pd
import datetime as dt
import sys
sys.path.append(r"D:\Personal\Credit-Risk")
from common import extract

#%% Vintage Workflow
class vintage:
    # Extract data and do primary transformation 
    def __init__(self, 
                 debit_path,
                 plan_path,
                 snap_date):
        self.debit_df = extract(debit_path)
        self.plan_df = extract(plan_path)

        # 由于两类表都是增量数据，选择时间节点提取数据切片
        self.debit_df = self.debit_df[self.debit_df['snapshot_date'] == snap_date]
        self.plan_df = self.plan_df[self.plan_df['snapshot_date'] == snap_date]
        print(f'当前时间节点{snap_date}下的数据量分别为{self.debit_df.shape}和{self.plan_df.shape}')

if __name__ == '__main__':
    #%% 设置参数
    debit_path = r'D:\Personal\Credit-Risk\Statement\Vintage\repay_actual_data_new.csv'
    plan_path=r'D:\Personal\Credit-Risk\Statement\Vintage\repay_plan_data_new.csv'
    snap_date=20221214
    snap_col = 'snapshot_date'
    now = dt.date(2022,12,21)

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
    # %% 对每单进行统计
    df['actual_repay_prin_cumsum'] = df['actual_repay_prin'].cumsum()
    df['balance'] = df['loan_amt'] - df['actual_repay_prin_cumsum']
    df['balance_pre'] = df.groupby('order_id')['balance'].transform(lambda x:x.shift(1))
    # %% 加工逾期标识
