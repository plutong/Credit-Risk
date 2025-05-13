#%% Package Area
import pandas as pd

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


#%% Test
if __name__ == '__main__':
    #
    state = vintage(debit_path=r'D:\Personal\Credit-Risk\Statement\Vintage\repay_actual_data_new.csv',
            plan_path=r'D:\Personal\Credit-Risk\Statement\Vintage\repay_plan_data_new.csv',
            snap_date=20221214)
# %%
