#%% 包区域
import pandas as pd

#%% 常用数据读取函数
def null_transform(df):
    print('== 检查空值')
    temp = df.isnull().sum(axis=1)
    print(f'共有{temp[temp>0].count()}行含有空值')
    df.dropna(inplace=True)
    return df

def dup_transfor(df, key_col):
    print('== 检查重复数据')
    print(f'共有{df.duplicated(subset=key_col).sum()}行含有重复值')
    df.drop_duplicates(subset=key_col, inplace=True)
    return df

def extract(path, key_col = None):
    df = pd.read_csv(path)
    df = null_transform(df)
    df = dup_transfor(df, key_col=key_col)
    print(f'== 数据读取与初步清洗完成\n大小为{df.shape}\n样表为{df.head()}')
    return df

#%% 测试用
if __name__ == '__main__':
    df = extract(r'D:\Personal\Credit-Risk\Statement\Vintage\repay_actual_data_new.csv')
# %%
