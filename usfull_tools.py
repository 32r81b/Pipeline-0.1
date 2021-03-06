import numpy as np
import pandas as pd
import gc


def load_DS(debug_mode, KAGGLE_DIR, KAGGLE_PREFIX, LOCAL_PREFIX='.csv'):
    if debug_mode == True:
        train = pd.read_csv(KAGGLE_DIR + 'train' + KAGGLE_PREFIX + LOCAL_PREFIX, nrows=10000)
        test = pd.read_csv(KAGGLE_DIR + 'test' + KAGGLE_PREFIX + LOCAL_PREFIX, nrows=10000)
    else:
        train = pd.read_csv(KAGGLE_DIR + 'train' + KAGGLE_PREFIX + LOCAL_PREFIX)
        test = pd.read_csv(KAGGLE_DIR + 'test' + KAGGLE_PREFIX + LOCAL_PREFIX)

    return  reduce_mem_usage(train), reduce_mem_usage(test)


def numeric_types():
    return ['int64','int32','int16','int8','float64','float32','float16']

def numeric_fraction_types():
    return ['float64','float32','float16']

def numeric_nonfraction_types():
    return ['int64','int32','int16','int8']




def reduce_mem_usage(df):
    gc.collect()
    start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    return df