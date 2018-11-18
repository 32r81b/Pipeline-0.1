import pandas as pd
import numpy as np

from usfull_tools import numeric_types
numeric_types = numeric_types()

# 1. Remove global ID column
# 2. Search columns with Nan
# 2.1 if NA < 50% rows:
#     - median for numeric
#     - most frequent for object  
# 2.2 if NA > 50% replace as _isnull column
def DS_reaplce_nan(train, test, target_column):   
    DS = train.append(test, sort = False)
    rows, cells = DS.shape
    
    #Search Nan in all columns except target
    for column in DS.columns.drop(target_column):
        # 1. Remove global ID column
        if DS[column].nunique() == rows:
            print('Drop as ID column: ', column)
            DS.drop(column, axis=1, inplace = True)
        #2. Search columns with Nan
        elif DS[column].isna().sum() > 0:
            DS[column + '_nan'] = DS[column].fillna('NAN')
            
            # 2.1 if NA < 50% rows, median for numeric, most frequent for object            
            if DS[column].isna().sum()/rows < 0.5:
                if DS[column].dtype in numeric_types:
                    print(column, np.round(DS[column].isna().sum()/rows*100), '% NAN replaced by:')
                    print ('median = ', DS[column].median())
                    DS[column + '_nan_median'] = DS[column].fillna(DS[column].median())
                    print ('min = ', DS[column].min())
                    DS[column + '_nan_min'] = DS[column].fillna(DS[column].min())
                    print ('max = ', DS[column].max())
                    DS[column + '_nan_max'] = DS[column].fillna(DS[column].max())
                    print ('by zero')
                    DS[column + '_nan_0'] = DS[column].fillna(0)
                if DS[column].dtype == object:
                    print(column, ': NAN ', np.round(DS[column].isna().sum()/rows*100), '%', ' replaced ', DS[column].value_counts().idxmax())
                    DS[column + '_idxmax'] = DS[column].fillna(DS[column].value_counts().idxmax())

            # 2.2 if NA > 50% replace as _isnull column
            else:
                DS[column + '_isnull'] = DS[column].isnull()             

            DS.drop(column, axis=1, inplace = True)
    
    train = DS[DS[target_column].notnull()]
    test = DS[DS[target_column].isnull()]
    
    return train, test.drop(target_column, axis=1), DS



def DS_reaplce_idcolumns(train, test, target_column):
    DS = train.append(test, sort = False)
    rows, cells = DS.shape
    
    #!todo: remove group id after group feature generator
    for column in DS.columns.drop(target_column):
        print(column)

    
    train = DS[DS[target_column].notnull()]
    test = DS[DS[target_column].isnull()]
    
    return train, test.drop(target_column, axis=1), DS