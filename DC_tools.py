import pandas as pd
import numpy as np

from usfull_tools import numeric_types
numeric_types = numeric_types()

def DS_clean(train, test, target_column):   
    DS = train.append(test, sort = False)
    rows, cells = DS.shape
    
    #Search Nan in all columns except target
    for column in DS.columns.drop(target_column):
        #remove ID column
        if DS[column].nunique() == rows:
            print('drop as ID column: ', column)
            DS.drop(column, axis=1, inplace = True)
        #search columns with Nan
        elif DS[column].isna().sum() > 0:
            DS[column + '_nan'] = DS[column]
            
            #try to fill NA if NA < 50% rows, median for numeric, most frequent for object            
            if DS[column].isna().sum()/rows < 0.5:
                if DS[column].dtype in numeric_types:
                    print(column, ': NAN ', np.round(DS[column].isna().sum()/rows*100), '%', ' replaced ', DS[column].median())
                    DS[column + '_median'] = DS[column].fillna(DS[column].median())
                if DS[column].dtype == object:
                    print(column, ': NAN ', np.round(DS[column].isna().sum()/rows*100), '%', ' replaced ', DS[column].value_counts().idxmax())
                    DS[column + '_idxmax'] = DS[column].fillna(DS[column].value_counts().idxmax())

            #if NA > 50% replace isnull
            else:
                DS[column + '_isnull'] = DS[column].isnull()             

            DS.drop(column, axis=1, inplace = True)
    
    train = DS[DS[target_column].notnull()]
    test = DS[DS[target_column].isnull()]
    
    return train, test.drop(target_column, axis=1), DS