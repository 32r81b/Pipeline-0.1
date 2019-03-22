import pandas as pd
import numpy as np

from usfull_tools import numeric_types
numeric_types = numeric_types()

#generate new features by sum and multiply all numerical columns
def DS_numerical_feature_generator(train, test, target_column):   
    DS = train.append(test, sort = False)
    
    column_types = DS.dtypes
    column_types = column_types[column_types != 'object']
    numeric_columns = column_types.index.tolist()
    numeric_columns.remove(target_column)
    print('Numerical columns: ', len(numeric_columns))

    for i in range(len(numeric_columns)):
        for m in range(i+1, len(numeric_columns)):
            new_column_name = str('sum_' + numeric_columns[i] + '_with_' + numeric_columns[m])
            DS[new_column_name] = DS[numeric_columns[i]] + DS[numeric_columns[m]]
            new_column_name = str('mult_' + numeric_columns[i] + '_with_' + numeric_columns[m])
            DS[new_column_name] = DS[numeric_columns[i]] * DS[numeric_columns[m]]
        
        
    train = DS[DS[target_column].notnull()]
    test = DS[DS[target_column].isnull()]
    
    return train, test.drop(target_column, axis=1)

#generate dummies from categorical columns
def DS_dummies(train, test, target_column):   
    DS = train.append(test, sort = False)
    rows, cells = DS.shape
    
    for column in DS.columns.drop(target_column):
        if DS[column].dtype == object and DS[column].nunique() < 20:
            dummies = pd.get_dummies(DS[column], prefix = str(column + '_dummie'))
            DS = pd.concat([DS, dummies], axis=1)
            print('Create dummies for', column, ':', len(dummies.columns))
            
    train = DS[DS[target_column].notnull()]
    test = DS[DS[target_column].isnull()]
    
    return train, test.drop(target_column, axis=1)