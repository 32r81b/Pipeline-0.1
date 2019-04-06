import pandas as pd
import numpy as np

from usfull_tools import numeric_types
numeric_types = numeric_types()
from set_vars import acc_sign, skew_trashold, feature_generator_trashold
from FastScore_tools import fast_score

def DS_skew(train, test, target_column):   
    DS = train.append(test, sort = False)
    rows, cells = DS.shape
    
    skew = DS.skew()
    skew = skew[abs(skew)>skew_trashold]
    print('Create log feature for skew features:\n', skew)

    for row in skew.iteritems():
        new_name = 'log_' + row[0]
        DS[new_name] = np.log(DS[row[0]]+1)
    
    train = DS[DS[target_column].notnull()]
    test = DS[DS[target_column].isnull()]
    
    return train, test.drop(target_column, axis=1)


#generate new features by sum and multiply all numerical columns
def DS_numerical_feature_generator(train, test, target_column):   
    DS = train.append(test, sort = False)
    
    column_types = DS.dtypes
    column_types = column_types[column_types != 'object']
    numeric_columns = column_types.index.tolist()
    numeric_columns.remove(target_column)
    print('Numerical columns: ', len(numeric_columns))

    for column in train.columns:
        if train[column].dtype == object:
            if train[column].nunique() < 20:
                dummies = pd.get_dummies(train[column], prefix = str(column + '_dummie'))
                train = pd.concat([train, dummies], axis=1)
            train.drop(column, axis=1, inplace=True)
    
    basline_acc, time = fast_score(train.copy())
    print('Basline fast_score:', np.round(basline_acc, 4), np.round(time), 'sec. taken')
    
    for i in range(len(numeric_columns)):
        for m in range(i+1, len(numeric_columns)):

            #sumary
            new_column_values = train[numeric_columns[i]] + train[numeric_columns[m]]
            if np.any(np.isnan(new_column_values)) == False and np.all(np.isfinite(new_column_values)) == True:
                    acc, time = fast_score(train.copy(), new_column_values)
                    if (basline_acc*acc_sign>acc*acc_sign*feature_generator_trashold):
                        print('fast_score sum', numeric_columns[i], numeric_columns[m], ':', np.round(acc, 4), np.round(time), 'sec. taken')
                        new_column_name = str('sum_' + numeric_columns[i] + '_' + numeric_columns[m])
                        DS[new_column_name] = DS[numeric_columns[i]] + DS[numeric_columns[m]]

            #subtraction
            new_column_values = train[numeric_columns[i]] - train[numeric_columns[m]]
            if np.any(np.isnan(new_column_values)) == False and np.all(np.isfinite(new_column_values)) == True:
                    acc, time = fast_score(train.copy(), new_column_values)
                    if (basline_acc*acc_sign>acc*acc_sign*feature_generator_trashold):
                        print('fast_score subtraction', numeric_columns[i], numeric_columns[m], ':', np.round(acc, 4), np.round(time), 
                              'sec. taken')
                        new_column_name = str('sub_' + numeric_columns[i] + '_' + numeric_columns[m])
                        DS[new_column_name] = DS[numeric_columns[i]] - DS[numeric_columns[m]]
                        
            #multiply
            new_column_values = train[numeric_columns[i]] * train[numeric_columns[m]]
            if np.any(np.isnan(new_column_values)) == False and np.all(np.isfinite(new_column_values)) == True:
                acc, time = fast_score(train.copy(), new_column_values)
                if (basline_acc*acc_sign>acc*acc_sign*feature_generator_trashold):
                    print('fast_score multiply', numeric_columns[i], numeric_columns[m], ':', np.round(acc, 4), np.round(time), 
                          'sec. taken')
                    new_column_name = str('multiply_' + numeric_columns[i] + '_' + numeric_columns[m])
                    DS[new_column_name] = DS[numeric_columns[i]] * DS[numeric_columns[m]]              
                        
            #division
            new_column_values = train[numeric_columns[i]] / train[numeric_columns[m]]
#             print('np.any(np.isnan(new_column_values)) np.all(np.isfinite(new_column_values))', 
#                   np.any(np.isnan(new_column_values)), np.all(np.isfinite(new_column_values)))
            
            if np.any(np.isnan(new_column_values)) == False and np.all(np.isfinite(new_column_values)) == True:
                acc, time = fast_score(train.copy(), new_column_values)
                if (basline_acc*acc_sign>acc*acc_sign*feature_generator_trashold):
                    print('fast_score division', numeric_columns[i], numeric_columns[m], ':', np.round(acc, 4), np.round(time), 
                          'sec. taken')
                    new_column_name = str('division_' + numeric_columns[i] + '_' + numeric_columns[m])
                    DS[new_column_name] = DS[numeric_columns[i]] / DS[numeric_columns[m]]         

        
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