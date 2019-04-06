import pandas as pd
import numpy as np
import math
import scipy.stats as st

from usfull_tools import numeric_types, numeric_fraction_types, numeric_nonfraction_types
from FastScore_tools import fast_score
numeric_types = numeric_types()
numeric_nonfraction_types = numeric_nonfraction_types()
numeric_fraction_types = numeric_fraction_types()

from set_vars import KAGGLE_PREFIX, KAGGLE_DIR, acc_sign

def DS_reaplce_nan(train, test, target_column):
    DS = train.append(test, sort = False)

    import csv
    with open(KAGGLE_DIR + KAGGLE_PREFIX + 'group_feature.csv') as data:
        reader = csv.reader(data)
        group_feature = list(reader)[0]

    rows, cells = DS.shape
    
    for column in train.columns:
        if train[column].dtype == object:
            if train[column].nunique() < 20:
                dummies = pd.get_dummies(train[column], prefix = str(column + '_dummie'))
                train = pd.concat([train, dummies], axis=1)
            train.drop(column, axis=1, inplace=True)
        
    #Search Nan in all columns except target
    for column in DS.columns.drop(target_column):
        # 1. Remove global ID column
        if DS[column].nunique() == rows:
            print('\n\n--- Drop as ID column: ', column)
            DS.drop(column, axis=1, inplace = True)
        #2. Search columns with Nan
        elif DS[column].isna().sum() > 0:
            # 2.1 if NA < 50% rows
            if DS[column].isna().sum()/rows < 0.5:
                #median/min/max/0 for fraction numeric            
                if DS[column].dtype in numeric_fraction_types:
                    print('\n\n---Searching best NAN replace for numeric_fraction:', column)
                   
                    nan_value = DS[column].median()
                    new_column_values = train[column].fillna(nan_value)
                    acc, time = fast_score(train.copy(), new_column_values, column)
                    best_column_values = new_column_values
                    best_acc = acc
                    print('fast_score with median:', np.round(acc, 4), np.round(time), 'sec. taken')
                   
                    nan_value = DS[column].min()
                    new_column_values = train[column].fillna(nan_value)
                    acc, time = fast_score(train.copy(), new_column_values, column)
                    
                    if (best_acc*acc_sign>acc*acc_sign):
                        best_acc=acc
                        best_column_values=new_column_values
                        print('fast_score with min   :', np.round(acc, 4), np.round(time), 'sec. taken')
                        
                    nan_value = DS[column].max()
                    new_column_values = train[column].fillna(nan_value)
                    acc, time = fast_score(train.copy(), new_column_values, column)
                    
                    if (best_acc*acc_sign>acc*acc_sign):
                        best_acc=acc
                        best_column_values=new_column_values
                        print('fast_score with max   :', np.round(acc, 4), np.round(time), 'sec. taken')
                   
                    nan_value = 0
                    new_column_values = train[column].fillna(nan_value)
                    acc, time = fast_score(train.copy(), new_column_values, column)
                    
                    if (best_acc*acc_sign>acc*acc_sign):
                        best_acc=acc
                        best_column_values=new_column_values
                        print('fast_score with 0     :', np.round(acc, 4), np.round(time), 'sec. taken')

                    for cat_column in group_feature:
                        nan_tmp = []
                        
                        #count median by group_feature, if group_feature median NAN replased by global median
                        cat_median = DS.groupby(cat_column)[column].median()
                        cat_median = cat_median.fillna(cat_median.median())
                        
                        for index, row in DS.iterrows():
                            if math.isnan(row[column]):
                                if pd.isnull(row[cat_column]) == False and row[cat_column] != '':
                                    nan_tmp.append(cat_median[row[cat_column]])
                                else:
                                    nan_tmp.append(cat_median.median())
                            else:
                                nan_tmp.append(row[column])
                                
                        new_column_values = pd.Series((nan_tmp))
                        
                        acc, time = fast_score(train.copy(), new_column_values, column)
                    
                        if (best_acc*acc_sign>acc*acc_sign):
                            best_acc=acc
                            best_column_values=new_column_values
                            print('fast_score with group_feature ', cat_column, ':', np.round(acc, 4), np.round(time), 'sec. taken')
 

                    DS[column] = best_column_values


                #most frequent for object
                if DS[column].dtype == object:
                    print('\n\n---', DS[column].dtype, column,': NAN ',
                          np.round(DS[column].isna().sum()/rows*100),'%',
                          '(', np.round(DS[column].isna().sum()), ' observations)',
                                        ' replaced ', DS[column].value_counts().idxmax())
                    DS[column] = DS[column].fillna(DS[column].value_counts().idxmax())

            # 2.2 if NA > 50% replace as _isnull column
            else:
                DS[column] = DS[column].isnull()

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