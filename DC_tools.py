import pandas as pd
import numpy as np

from usfull_tools import numeric_types, numeric_fraction_types, numeric_nonfraction_types
numeric_types = numeric_types()
numeric_nonfraction_types = numeric_nonfraction_types()
numeric_fraction_types = numeric_fraction_types()

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
            # 2.1 if NA < 50% rows
            if DS[column].isna().sum()/rows < 0.5:
                #/min/max/0/most_frequent for non fraction numeric            
                if DS[column].dtype in numeric_nonfraction_types:
# TO DO: протестировать INT. Или NAN с INT уже не получишь?

                    print('-----' ,column ,'------')
                    nan_min = DS[column].fillna(DS[column].min())
                    nan_max = DS[column].fillna(DS[column].max())
                    nan_0 = DS[column].fillna(0)
                    nan_idxmax = DS[column].fillna(DS[column].value_counts().idxmax())
                    print('nan_min corr:', DS[target_column].corr(nan_min))
                    print('nan_max corr:', DS[target_column].corr(nan_max))
                    print('nan_0 corr:', DS[target_column].corr(nan_0))
                    print('nan_idxmax corr:', DS[target_column].corr(nan_idxmax))
                    
                    nan_name = '_nan_idxmax'
                    nan_value = DS[column].fillna(DS[column].value_counts().idxmax())
                    nan_corr = abs(DS[column].fillna(DS[column].value_counts().idxmax()))
                    
                    if abs(nan_corr)<abs(DS[target_column].corr(DS[column].fillna(DS[column].min()))):
                        nan_name = '_nan_min'
                        nan_value = DS[column].fillna(DS[column].min())
                        nan_corr = abs(DS[target_column].corr(nan_min))                    
                           
                    if abs(nan_corr)<abs(DS[target_column].corr(DS[column].fillna(DS[column].max()))):
                        nan_name = '_nan_max'
                        nan_value = DS[column].fillna(DS[column].max())
                        nan_corr = abs(DS[target_column].corr(nan_max))                    
                           
                    if abs(nan_corr)<abs(DS[target_column].corr(DS[column].fillna(0))):
                        nan_name = '_nan_0'
                        nan_value = DS[column].fillna(0)
                        nan_corr = abs(DS[target_column].corr(DS[column].fillna(0)))
                        
                    print(nan_name, nan_corr)
                    DS[column + nan_name] = nan_value
                    print (DS[column].dtype, column, np.round(DS[column].isna().sum()/rows*100), '% NAN replaced by: ', nan_name)
                    
                #median/min/max/0 for fraction numeric            
                if DS[column].dtype in numeric_fraction_types:
#                     print('-----' ,column ,'------')
#                     nan_min = DS[column].fillna(DS[column].min())
#                     nan_max = DS[column].fillna(DS[column].max())
#                     nan_0 = DS[column].fillna(0)
#                     nan_median = DS[column].fillna(DS[column].median())
#                     print('nan_min corr:', DS[target_column].corr(nan_min))
#                     print('nan_max corr:', DS[target_column].corr(nan_max))
#                     print('nan_0 corr:', DS[target_column].corr(nan_0))
#                     print('nan_median corr:', DS[target_column].corr(nan_median))
                    
                    nan_name = '_nan_median'
                    nan_value = DS[column].fillna(DS[column].median())
                    nan_corr = abs(DS[target_column].corr(DS[column].fillna(DS[column].median())))
                    
                    if abs(nan_corr)<abs(DS[target_column].corr(DS[column].fillna(DS[column].min()))):
                        nan_name = '_nan_min'
                        nan_value = DS[column].fillna(DS[column].min())
                        nan_corr = abs(DS[target_column].corr(DS[column].fillna(DS[column].min())))                    
                           
                    if abs(nan_corr)<abs(DS[target_column].corr(DS[column].fillna(DS[column].max()))):
                        nan_name = '_nan_max'
                        nan_value = DS[column].fillna(DS[column].max())
                        nan_corr = abs(DS[target_column].corr(DS[column].fillna(DS[column].max())))                    
                           
                    if abs(nan_corr)<abs(DS[target_column].corr(DS[column].fillna(0))):
                        nan_name = '_nan_0'
                        nan_value = DS[column].fillna(0)
                        nan_corr = abs(DS[target_column].corr(DS[column].fillna(0)))
                        
                    DS[column + nan_name] = nan_value
                    print (DS[column].dtype, column, np.round(DS[column].isna().sum()/rows*100), '% NAN replaced by: ', nan_name)

                #most frequent for object
                if DS[column].dtype == object:
                    print(DS[column].dtype, column,': NAN ',np.round(DS[column].isna().sum()/rows*100),'%',' replaced ', DS[column].value_counts().idxmax())
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