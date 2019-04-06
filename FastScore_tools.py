import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import scipy as sp 
import scipy.stats 


from set_vars import target_column, target_type
from usfull_tools import numeric_types, numeric_fraction_types, numeric_nonfraction_types
numeric_types = numeric_types()
numeric_nonfraction_types = numeric_nonfraction_types()
numeric_fraction_types = numeric_fraction_types()

def fast_score(train, new_column_values, old_column):
    start = time. time()
    train_tmp = train.drop(old_column, axis=1)

    #Числовые NAN заменеям на median
    for column in train_tmp.columns:
        if train_tmp[column].isna().sum() > 0:
            if train_tmp[column].dtype in numeric_types:
                nan_value = train_tmp[column].median()
                train_tmp[column] = train_tmp[column].fillna(nan_value)    
    
    train_tmp['new_column'] = new_column_values
    y = train_tmp[target_column]
    train_tmp.drop(target_column, axis=1, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(train_tmp, y, test_size=0.3, random_state=42)
    
    if target_type=='binary':
        model = LogisticRegression(solver='liblinear')
    elif target_type=='interval':
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
        
    if target_type=='binary':
         return accuracy_score(y_test, y_pred), time. time() - start
    elif target_type=='interval':
         return mean_absolute_error(y_test, y_pred), time. time() - start
        
        
def mean_confidence_interval(data, confidence=0.95): 
    a = 1.0*np.array(data) 
    n = len(a) 
    m, se = np.mean(a), scipy.stats.sem(a) 
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1) 
    return m, m-h, m+h 