import pandas as pd
import numpy as np
import scipy.stats as st

from usfull_tools import numeric_types
numeric_types = numeric_types()
from set_vars import group_feature_detection_trashold

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

def DS_stat(DS_nan, target_column = ''):    
    rows_nan, cols_nan = DS_nan.shape
    skew = DS_nan.skew()
    print('Shape:', rows_nan, ' * ', cols_nan)
    print('First 5 rows:')
    display(DS_nan.head(10))
    
    group_feature = []
    
    for column in DS_nan.columns:
        print('\n------------------', column, '---------------------')
        print('Nan values: ', DS_nan[column].isna().sum(), ' (', np.round(DS_nan[column].isna().sum()/rows_nan*100,2), '%)')
        DS=DS_nan.dropna(subset=[column])
        rows, cols = DS.shape
        print('Most frequent: ', DS[column].value_counts().idxmax(), ' - ', 
              DS[column].value_counts().max(), ' (', np.round(DS[column].value_counts().max() / rows * 100, 2),'%)')
        
        if (DS[column].dtypes in numeric_types):            
            print('Unique values:', DS[column].nunique())
            print('Min / Max:', DS[column].min(), ' / ', DS[column].max())
            if skew[column]>3: print('Scew:', np.round(skew[column], 2))

            
            if DS[column].nunique() > group_feature_detection_trashold:
                print('5% / 95% percentile:', np.round(np.percentile(DS[column], 5),2), ' / ', np.round(np.percentile(DS[column], 95),2))
            elif column != target_column:
                group_feature.append(column)

            sns.distplot(DS[column], kde=False)
            plt.show()
            
            not_null_counter = DS[column].count()
            sort = DS[column].sort_values()
            sort = sort[int(not_null_counter*0.05):int(not_null_counter*0.95)]
            ax = sns.distplot(sort, kde=False)
            ax.set(xlabel=str(column + ' (remove first and last 5%)'))
            plt.show()
        
        if DS[column].dtypes == 'object':
            print('Unique values of:', DS[column].nunique(), ' (', np.round((DS[column].nunique()/DS[column].count())*100,2),'%)')
            if DS[column].nunique() <= group_feature_detection_trashold:
                print(DS[column].value_counts())
                if column != target_column:
                    group_feature.append(column)
                if target_column != '':
                    if DS[target_column].nunique() == 2:
                        sns.barplot(x = column, y = target_column, data=DS)
                        plt.show()
                    else:
                        sns.boxplot(x = target_column, y = column, data=DS)
                        plt.show()
    if target_column != '':
        import csv
        from set_vars import KAGGLE_PREFIX, KAGGLE_DIR
        with open(KAGGLE_DIR + KAGGLE_PREFIX + 'group_feature.csv', 'w') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(group_feature)