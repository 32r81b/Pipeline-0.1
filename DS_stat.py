import pandas as pd
import numpy as np

from usfull_tools import numeric_types
numeric_types = numeric_types()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

def DS_stat(DS_nan, target_column, role):
    categorical_max_size = 20  # if less then determinate as categorical
    
    rows_nan, cols_nan = DS_nan.shape
    skew = DS_nan.skew()
    print('Shape:', rows_nan, ' * ', cols_nan)
    print('First 5 rows from ', role, ':')
    display(DS_nan.head())
  
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

            
            if DS[column].nunique() > categorical_max_size:
                print('5% / 95% percentile:', np.round(np.percentile(DS[column], 5),2), ' / ', np.round(np.percentile(DS[column], 95),2))

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
            if DS[column].nunique() < categorical_max_size:
                print(DS[column].value_counts())
                if DS[target_column].nunique() == 2:
                    sns.barplot(x = column, y = target_column, data=DS)
                    plt.show()
                else:
                    sns.boxplot(x = target_column, y = column, data=DS)
                    plt.show()
