debug_mode = True

KAGGLE_PREFIX = ''
#KAGGLE_DIR = 'C:/Users/Acer/PycharmProjects/Pipeline-0.1/input_pubg-finish-placement-prediction/';KAGGLE_PREFIX = '_V2';target_column = 'winPlacePerc';



#KAGGLE_DIR = 'C:/Users/Ruslan/Documents/GitHub/Pipeline-0.1/input_titanic/'; target_column = 'Survived';
#id_column= 'PassengerId'; target_type='binary'; forced_dtype = dict[];loss_function='CrossEntropy';custom_metric='Accuracy';


KAGGLE_DIR = 'C:/Users/Ruslan/Documents/GitHub/Pipeline-0.1/input_house/';target_column ='SalePrice';id_column= 'Id';target_type='interval';
forced_dtype = {'BsmtFinSF1': 'float16', 'BsmtFinSF2': 'float16', 'BsmtUnfSF': 'float16', 'TotalBsmtSF': 'float16', 'GarageArea': 'float16',
                'BsmtFullBath': 'float16', 'BsmtHalfBath': 'float16', 'GarageCars': 'float16'}
loss_function='RMSE';custom_metric='RMSE';