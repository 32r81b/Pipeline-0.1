import pandas as pd
import numpy as np
from usfull_tools import load_DS
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from set_vars import KAGGLE_PREFIX, debug_mode, KAGGLE_DIR, target_column, target_type, loss_function, custom_metric

# pd.options.display.max_columns = None
# %matplotlib inline

train, test = load_DS(debug_mode, KAGGLE_DIR, KAGGLE_PREFIX, '_prepare.csv')
del test

X_train, X_test, y_train, y_test = train_test_split(train[train.columns.drop(target_column)], train[target_column],
                                                    test_size=0.3, random_state=42)
print(X_train.shape)


iterations = np.round(10 + len(X_train.columns)/20)
print('iterations :', iterations)

if target_type == 'binary':
    model = CatBoostClassifier(random_seed=42, iterations=iterations, depth=2, learning_rate=0.1,
                               loss_function=loss_function, custom_metric=loss_function, od_type = 'Iter',
                               task_type='GPU', devices='0')
elif target_type == 'interval':
    model = CatBoostRegressor(random_seed=42, iterations=iterations, depth=12, learning_rate=0.1,
                              loss_function=loss_function, custom_metric=loss_function, od_type = 'Iter',
                              task_type='GPU', devices='0')

#https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/#python-reference_parameters-list

# Для CatBoost требуется явно указывать категориальные переменные
i = 0
cat_features = []
for column in X_train.columns:
    if X_train[column].dtype == 'object': cat_features.append(i)
    i += 1

model.fit(X_train, y_train, cat_features)

from sklearn.metrics import mean_absolute_error, accuracy_score

if custom_metric=='Accuracy':
    print("Accuracy: %.3f"
          % accuracy_score(model.predict(X_test), y_test))

if custom_metric=='RMSE':
    print("RMSE: %.3f"
          % mean_absolute_error(model.predict(X_test), y_test))


feature_importance = pd.DataFrame(list(zip(X_test.dtypes.index,
                                           model.get_feature_importance(Pool(X_test, label=y_test, cat_features=cat_features)))),
                                    columns=['Feature','Score'])

feature_importance = feature_importance.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')

#TO DO: сделать отбор лучшего из нескольких корелирующих параметров



#Keep features with > 1% normalize importance
fi = feature_importance[feature_importance.Score > 1]
print(len(X_train.columns), '->', len(fi.index), 'non zero important features:', np.round(fi.Score.sum(),1), '%')
fi.sort_values('Score', ascending = False)

fi.to_csv(KAGGLE_DIR + KAGGLE_PREFIX + '_important_columns.csv', index=False)
