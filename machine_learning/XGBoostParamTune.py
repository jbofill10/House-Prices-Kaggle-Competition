from xgboost.sklearn import XGBRegressor
from sklearn.feature_selection import RFECV
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np


def param_tune(train_df, target):

    return xgboost_reg(train_df, target)


def xgboost_reg(train_df, target):

    if not os.path.isfile('Data/pickles/models/xgboost_model'):
        params = {
            'n_estimators': [10, 20, 30, 40, 50, 100, 250, 500, 1000],
            'max_depth': [1, 3, 5],
            'learning_rate': [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.1, 0.3, 0.5, 0.7, 1],
            'reg_alpha': [0, 0.001, 0.1, 0.5, 1, 2, 5],
            'reg_lambda': [0, 0.001, 0.1, 1, 2, 5],
            'n_jobs': [3],
            'early_stopping_rounds': [6]

        }

        model = XGBRegressor(objective='reg:linear')
        grid = GridSearchCV(estimator=model, param_grid=params, verbose=3, cv=3, scoring='neg_root_mean_squared_error')

        grid.fit(train_df, target)

        print(grid.best_params_)
        with open('Data/pickles/models/xgboost_model', 'wb') as file:
            boost_model = grid.best_estimator_;
            pickle.dump(boost_model, file);



    else:
        with open('Data/pickles/models/xgboost_model', 'rb') as file:
            model = pickle.load(file)

    train_split_model = XGBRegressor(objective='reg:linear', learning_rate=0.08, max_depth=3, n_estimators=500, n_jobs=3, reg_alpha=0.001, reg_lambda=1)

    x_train, x_test, y_train, y_test = train_test_split(train_df, target)

    train_split_model.fit(x_train, y_train)

    y_pred = train_split_model.predict(x_test)

    '''best params: {'learning_rate': 0.08, 'max_depth': 3, 'n_estimators': 500, 'n_jobs': 3, 'reg_alpha': 0.001, 'reg_lambda': 1}'''

    print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

    return model

