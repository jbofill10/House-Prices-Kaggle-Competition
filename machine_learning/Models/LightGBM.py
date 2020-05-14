from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

import os
import pickle


def run_lightgbm(train_scaled, target):
    if not os.path.isfile('Data/pickles/models/lightgbm_pickle'):
        model = LGBMRegressor(num_threads=8, metric='rmse')

        params = {
            'boosting_type': ['gbdt', 'goss', 'dart'],
            'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.1, 0.3, 0.5, 0.7, 1],
            'num_leaves': [10, 30, 50, 70, 90, 110],
            'max_depth': [0, 10, 25, 50, 75, 100],
            'lambda_l1': [0, 0.02, 0.05, 0.07, 0.1, 5],
            'lambda_l2': [0, 0.02, 0.05, 0.07, 0.1, 5]

        }

        grid = GridSearchCV(model, param_grid=params, cv=3, verbose=3)

        grid.fit(train_scaled, target)

        print(grid.best_params_)
        with open('Data/pickles/models/lightgbm_pickle', 'wb') as file:

            lightgbm_model = grid.best_estimator_
            pickle.dump(lightgbm_model, file)
    else:
        with open('Data/pickles/models/lightgbm_pickle', 'rb') as file:
            lightgbm_model = pickle.load(file)

    '''Best params: [boosting_type='gbdt', lambda_l1=0, lambda_l2=0,1, learning_rate=0.1, max_depth=0, num_leaves=10]'''
