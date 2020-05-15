from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os
import pickle
import numpy as np
import pandas as pd
import time
import sys


def weighted_avg(train_scaled, test_scaled, target, test_id):
    rfr = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                                max_depth=5, max_features='auto', max_leaf_nodes=None,
                                max_samples=None, min_impurity_decrease=0.0,
                                min_impurity_split=None, min_samples_leaf=4,
                                min_samples_split=2, min_weight_fraction_leaf=0.0,
                                n_estimators=700, n_jobs=-1, oob_score=True,
                                random_state=None, verbose=3, warm_start=False)

    xgboost = XGBRegressor(learning_rate=0.08, max_depth=3, n_estimators=500, n_jobs=-1,
                           reg_alpha=0.001, reg_lambda=1, verbosity=2)

    svr = SVR(C=5, cache_size=200, coef0=0.0, degree=1, epsilon=0.01, gamma='auto',
              kernel='poly', max_iter=-1, shrinking=True, tol=0.001, verbose=3)

    lgbm = LGBMRegressor(boosting_type='gbdt', lambda_l1=0,
                         lambda_l2=0.1, learning_rate=0.1,
                         max_depth=0, num_leaves=10, n_jobs=-1)

    if not os.path.isfile('Data/pickles/weighted_combs'):

        estimators = {'rfr': rfr, 'xgboost': xgboost, 'svr': svr, 'lgbm': lgbm}
        start = time.time()
        scores = find_weights(estimators, train_scaled, target)
        end = time.time() - start
        print(end)
        with open('Data/pickles/weighted_combs', 'wb') as file:
            pass
            pickle.dump(scores, file)

    else:
        with open('Data/pickles/weighted_combs', 'rb') as file:
            scores = pickle.load(file)

    print(scores)
    weight_comb = ''
    rmse_score = sys.maxsize

    for key, value in scores.items():
        if key < rmse_score:
            weight_comb = value
            rmse_score = key

    best_weights = weight_comb.split(',')
    print(weight_comb)

    '''x_train, x_test, y_train, y_test = train_test_split(train_scaled, target)

    for model_type, model in estimators.items():
        cv_results.append(f'{model_type}: {np.mean(cross_val_score(model, x_train, y_train, cv=5))}')

    [print(i) for i in cv_results]'''

    rfr.fit(train_scaled, target)
    xgboost.fit(train_scaled, target)
    svr.fit(train_scaled, target)
    lgbm.fit(train_scaled, target)

    rfr_y_pred = np.expm1(rfr.predict(test_scaled))
    xgboost_y_pred = np.expm1(xgboost.predict(test_scaled))
    svr_y_pred = np.expm1(svr.predict(test_scaled))
    lgbm_y_pred = np.expm1(lgbm.predict(test_scaled))

    y_pred = (rfr_y_pred * float(best_weights[0])) + (xgboost_y_pred * float(best_weights[1])) + \
             (svr_y_pred * float(best_weights[2])) + (lgbm_y_pred * float(best_weights[3]))

    submission_df = pd.DataFrame(y_pred, index=test_id, columns=['SalePrice'])

    submission_df.to_csv('Data/Submission/S8.csv')


def find_weights(estimators, train_scaled, target):
    x_train, x_test, y_train, y_test = train_test_split(train_scaled, target, shuffle=True)
    scores = {}
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for name, model in estimators.items():
        model.fit(x_train, y_train)

    for i in weights:
        for j in weights:
            for k in weights:
                for l in weights:
                    y_pred = (estimators['rfr'].predict(x_test) * i) + (estimators['xgboost'].predict(x_test) * j) + \
                             (estimators['svr'].predict(x_test) * k) + (estimators['lgbm'].predict(x_test) * l)
                    scores[np.sqrt(mean_squared_error(y_test, y_pred))] = f'{i},{j},{k},{l}'

    return scores
