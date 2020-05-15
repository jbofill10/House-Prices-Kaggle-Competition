from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV
from lightgbm import LGBMRegressor

import numpy as np
import pandas as pd
import os
import pickle


def init_stacking(train_scaled, test_scaled, target, test_id):
    if not os.path.isfile('Data/pickles/models/pancake_stack'):

        estimators = [
            ('rfr', RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                                          max_depth=5, max_features='auto', max_leaf_nodes=None,
                                          max_samples=None, min_impurity_decrease=0.0,
                                          min_impurity_split=None, min_samples_leaf=4,
                                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                                          n_estimators=700, n_jobs=None, oob_score=True,
                                          random_state=None, verbose=3, warm_start=False)),

            ('xgboost', XGBRegressor(learning_rate=0.08, max_depth=3, n_estimators=500, n_jobs=-1,
                                     reg_alpha=0.001, reg_lambda=1, verbosity=2)),

            ('svr', SVR(C=5, cache_size=200, coef0=0.0, degree=1, epsilon=0.01, gamma='auto',
                        kernel='poly', max_iter=-1, shrinking=True, tol=0.001, verbose=3)),

            ('lgbm', LGBMRegressor(boosting_type='gbdt', lambda_l1=0,
                                   lambda_l2=0.1, learning_rate=0.1,
                                   max_depth=0, num_leaves=10))
        ]

        stack = StackingRegressor(estimators=estimators, final_estimator=LassoCV(cv=5), verbose=3)

        stack.fit(train_scaled, target)

        with open('Data/pickles/models/pancake_stack', 'wb') as file:
            pass
            pickle.dump(stack, file)

    else:
        with open('Data/pickles/models/pancake_stack', 'rb') as file:
            stack = pickle.load(file)

    y_pred = stack.predict(test_scaled)

    y_pred = np.exp(y_pred)

    submission_df = pd.DataFrame(y_pred, index=test_id, columns=['SalePrice'])

    submission_df.to_csv('Data/Submission/S6.csv')
