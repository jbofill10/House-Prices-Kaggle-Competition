import os
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


def runSVM(train_scaled, target):
    if not os.path.isfile('Data/pickles/models/svr_model'):
        model = SVR()

        grid = paramTune(train_scaled, target, model)

        print(grid.best_params_)
        with open('Data/pickles/models/svr_model', 'wb') as file:
            svr_model = grid.best_estimator_
            pickle.dump(svr_model, file)
    else:
        with open('Data/pickles/models/svr_model', 'rb') as file:
            svr_model = pickle.load(file)

    print(svr_model)


def paramTune(train_scaled, target, model):

    params = {
        'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
        'C': [0.1, 0.5, 1, 3, 5],
        'degree': [1, 2, 3, 5],
        'epsilon': [0.01, 0.5, 1],
        'gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(model, param_grid=params, cv=3, scoring='neg_root_mean_squared_error', verbose=3)

    grid.fit(train_scaled, target)

    return grid
