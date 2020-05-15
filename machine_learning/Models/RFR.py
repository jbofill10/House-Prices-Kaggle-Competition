import os
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


def runRFR(train_scaled, target):
    if not os.path.isfile('Data/pickles/models/rfr_model'):
        model = RandomForestRegressor()

        grid = paramTune(train_scaled, target, model);

        with open('Data/pickles/models/rfr_model', 'wb') as file:
            rfr_model = grid.best_estimator_
            pickle.dump(rfr_model, file)
    else:
        with open('Data/pickles/models/rfr_model', 'rb') as file:
            rfr_model = pickle.load(file)

    print(rfr_model)


def paramTune(train_scaled, target, model):
    params = {
        'n_estimators': [10, 50, 100, 300, 500, 700, 1000, 1500, 2000],
        'max_depth': [1, 3, 5],
        'bootstrap': [True, False],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'oob_score': [True, False]
    }

    grid = GridSearchCV(model, param_grid=params, cv=3, verbose=3, n_jobs=-1)

    grid.fit(train_scaled, target)

    return grid
