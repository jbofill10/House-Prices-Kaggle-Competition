from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

def run_lasso(train_scaled, target):
    params = {
        'alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    }

    grid = GridSearchCV(Lasso(), param_grid=params, cv=5, verbose=3)

    grid.fit(train_scaled, target)

    print(grid.best_params_)
