from xgboost.sklearn import XGBRegressor
from sklearn.feature_selection import RFECV
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np

def find_features(train_df, target):
    features = ['MSSubClass', 'LotFrontage', 'LotArea', 'Street', 'Alley',
                'Utilities', 'LandSlope', 'OverallQual', 'OverallCond',
                'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual',
                'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                'HeatingQC', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
                'FireplaceQu', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',
                'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscVal', 'MoSold', 'YrSold',
                'Remodel_Age', 'House_Age', 'Garage_Age', 'GarageType_Attchd', 'GarageType_Basment',
                'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'GarageType_NA', 'Electrical_FuseF',
                'Electrical_FuseP', 'Electrical_Mix', 'Electrical_SBrkr', 'CentralAir_Y', 'Heating_GasA',
                'Heating_GasW',
                'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'Foundation_CBlock', 'Foundation_PConc',
                'Foundation_Slab',
                'Foundation_Stone', 'Foundation_Wood', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone',
                'Exterior1st_AsphShn',
                'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CBlock', 'Exterior1st_CemntBd',
                'Exterior1st_HdBoard', 'Exterior1st_ImStucc',
                'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_Stone', 'Exterior1st_Stucco',
                'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng',
                'Exterior1st_WdShing', 'Exterior2nd_AsphShn', 'Exterior2nd_Brk Cmn', 'Exterior2nd_BrkFace',
                'Exterior2nd_CBlock', 'Exterior2nd_CmentBd',
                'Exterior2nd_HdBoard', 'Exterior2nd_ImStucc', 'Exterior2nd_MetalSd', 'Exterior2nd_Other',
                'Exterior2nd_Plywood', 'Exterior2nd_Stone', 'Exterior2nd_Stucco',
                'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng', 'Exterior2nd_Wd Shng', 'RoofMatl_CompShg',
                'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv',
                'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip',
                'RoofStyle_Mansard', 'RoofStyle_Shed', 'HouseStyle_1.5Unf',
                'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer',
                'HouseStyle_SLvl', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs',
                'BldgType_TwnhsE', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN',
                'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn',
                'Condition2_Feedr', 'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN', 'Condition2_RRAe',
                'Condition2_RRAn', 'Condition2_RRNn', 'Neighborhood_Blueste', 'Neighborhood_BrDale',
                'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor',
                'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR',
                'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill',
                'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt',
                'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW',
                'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber',
                'Neighborhood_Veenker', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside',
                'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'LotShape_IR2',
                'LotShape_IR3', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM',
                'MiscFeature_NA', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MiscFeature_TenC',
                'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal',
                'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD',
                'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD']

    return xgboost_reg(train_df, target, features)


def xgboost_reg(train_df, target, features):
    if not os.path.isfile('Data/pickles/xgboost_best_features'):
        model = XGBRegressor()

        model.fit(train_df, target)


        feature_importance = pd.DataFrame(model.feature_importances_, index=[features], columns=['Importance']).sort_values(
            ascending=False, by=['Importance'])

        for i in range(len(feature_importance)):
            print('{}: {}'.format(feature_importance.index[i][0], feature_importance['Importance'].iloc[i]))

        model = XGBRegressor(objective='reg:squarederror', verbosity=2)

        elim_features = RFECV(model, cv=3, scoring='neg_root_mean_squared_error')

        elim_features.fit(train_df, target)

        best_features = elim_features.get_support(1)  # the most important features
        best_features_df = train_df[train_df.columns[best_features]]  # final features`

        best_features_df.to_pickle('Data/pickles/xgboost_best_features')
    else:
        df = pd.read_pickle('Data/pickles/xgboost_best_features')

    if not os.path.isfile('Data/pickles/models/xgboost_model'):
        params = {
            'n_estimators': [10, 20, 30, 40, 50, 100, 250, 500],
            'max_depth': [1, 3, 5],
            'learning_rate': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0.1, 1, 5, 10],
            'reg_lambda': [0.1, 1, 5, 10]
        }

        model = XGBRegressor(objective='reg:squarederror')
        grid = GridSearchCV(estimator=model, param_grid=params, verbose=3, cv=3, scoring='neg_root_mean_squared_error')

        grid.fit(df, target)
        print(grid.best_params_)
        with open('Data/pickles/models/xgboost_model', 'wb') as file:
            pickle.dump(grid.best_estimator_, file)

    else:
        model = None
        with open('Data/pickles/models/xgboost_model', 'rb') as file:
            model = pickle.load(file)

        train_split_model = XGBRegressor(n_estimators=250, learning_rate=0.1,
                                         max_depth=3, reg_alpha=0.1, reg_lambda=1)

        x_train, x_test, y_train, y_test = train_test_split(df, target)

        train_split_model.fit(x_train, y_train)

        y_pred = train_split_model.predict(x_test)

    '''best params: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 250, 'reg_alpha': 0.1, 'reg_lambda': 1}'''

    print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

    return model, df


