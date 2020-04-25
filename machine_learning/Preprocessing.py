from sklearn.preprocessing import RobustScaler

import pandas as pd
import numpy as np


def preprocess(train_df, test_df):

    dfs = [train_df, test_df]

    encoded_features = ['Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual',
                        'ExterCond', 'ExterQual', 'Fence', 'FireplaceQu', 'Functional', 'GarageCond',
                        'GarageQual', 'GarageFinish', 'HeatingQC', 'KitchenQual', 'LandSlope', 'PavedDrive', 'PoolQC', 'Street', 'Utilities']

    one_hot_features = ['GarageType', 'Electrical', 'CentralAir', 'Heating', 'Foundation', 'MasVnrType', 'Exterior1st',
                          'Exterior2nd', 'RoofMatl', 'RoofStyle', 'HouseStyle', 'BldgType', 'Condition1', 'Condition2',
                          'Neighborhood', 'LotConfig', 'LandContour', 'LotShape', 'MSZoning', 'MiscFeature', 'SaleCondition', 'SaleType']

    for df in dfs:
        for col in df.columns:
            if col not in one_hot_features and col not in encoded_features:
                df[col] = np.log1p(df[col])

    for df in dfs:
        df['Alley'] = df['Alley'].map({'NA': 0, 'Grvl': 1, 'Pave': 2})

        df['BsmtCond'] = df['BsmtCond'].map({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
        df['BsmtExposure'] = df['BsmtExposure'].map({'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
        df['BsmtFinType1'] = df['BsmtFinType1'].map({'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
        df['BsmtFinType2'] = df['BsmtFinType2'].map({'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
        df['BsmtQual'] = df['BsmtQual'].map({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        df['ExterCond'] = df['ExterCond'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
        df['ExterQual'] = df['ExterQual'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        df['Fence'] = df['Fence'].map({'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4})

        df['FireplaceQu'] = df['FireplaceQu'].map({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        df['Functional'] = df['Functional'].map({'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8})

        df['GarageCond'] = df['GarageCond'].map({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
        df['GarageQual'] = df['GarageQual'].map({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
        df['GarageFinish'] = df['GarageFinish'].map({'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})

        df['HeatingQC'] = df['HeatingQC'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
        df['KitchenQual'] = df['KitchenQual'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        df['LandSlope'] = df['LandSlope'].map({'Sev': 1, 'Mod': 2, 'Gtl': 3})

        df['PavedDrive'] = df['PavedDrive'].map({'N': 1, 'P': 2, 'Y': 3})

        df['PoolQC'] = df['PoolQC'].map({'NA': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

        df['Street'] = df['Street'].map({'Grvl': 1, 'Pave': 2})

        df['Utilities'] = df['Utilities'].map({'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4})

    '''One hot encode:
        GarageType
        Electrical
        CentralAir
        Heating
        Foundation
        MasVnrType
        Exterior2nd
        Exterior1st
        RoofMatl
        RoofStyle
        HouseStyle
        BldgType
        Condition1
        Condition2
        Neighborhood
        LotConfig
        LandContour
        LotShape
        MSZoning
        MiscFeature
        SaleCondition
        SaleType'''

    train_encoded = pd.get_dummies(train_df, prefix=one_hot_features, columns=one_hot_features, drop_first=True)

    test_encoded = pd.get_dummies(test_df, prefix=one_hot_features, columns=one_hot_features, drop_first=True)

    train_scaled = train_encoded.copy()
    test_scaled = test_encoded.copy()

    scaler = RobustScaler()

    target = train_encoded['SalePrice']
    train_encoded.drop(['SalePrice'], axis=1, inplace=True)

    # Add missing cols to test
    missing = ['Electrical_Mix', 'Heating_GasA', 'Heating_OthW', 'Exterior1st_ImStucc', 'Exterior1st_Stone', 'Exterior2nd_Other', 'RoofMatl_CompShg',
               'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'HouseStyle_2.5Fin', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'MiscFeature_TenC']

    for col in missing:
        test_encoded[col] = 0

    train_scaled = pd.DataFrame(scaler.fit_transform(train_encoded), columns=train_encoded.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_encoded), columns=test_encoded.columns)

    return train_scaled, target, test_scaled

