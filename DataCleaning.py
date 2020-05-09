import pandas as pd
import numpy as np

from scipy.stats import skew


def clean_data(train_df, test_df):
    dfs = [train_df, test_df]

    target = train_df['SalePrice'].reset_index(drop=True)
    train_features = train_df.drop(['SalePrice'], axis=1)

    train_test_df = pd.concat([train_features, test_df]).reset_index(drop=True)

    test_id = test_df['Id']

    train_test_df.drop(['Id'], axis=1, inplace=True)
    train_test_df['MSSubClass'] = train_test_df['MSSubClass'].astype(str)
    train_test_df['YrSold'] = train_test_df['YrSold'].astype(str)
    train_test_df['MoSold'] = train_test_df['MoSold'].astype(str)

    train_test_df['LotFrontage'] = train_test_df.groupby(['Neighborhood'])['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )

    train_test_df['Alley'].fillna('NA', inplace=True)
    train_test_df['PoolQC'].fillna('NA', inplace=True)
    train_test_df['MiscFeature'].fillna('NA', inplace=True)
    train_test_df['Fence'].fillna('NA', inplace=True)
    train_test_df['FireplaceQu'].fillna('NA', inplace=True)
    train_test_df['BsmtQual'].fillna('NA', inplace=True)
    train_test_df['BsmtCond'].fillna('NA', inplace=True)
    train_test_df['BsmtExposure'].fillna('NA', inplace=True)
    train_test_df['BsmtFinType1'].fillna('NA', inplace=True)
    train_test_df['BsmtFinType2'].fillna('NA', inplace=True)
    train_test_df['GarageType'].fillna('NA', inplace=True)
    train_test_df['GarageFinish'].fillna('NA', inplace=True)
    train_test_df['GarageCond'].fillna('NA', inplace=True)
    train_test_df['GarageQual'].fillna('NA', inplace=True)

    train_test_df['MasVnrType'].fillna('None', inplace=True)

    train_test_df['MasVnrArea'].fillna(0, inplace=True)
    train_test_df['BsmtFinSF1'].fillna(0, inplace=True)
    train_test_df['BsmtFinSF2'].fillna(0, inplace=True)
    train_test_df['BsmtUnfSF'].fillna(0, inplace=True)
    train_test_df['TotalBsmtSF'].fillna(0, inplace=True)
    train_test_df['BsmtFullBath'].fillna(0, inplace=True)
    train_test_df['BsmtHalfBath'].fillna(0, inplace=True)
    train_test_df['GarageCars'].fillna(0, inplace=True)
    train_test_df['GarageArea'].fillna(0, inplace=True)
    train_test_df['GarageYrBlt'].fillna(0, inplace=True)

    train_test_df['Exterior1st'].fillna(train_test_df['Exterior1st'].mode()[0], inplace=True)
    train_test_df['Exterior2nd'].fillna(train_test_df['Exterior2nd'].mode()[0], inplace=True)
    train_test_df['Electrical'].fillna(train_test_df['Electrical'].mode()[0], inplace=True)
    train_test_df['KitchenQual'].fillna(train_test_df['KitchenQual'].mode()[0], inplace=True)
    train_test_df['Functional'].fillna(train_test_df['Functional'].mode()[0], inplace=True)
    train_test_df['SaleType'].fillna(train_test_df['SaleType'].mode()[0], inplace=True)

    # Normalize Data
    target = np.log1p(target)
    train_test_df = normalize_data(train_test_df)

    # Feature Engineering
    neighbor_groups = train_test_df.groupby(['Neighborhood'])

    train_test_df['MSZoning'] = neighbor_groups['MSZoning'].apply(
        lambda x: x.fillna(x.mode()[0]))

    train_test_df['Utilities'] = neighbor_groups['Utilities'].apply(
        lambda x: x.fillna(x.mode()[0]))

    train_test_df['BsmtFinType1_Unf'] = 1 * (train_test_df['BsmtFinType1'] == 'Unf')
    train_test_df['Has_WoodDeck'] = (train_test_df['WoodDeckSF'] == 0) * 1
    train_test_df['Has_OpenPorch'] = (train_test_df['OpenPorchSF'] == 0) * 1
    train_test_df['Has_EnclosedPorch'] = (train_test_df['EnclosedPorch'] == 0) * 1
    train_test_df['Has_3SsnPorch'] = (train_test_df['3SsnPorch'] == 0) * 1
    train_test_df['Has_ScreenPorch'] = (train_test_df['ScreenPorch'] == 0) * 1
    train_test_df['Years_SinceRemodel'] = train_test_df['YrSold'].astype(int) - train_test_df['YearRemodAdd'].astype(
        int)

    train_test_df['Total_Home_Quality'] = train_test_df['OverallQual'] + train_test_df['OverallCond']
    train_test_df = train_test_df.drop(['Utilities', 'Street', 'PoolQC', ], axis=1)
    train_test_df['TotalSF'] = train_test_df['TotalBsmtSF'] + train_test_df['1stFlrSF'] + train_test_df['2ndFlrSF']
    train_test_df['YrBltAndRemod'] = train_test_df['YearBuilt'] + train_test_df['YearRemodAdd']

    train_test_df['Total_sqr_footage'] = (train_test_df['BsmtFinSF1'] + train_test_df['BsmtFinSF2'] +
                                          train_test_df['1stFlrSF'] + train_test_df['2ndFlrSF'])

    train_test_df['Total_Bathrooms'] = (train_test_df['FullBath'] + (0.5 * train_test_df['HalfBath']) +
                                        train_test_df['BsmtFullBath'] + (0.5 * train_test_df['BsmtHalfBath']))

    train_test_df['Total_porch_sf'] = (train_test_df['OpenPorchSF'] + train_test_df['3SsnPorch'] +
                                       train_test_df['EnclosedPorch'] + train_test_df['ScreenPorch'] +
                                       train_test_df['WoodDeckSF'])

    train_test_df['Has_Pool'] = train_test_df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    train_test_df['Has_2ndFloor'] = train_test_df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    train_test_df['Has_Garage'] = train_test_df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    train_test_df['Has_Bsmt'] = train_test_df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    train_test_df['Has_Fireplace'] = train_test_df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    train_test_df['TotalBsmtSF'] = train_test_df['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    train_test_df['2ndFlrSF'] = train_test_df['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
    train_test_df['GarageArea'] = train_test_df['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
    train_test_df['GarageCars'] = train_test_df['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
    train_test_df['LotFrontage'] = train_test_df['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
    train_test_df['MasVnrArea'] = train_test_df['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
    train_test_df['BsmtFinSF1'] = train_test_df['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

    return train_df, test_df, test_id, train_test_df, target


def normalize_data(df):
    skewness = dict()

    numerical_cols = [i for i in df if df[i].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']]

    for i in numerical_cols:
        skewness[i] = skew(df[i])

    skewness = {k: v for k, v in sorted(skewness.items(), key=lambda item: item[1])}

    # [print("{}: {}".format(i, skewness[i])) for i in skewness]

    for key, value in skewness.items():
        if value > 0.5:
            df[key] = np.log1p(df[key])

    return df
