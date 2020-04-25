def clean_data(train_df, test_df):

    dfs = [train_df, test_df]

    test_id = test_df['Id']

    for df in dfs:
        df.drop(['Id'], axis=1, inplace=True)

    for df in dfs:
        df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)

        df['Alley'].fillna('NA', inplace=True)
        df['PoolQC'].fillna('NA', inplace=True)
        df['MiscFeature'].fillna('NA', inplace=True)
        df['Fence'].fillna('NA', inplace=True)
        df['FireplaceQu'].fillna('NA', inplace=True)
        df['BsmtQual'].fillna('NA', inplace=True)
        df['BsmtCond'].fillna('NA', inplace=True)
        df['BsmtExposure'].fillna('NA', inplace=True)
        df['BsmtFinType1'].fillna('NA', inplace=True)
        df['BsmtFinType2'].fillna('NA', inplace=True)
        df['GarageType'].fillna('NA', inplace=True)
        df['GarageFinish'].fillna('NA', inplace=True)
        df['GarageCond'].fillna('NA', inplace=True)
        df['GarageQual'].fillna('NA', inplace=True)

        df['MasVnrType'].fillna('None', inplace=True)

        df['MasVnrArea'].fillna(0, inplace=True)
        df['BsmtFinSF1'].fillna(0, inplace=True)
        df['BsmtFinSF2'].fillna(0, inplace=True)
        df['BsmtUnfSF'].fillna(0, inplace=True)
        df['TotalBsmtSF'].fillna(0, inplace=True)
        df['BsmtFullBath'].fillna(0, inplace=True)
        df['BsmtHalfBath'].fillna(0, inplace=True)
        df['GarageCars'].fillna(0, inplace=True)
        df['GarageArea'].fillna(0, inplace=True)
        df['GarageYrBlt'].fillna(0, inplace=True)

        df['Exterior1st'].fillna(df['Exterior1st'].mode()[0], inplace=True)
        df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0], inplace=True)
        df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
        df['KitchenQual'].fillna(df['KitchenQual'].mode()[0], inplace=True)
        df['Functional'].fillna(df['Functional'].mode()[0], inplace=True)
        df['SaleType'].fillna(df['SaleType'].mode()[0], inplace=True)

        # Feature Engineering
        df['Remodel_Age'] = df['YearRemodAdd'] - df['YrSold']
        df['House_Age'] = df['YrSold'] - df['YearBuilt']
        df['Garage_Age'] = df['YrSold'] - df['GarageYrBlt']

        df['Garage_Age'] = df["Garage_Age"].apply(lambda x: df['Garage_Age'].median() if x > 1000 else x)

        df['Remodel_Age'] = df['Remodel_Age'].apply(lambda x: 0 if x < 0 else x)
        df['House_Age'] = df['House_Age'].apply(lambda x: 0 if x < 0 else x)
        df['Garage_Age'] = df['Garage_Age'].apply(lambda x: 0 if x < 0 else x)

    neighbor_groups = test_df.groupby(['Neighborhood'])

    test_df['MSZoning'] = neighbor_groups['MSZoning'].apply(
        lambda x: x.fillna(x.mode()[0]))

    test_df['Utilities'] = neighbor_groups['Utilities'].apply(
        lambda x: x.fillna(x.mode()[0]))

    return train_df, test_df, test_id
