from sklearn.preprocessing import RobustScaler

import pandas as pd


def preprocess(train_test_df, train_df, test_df):

    train_test_encoded = pd.get_dummies(train_test_df)

    train_encoded = train_test_encoded.iloc[:len(train_df)]
    test_encoded = train_test_encoded.iloc[len(test_df)+1:]

    print(train_encoded.shape)
    print(test_encoded.shape)

    scaler = RobustScaler()

    train_scaled = scaler.fit_transform(train_encoded)
    test_scaled = scaler.transform(test_encoded)

    return train_scaled, test_scaled

