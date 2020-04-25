import pandas as pd
import DataCleaning
from machine_learning import Preprocessing, FindBestFeatures
from machine_learning.Models import XGBoost
from EDA import HousingEDA


def main():

    train_df = pd.read_csv('Data/house_prices/train.csv')

    test_df = pd.read_csv('Data/house_prices/test.csv')

    train_df, test_df, test_id = DataCleaning.clean_data(train_df, test_df)

    HousingEDA.housing_eda(train_df)

    train_scaled, x_target, test_scaled = Preprocessing.preprocess(train_df, test_df)

    model, best_feats = FindBestFeatures.find_features(train_scaled, x_target)

    test_scaled = test_scaled[[i for i in best_feats]]

    XGBoost.run(model, test_scaled, test_id)


if __name__ == '__main__':
    main()