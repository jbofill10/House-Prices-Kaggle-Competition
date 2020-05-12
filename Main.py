import pandas as pd
import DataCleaning
from machine_learning import Preprocessing, XGBoostParamTune
from machine_learning.Models import XGBoost, SVR, RFR
from machine_learning.Stacking import Stacking
from EDA import HousingEDA


def main():

    train_df = pd.read_csv('Data/house_prices/train.csv')

    test_df = pd.read_csv('Data/house_prices/test.csv')

    train_df, test_df, test_id, train_test_df, target = DataCleaning.clean_data(train_df, test_df)

    #HousingEDA.housing_eda(train_df)

    train_df = train_test_df.iloc[:len(train_df)]
    test_df = train_test_df.iloc[len(train_df):, :]

    train_scaled, test_scaled = Preprocessing.preprocess(train_test_df, train_df, test_df)

    #model = XGBoostParamTune.param_tune(train_scaled, target)

    #SVR.runSVM(train_scaled, target)

    #RFR.runRFR(train_scaled, target)

    Stacking.init_stacking(train_scaled, test_scaled, target, test_id)

    #XGBoost.run(model, test_scaled, test_id)


if __name__ == '__main__':
    main()