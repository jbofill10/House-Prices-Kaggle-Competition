import pandas as pd
import DataCleaning

from EDA import HousingEDA

def main():

    train_df = pd.read_csv('Data/house_prices/train.csv')

    test_df = pd.read_csv('Data/house_prices/test.csv')

    train_df, test_df = DataCleaning.clean_data(train_df, test_df)

    HousingEDA.housing_eda(train_df)

if __name__ == '__main__':
    main()