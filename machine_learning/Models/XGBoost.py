import pandas as pd
import numpy as np


def run(model, test_df, test_id):

    y_pred = model.predict(test_df)

    y_pred = np.exp(y_pred)

    submission_df = pd.DataFrame(y_pred, index=test_id, columns=['SalePrice'])

    submission_df.to_csv('Data/Submission/S4.csv')