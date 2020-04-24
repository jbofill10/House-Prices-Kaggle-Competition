import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns


def housing_eda(train_df):
    # Look at target variable

    f = px.histogram(train_df, x='SalePrice', nbins=20)
    f.update_layout(
        title='Sale Price Distribution',
        width=1600,
        height=1000,
        xaxis_title='Sale Price'
    )

    iplot(f)
    log_target = np.log(train_df['SalePrice']+1)

    print(log_target)

    f = px.histogram(x=log_target, nbins=20)

    f.update_layout(
        title='Sale Price (Log) Distribution',
        width=1600,
        height=1000,
        xaxis_title='Sale Price (Log)'
    )

    iplot(f)

    numerical_corr = train_df.corr(method='pearson')

    make_heat_map(numerical_corr, [15,10], 'Correlation Between Numerical Features to Target Label', 'NumericalCorr', annot=False)


    numerical_corr = train_df[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'YearBuilt', 'YearRemodAdd',
                                         'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces',
                                         'GarageYrBlt', 'LotArea', 'BsmtUnfSF', 'SalePrice']].copy()

    make_heat_map(numerical_corr.corr(method='pearson'), [15,10],
                  'Correlation Between Stronger Numerical Features to Target Label', 'StrongerNumericalCorrs', annot=True)

    numerical_corr = train_df[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'YearBuilt', 'YearRemodAdd',
                                         'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces',
                                         'GarageYrBlt','SalePrice']].copy()

    make_heat_map(numerical_corr.corr(method='pearson'), [15, 10], 'Strongest Correlating Numerical Features to Target Label', name='StrongestNumericalCorrelatingFeaturesToTargetLabel', annot=True)

    categorical_df = pd.get_dummies(train_df.select_dtypes(include=[np.object]).copy(), drop_first=True)
    categorical_df_s1 = categorical_df[categorical_df.columns[:100]].copy()
    categorical_df_s2 = categorical_df[categorical_df.columns[101:]].copy()
    categorical_df_s1['SalePrice'] = train_df['SalePrice']
    categorical_df_s2['SalePrice'] = train_df['SalePrice']

    categorical_df_s1 = categorical_df_s1.corr(method='spearman')
    categorical_df_s2 = categorical_df_s2.corr(method='spearman')

    make_heat_map(categorical_df_s1, [25, 25],
                  'Correlation Between Categorical Features and Target Label (Columns 0-100)',
                  'CategoricalCorr_Col_0_100', annot=False)

    make_heat_map(categorical_df_s2, [25, 25],
                  'Correlation Between Categorical Features and Target Label (Columns 101-209)',
                  'CategoricalCorr_Col_101_209', annot=False)


def make_heat_map(df, figsize, title, name, annot):
    plt.subplots(1, 1, figsize=(figsize[0], figsize[1]))
    sns.heatmap(df, cmap='Blues', annot=annot, annot_kws={'size': 5})
    plt.title(title)
    plt.savefig('Charts/{}.png'.format(name), bbox_inches='tight')
    plt.show()
