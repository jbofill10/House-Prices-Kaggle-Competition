# House-Prices-Kaggle-Competition

Predicting the prices of houses from a data set considering of 80 features and 1460 records

# EDA
Let's look at the target variable first

![image](Charts/TargetDistribution.png)

It's clear that there is a right skew in the Sales Price. That means a log transformation is needed

![image](Charts/TargetDistribution_Log.png)

That looks a lot better to work with.

# Corollary Features
Since there are 80 features in this data set, some categorical and some numerical, I will first compare the numerical features, then compare the categorical, and then combine the two.

## Numerical Features Matrix
![image](Charts/NumericalCorr.png)

Correlations with Target Label:  
- Really Strong
    - OverallQual
    - GrLivArea (Above Ground Living Area)
    - GarageCars
    - GarageArea
- Strong
    - Year Built
    - YearRemodAdd (Year Remodeled)
    - MasVnrArea
    - TotalBsmtSF
    - 1stFlrSF
    - FullBath
    - TotRmsAbvGrd (Total Rooms Above Ground)
    - Fireplaces
    - GarageYrBlt
- OK
    - LotArea
    - BsmtUnfSF (Unfinished square feet of basement area)
    - 2ndFlrSF
    - BsmtFullBath
    - HalfBath
    - WoodDeckSF
    - OpenPorchSF  

I'll take these features and put them into their own correlation matrix so I can more accurately see their Pearson scores and eliminate more.
![image](Charts/StrongerNumericalCorrs.png)  

This is a bit better, but I have some features with p-values close to 0. So I will remove those as well

![image](Charts/StrongestNumericalCorrelatingFeaturesToTargetLabel.png)  
This looks pretty good.  

There are some possibilities of multicollinearity:
- GrLivArea and TotRmsAbvGround
- YearBuilt and GarageYrBlt
- 1stFlrSF and TotalBsmtSF
- GarageArea and GarageCars (Expected) 

In relation to the target label, overall quality seems to be the strongest correlating feature in the data set.

## Correlation Between Categorical Features
Same process as before, except this time I got dummy variables of all the categorical features and used spearman's method instead of pearson.  
Due to the dummies call, the dataframe has 209 columns. So I will do two heat maps, splitting the columns so it is somewhat readable

![image](Charts/CategoricalCorr_Col_0_100.png)  

![image](Charts/CategoricalCorr_Col_101_209.png)  

I am not going to write out the features that are important yet, too many to go through. Instead I will take all features I currently have and use XGBRegression to get feature importance.


