# Zillow Data Science Team: What is driving the errors in the Zestimates?

## Overview

#### The purpose of this is to uncover what the drivers of Log Error in the Zestimate is using the 2017 properties and predictions data for single unit / single family homes.

## Goal

#### To improve our original estimate of the log error by using clustering methodologies.


## initial hypotheses:

#### 𝐻0: There is no significant difference between log error in June and all months.
#### 𝐻𝑎: Log Error in June is significantly lower than log error in all months.

#### 𝐻0: There is no linear correlation between calculatedfinishedsquarefeet and logerror.
#### 𝐻𝑎: There is a linear correlation between calculatedfinishedsquarefeet and logerror.

#### 𝐻0: There is no signficant difference between the logerror means of each cluster.
#### 𝐻𝑎: There is a signficant difference between the logerror means of at least two clusters.


## Entire Zillow Data Dictionary can be found here:

https://www.kaggle.com/competitions/zillow-prize-1/data?select=zillow_data_dictionary.xlsx


## project planning:
- Acquire
    - Acquired data using SQL from the zillow database.
        - Note: Functions to acquire data are built into the acquire.py file.
        - Loaded and inspected dataset.
        - prepared some of the features from import of SQL instead of in prepare steps.

- Prepare (can be viewed in prepare.py file)
    - Dropped and filled N/As and missing data with mean.
    - Only showed properties less than and equal to 6 bed/baths.
    - Removed properties where there are no baths and no beds.
    - Kept only properties less than 3000 square feet
    - Kept only properties valued less than $1m.
    - Removed unneccessary columns (check prepare.py for list)
    - Converted the following to int/obj
       - df['yearbuilt'] = df['yearbuilt'].astype(int)
       - df["bedroomcnt"] = df["bedroomcnt"].astype(int)
       - df["calculatedfinishedsquarefeet"] = df["calculatedfinishedsquarefeet"].astype(int)
       - df["fips"] = df["fips"].astype(int)
       - df["lotsizesquarefeet"] = df["lotsizesquarefeet"].astype(int)
       - df["rawcensustractandblock"] = df["rawcensustractandblock"].astype(int)
       - df["regionidcity"] = df["regionidcity"].astype(int)
       - df["regionidzip"] = df["regionidzip"].astype(int)
       - df["censustractandblock"] = df["censustractandblock"].astype(int)
       - df["structuretaxvaluedollarcnt"] = df["structuretaxvaluedollarcnt"].astype(int)
       - df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int)
       - df["landtaxvaluedollarcnt"] = df["landtaxvaluedollarcnt"].astype(int)
       - df["taxamount"] = df["taxamount"].astype(int)
       - df.yearbuilt = df.yearbuilt.astype(object) 
       - df['age'] = df['age'].astype('int')
       - df['transaction_month'] = df['transaction_month'].astype(int)
    - Added these features:
       - df['age'] = 2017-df['yearbuilt']
       - df['tax_rate'] = (df.taxamount/df.taxvaluedollarcnt) * 100
       - df['transaction_month'] = df.transactiondate.str.split('-',expand=True)[1]
       - df['county'] = np.where(df.fips == 6037, 'Los Angeles', np.where(df.fips == 6059, 'Orange','Ventura') )


- Explore
    - Performed univariate analysis on independent features and logerror.
    - Performed bivariate and multivariate exploration on several features in correlation to logerror.
    - Further explored features using clustering methodologies.
    - Visualized features of logerror using different types of charts.

- Model
    - Train, validated, and tested the predictors/independent features.
    - Determined my baseline prediciton.
    - Trained on classification models:
        - Logistic Regression
        - Polynomial
        - LassoLars

## instructions on how to reproduce this project and findings

- Download acquire.py module and use it to acquire the data. Requires credentials to access the zillow database.
- Download prepare.py module and use its functions to prepare the data.
- Explore on your own.

## Conclusion
#### key findings, recommendations, and takeaways on Exploring and Modeling

- Independent features explored did not have linear correlation
- Visualy can see non-linear correlation with transaction month
- No clusters had significant difference within their groupings
- Used Elbow test to select features, but some groupings still had very little observations - narrow down k
- Further testing with different features is needed to determine if there are significant clusters

- Polynomial Model outperformed on train, but did WAAAY worse on validate/test.
- Linear Regression did not outperform the baseline, but stayed more consistent between train and validate.
- LassoLars was even to baseline - will need to research why?

#### If I had more time:
- Further testing with different features is needed to determine if there are significant clusters.