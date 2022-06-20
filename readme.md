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


## data dictionary (only features used in exploration):

Feature	Description
'airconditioningtypeid'         - Type of cooling system present in the home (if any)
'architecturalstyletypeid'      - Architectural style of the home (i.e. ranch, colonial, split-level, etc…)
'basementsqft'                  - Finished living area below or partially below ground level
'bathroomcnt'                   - Number of bathrooms in home including fractional bathrooms
'bedroomcnt'                    - Number of bedrooms in home 
'buildingqualitytypeid'         - Overall assessment of condition of the building from best (lowest) to worst (highest)
'buildingclasstypeid'           - The building framing type (steel frame, wood frame, concrete/brick) 
'calculatedbathnbr'             - Number of bathrooms in home including fractional bathroom
'decktypeid'                    - Type of deck (if any) present on parcel
'threequarterbathnbr'           - Number of 3/4 bathrooms in house (shower + sink + toilet)
'finishedfloor1squarefeet'      - Size of the finished living area on the first (entry) floor of the home
'calculatedfinishedsquarefeet'  - Calculated total finished living area of the home 
'finishedsquarefeet6'           - Base unfinished and finished area
'finishedsquarefeet12'          - Finished living area
'finishedsquarefeet13'          - Perimeter  living area
'finishedsquarefeet15'          - Total area
'finishedsquarefeet50'          - Size of the finished living area on the first (entry) floor of the home
'fips'                          - Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details
'fireplacecnt'	                - Number of fireplaces in a home (if any)
'fireplaceflag'	                - Is a fireplace present in this home 
'fullbathcnt'	                - Number of full bathrooms (sink, shower + bathtub, and toilet) present in home
'garagecarcnt'	                - Total number of garages on the lot including an attached garage
'garagetotalsqft'	            - Total number of square feet of all garages on lot including an attached garage
'hashottuborspa'	            - Does the home have a hot tub or spa
'heatingorsystemtypeid'	        - Type of home heating system
'latitude'	                    - Latitude of the middle of the parcel multiplied by 10e6
'longitude'	                    - Longitude of the middle of the parcel multiplied by 10e6
'lotsizesquarefeet'	            - Area of the lot in square feet
'numberofstories'	            - Number of stories or levels the home has
'parcelid'	                    - Unique identifier for parcels (lots) 
'poolcnt'	                    - Number of pools on the lot (if any)
'poolsizesum'	                - Total square footage of all pools on property
'pooltypeid10'	                - Spa or Hot Tub
'pooltypeid2'	                - Pool with Spa/Hot Tub
'pooltypeid7'	                - Pool without hot tub
'propertycountylandusecode'	    - County land use code i.e. it's zoning at the county level
'propertylandusetypeid'	        - Type of land use the property is zoned for
'propertyzoningdesc'	        - Description of the allowed land uses (zoning) for that property
'rawcensustractandblock'	    - Census tract and block ID combined - also contains blockgroup assignment by extension
'censustractandblock'	        - Census tract and block ID combined - also contains blockgroup assignment by extension
'regionidcounty'	            - County in which the property is located
'regionidcity'	                - City in which the property is located (if any)
'regionidzip'	                - Zip code in which the property is located
'regionidneighborhood'          - Neighborhood in which the property is located
'roomcnt'	                    - Total number of rooms in the principal residence
'storytypeid'	                - Type of floors in a multi-story house (i.e. basement and main level, split-level, attic, etc.).  See tab for details.
'typeconstructiontypeid'    	- What type of construction material was used to construct the home
'unitcnt'	                    - Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...)
'yardbuildingsqft17'	        - Patio in  yard
'yardbuildingsqft26'	        - Storage shed/building in yard
'yearbuilt'	                    - The Year the principal residence was built 
'taxvaluedollarcnt'	            - The total tax assessed value of the parcel
'structuretaxvaluedollarcnt'    - The assessed value of the built structure on the parcel
'landtaxvaluedollarcnt'	        - The assessed value of the land area of the parcel
'taxamount'	                    - The total property tax assessed for that assessment year
'assessmentyear'	            - The year of the property tax assessment 
'taxdelinquencyflag'        	- Property taxes for this parcel are past due as of 2015
'taxdelinquencyyear'	        - Year for which the unpaid propert taxes were due 


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
        df['yearbuilt'] = df['yearbuilt'].astype(int)
        df["bedroomcnt"] = df["bedroomcnt"].astype(int)
        df["calculatedfinishedsquarefeet"] = df["calculatedfinishedsquarefeet"].astype(int)
        df["fips"] = df["fips"].astype(int)
        df["lotsizesquarefeet"] = df["lotsizesquarefeet"].astype(int)
        df["rawcensustractandblock"] = df["rawcensustractandblock"].astype(int)
        df["regionidcity"] = df["regionidcity"].astype(int)
        df["regionidzip"] = df["regionidzip"].astype(int)
        df["censustractandblock"] = df["censustractandblock"].astype(int)
        df["structuretaxvaluedollarcnt"] = df["structuretaxvaluedollarcnt"].astype(int)
        df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int)
        df["landtaxvaluedollarcnt"] = df["landtaxvaluedollarcnt"].astype(int)
        df["taxamount"] = df["taxamount"].astype(int)
        df.yearbuilt = df.yearbuilt.astype(object) 
        df['age'] = df['age'].astype('int')
        df['transaction_month'] = df['transaction_month'].astype(int)
    - Added these features:
        df['age'] = 2017-df['yearbuilt']
        df['tax_rate'] = (df.taxamount/df.taxvaluedollarcnt) * 100
        df['transaction_month'] = df.transactiondate.str.split('-',expand=True)[1]
        df['county'] = np.where(df.fips == 6037, 'Los Angeles', np.where(df.fips == 6059, 'Orange','Ventura') )


- Explore
    - Performed univariate analysis on independent features and logerror.
    - Performed bivariate and multivariate exploration on several features to find recommendations that drive tax_value.
    - Further explored features using clustering methodologies.
    - Visualized features of logerror by using countplots and stackedplots and kdeplots.

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