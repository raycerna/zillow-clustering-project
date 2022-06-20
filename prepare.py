from collections import Counter
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn import metrics
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from statsmodels.formula.api import ols
from math import sqrt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import median_absolute_error, r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt



#########################################################################################
def split_zillow_data(df):
    '''
   # This function performs split on zillow data.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)

    print('Train: %d rows, %d cols' % train.shape)
    print('Validate: %d rows, %d cols' % validate.shape)
    print('Test: %d rows, %d cols' % test.shape)
    return train, validate, test

##########################################################################################
def handle_missing_values(df, prop_required_column=0.5 , prop_required_row=0.5):
    '''
    This function takes in a pandas dataframe, default proportion of required columns (set to 50%)
    and proportion of required rows (set to 50%). It drops any rows or columns that contain null
    values more than the threshold specified from the original dataframe and returns that dataframe.
    Prior to returning that data, prints statistics and list counts/names of removed rows/cols 
    '''
    original_cols = df.columns.to_list()
    original_rows = df.shape[0]
    threshold = int(round(prop_required_column * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    remaining_cols = df.columns.to_list()
    remaining_rows = df.shape[0]
    dropped_col_count = len(original_cols) - len(remaining_cols)
    dropped_cols = list((Counter(original_cols) - Counter(remaining_cols)).elements())
    
    print(f'The following {dropped_col_count} columns were dropped because they were missing more than {prop_required_column * 100}% of data: \n{dropped_cols}\n')
    dropped_rows = original_rows - remaining_rows
    
    print(f'{dropped_rows} rows were dropped because they were missing more than {prop_required_row * 100}% of data')
          
    return df
############################################################################################
# combined in one function
def data_prep(df, cols_to_remove=[], prop_required_column=0.5, prop_required_row=0.5):
    '''
    This function calls the remove_columns and handle_missing_values
    to drop columns that need to be removed. It also drops rows and columns that have more 
    missing values than the specified threshold.
    '''
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df
############################################################################################
def remove_columns(df, cols_to_remove):
    '''
    This function takes in a pandas dataframe and a list of columns to remove.
    It drops those columns from the original df and returns the df.
    '''
    df = df.drop(columns=cols_to_remove)
    return df
############################################################################################

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe and return that dataframe'''  
    for col in col_list:
        # get quartiles
        q1, q3 = df[f'{col}'].quantile([.25, .75])  
        # calculate interquartile range
        iqr = q3 - q1   
        # get upper bound
        upper_bound = q3 + k * iqr 
        # get lower bound
        lower_bound = q1 - k * iqr   
        # return dataframe without outliers        
        df = df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]        
    return df
############################################################################################

def prep_zillow(df):
    '''
    Returns clean dataframe
    '''
    df = data_prep(df)    
    df = df [(df.propertylandusedesc == 'Single Family Residential')]
    # Only show properties less than and equal to 6 bed/baths 
    df = df[(df.bedroomcnt <= 6 ) & (df.bathroomcnt <= 6 )]    
    # Remove properties where there are no baths and no beds
    df = df[df.bathroomcnt > 0]  
    df = df[df.bedroomcnt > 0]  
    # keep only properties less than 3000 square feet
    df = df[df.calculatedfinishedsquarefeet <= 3000 ]    
    # keep only properties less than 1m.
    df = df[df.taxvaluedollarcnt <= 100000]
    # do not need any of finishedsquarefeet columns
    # removing these columns that are repeated and unnecessary
    df =df.drop(columns= ['finishedsquarefeet12', 'fullbathcnt', 'calculatedbathnbr',
                      'propertyzoningdesc', 'unitcnt', 'propertylandusedesc',
                      'assessmentyear', 'roomcnt', 'regionidcounty', 'propertylandusetypeid',
                      'heatingorsystemtypeid', 'id', 'heatingorsystemdesc', 'buildingqualitytypeid'],
            axis=1)    
    # The last nulls will be filled with mean 
    df = df.fillna(df.mean())
    # convert the following to int.
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
    df['tax_rate'] = (df.taxamount/df.taxvaluedollarcnt) * 100
    df.yearbuilt = df.yearbuilt.astype(object) 
    df['age'] = 2017-df['yearbuilt']
    df = df.drop(columns='yearbuilt')
    df['age'] = df['age'].astype('int')

    # add month feature
    df['transactiondate'] = df.transactiondate.astype('str')
    df['transaction_month'] = df.transactiondate.str.split('-',expand=True)[1]
    df['transaction_month'] = df['transaction_month'].astype(int)
    # add county names for fips feature
    df['county'] = np.where(df.fips == 6037, 'Los Angeles', np.where(df.fips == 6059, 'Orange','Ventura') )
    #df = df.drop(columns = ‘fips’)
    print('Features added were transaction_month extracted from transaction_date, County names instead of fips code, and converted yearbuilt to age in years. \n')   
    # Removing outliers 
    df = remove_outliers(df, 3, ['lotsizesquarefeet', 'structuretaxvaluedollarcnt','rawcensustractandblock']) 


    return df
    

############################################################################################   
    
def nulls_by_col(df):
    '''
    This function  takes in a dataframe of observations and attributes(or columns)
    and returns a dataframe where each row is an atttribute name, the first column is the 
    number of rows with missing values for that attribute, and the second column is percent
    of total rows that have missing values for that attribute.
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = (num_missing / rows * 100)
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 
                                 'percent_rows_missing': prcnt_miss})\
    .sort_values(by='percent_rows_missing', ascending=False)
    return cols_missing.applymap(lambda x: f"{x:0.1f}")
############################################################################################
def nulls_by_row(df):
    '''
    This function takes in a dataframe and returns a dataframe with 3 columns:
    the number of columns missing, percent of columns missing, 
    and number of rows with n columns missing.
    '''
    num_missing = df.isnull().sum(axis = 1)
    prcnt_miss = (num_missing / df.shape[1] * 100)
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 
                                 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index().set_index('num_cols_missing')\
    .sort_values(by='percent_cols_missing', ascending=False)
    return rows_missing
############################################################################################ 

def overview(df):
    print('--- Shape: {}'.format(df.shape))
    print('--- Info')
    df.info()
    print('--- Column Descriptions')
    print(df.describe(include='all'))
############################################################################################

def percentage_stacked_plot(columns_to_plot, title, prep_zillow):
    
    '''
    Returns a 100% stacked plot of the response variable for independent variable of the list columns_to_plot.
    Parameters: columns_to_plot (list of string): Names of the variables to plot
    '''
    
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot)/2)

    # create a figure
    fig = plt.figure(figsize=(12, 5 * number_of_rows)) 
    fig.suptitle(title, fontsize=22,  y=.95)
 

    # loop to each column name to create a subplot
    for index, column in enumerate(columns_to_plot, 1):

        # create the subplot
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)

        # calculate the percentage of observations of the response variable for each group of the independent variable
        # 100% stacked bar plot
        prop_by_independent = pd.crosstab(prep_zillow[column], prep_zillow['logerror']).apply(lambda x: x/x.sum()*100, axis=1)

        prop_by_independent.plot(kind='bar', ax=ax, stacked=True,
                                 rot=0, color=['#94bad4','#ebb086'])

        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5),
                  title='LogError', fancybox=True)

        # eliminate the frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)

    return percentage_stacked_plot

def scale_data(train, validate, test):

    train = train.drop(['logerror','county','transactiondate','propertycountylandusecode'], axis=1)
    validate = validate.drop(['logerror','county','transactiondate','propertycountylandusecode'], axis=1)
    test = test.drop(['logerror','county','transactiondate','propertycountylandusecode'], axis=1)

    # Create the Scaling Object
    scaler = sklearn.preprocessing.StandardScaler()

    # Fit to the train data only
    scaler.fit(train)

    # use the object on the whole df
    # this returns an array, so we convert to df in the same line
    train_scaled = pd.DataFrame(scaler.transform(train))
    validate_scaled = pd.DataFrame(scaler.transform(validate))
    test_scaled = pd.DataFrame(scaler.transform(test))

    # the result of changing an array to a df resets the index and columns
    # for each train, validate, and test, we change the index and columns back to original values

    # Train
    train_scaled.index = train.index
    train_scaled.columns = train.columns

    # Validate
    validate_scaled.index = validate.index
    validate_scaled.columns = validate.columns

    # Test
    test_scaled.index = test.index
    test_scaled.columns = test.columns

    return train_scaled, validate_scaled, test_scaled

#########################################################################

#function to plot parameter values in graph
def univariate(data,col,vartype=[0,1],hue =None):    
    '''
    Univariate function will plot parameter values in graphs.
    df      : dataframe name
    col     : Column name
    vartype : variable type : continuous or categorical
                Continuous(0)   : Distribution, Violin & Boxplot will be plotted.
                Categorical(1) : Countplot will be plotted.
    hue     : Only applicable in categorical analysis.
    '''
    sns.set(style="darkgrid")
    df = data.copy()
    if vartype == 0:
        fig, ax=plt.subplots(nrows =1,ncols=5,figsize=(20,6))
        #
        ax[0].set_title(col+" Distribution Plot")
        sns.distplot(df[col],ax=ax[0])

        ax[1].set_title(col+" Violin Plot")
        sns.violinplot(data =df, x=col,ax=ax[1], inner="quartile")#.set(ylabel='')
        #
        ax[2].set_title(col+" Box Plot")
        sns.boxplot(data =df, x=col,ax=ax[2],orient='v')
        #
        ax[3].set_title(col+" strip Plot")
        sns.stripplot(data =df, x=col,ax=ax[3])
        df[col]=np.log(df[col])
        ax[4].set_title(col+" scatter Plot")
        sns.scatterplot(x =df[col], y=df['logerror'],ax=ax[4])

        

    if vartype == 1:
        temp = pd.Series(data = hue)
        fig, ax = plt.subplots()
      
        width = len(df[col].unique()) + 3 + 2*len(temp.unique())
        fig.set_size_inches(width , 4)
        ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue) 
        if len(temp.unique()) > 0:
            for p in ax.patches:
                ax.annotate('{:1.1f}%'.format((p.get_height()*100)/float(len(df))), (p.get_x()+0.05, p.get_height()+10))  
        else:
            for p in ax.patches:
                ax.annotate(p.get_height(), (p.get_x()+0.16, p.get_height()+10)) 
        del temp
    else:
        exit
    fig.tight_layout()
    plt.show()

def create_clusters(train_scaled, validate_scaled, test_scaled):
    '''
    Function creates three clusters from scaled train - Tax, SQFT, Rooms
    Fits KMeans to train, predicts on train, validate, test to create clusters for each.
    Appends clusters to scaled data for modeling.
    '''

    # Tax Cluster
    # Selecting Features
    X_1 = train_scaled[['taxvaluedollarcnt', 'taxamount','tax_rate']]
    X_2 = validate_scaled[['taxvaluedollarcnt', 'taxamount','tax_rate']]
    X_3 = test_scaled[['taxvaluedollarcnt', 'taxamount','tax_rate']]
    # Creating Object
    kmeans = KMeans(n_clusters=4)
    # Fitting to Train Only
    kmeans.fit(X_1)
    # Predicting to add column to train
    train_scaled['cluster_tax'] = kmeans.predict(X_1)
    # Predicting to add column to validate
    validate_scaled['cluster_tax'] = kmeans.predict(X_2)
    # Predicting to add column to test
    test_scaled['cluster_tax'] = kmeans.predict(X_3)

    # SQFT Cluster
    # Selecting Features
    X_4 = train_scaled[['calculatedfinishedsquarefeet', 'lotsizesquarefeet']]
    X_5 = validate_scaled[['calculatedfinishedsquarefeet', 'lotsizesquarefeet']]
    X_6 = test_scaled[['calculatedfinishedsquarefeet', 'lotsizesquarefeet']]
    # Creating Object
    kmeans = KMeans(n_clusters=5)
    # Fitting to Train Only
    kmeans.fit(X_4)
    # Predicting to add column to train
    train_scaled['cluster_sqft'] = kmeans.predict(X_4)
    # Predicting to add column to validate
    validate_scaled['cluster_sqft'] = kmeans.predict(X_5)
    # Predicting to add column to test
    test_scaled['cluster_sqft'] = kmeans.predict(X_6)

    # Rooms Cluster
    # Selecting Features
    X_7 = train_scaled[['bedroomcnt', 'bathroomcnt']]
    X_8 = validate_scaled[['bedroomcnt', 'bathroomcnt']]
    X_9 = test_scaled[['bedroomcnt', 'bathroomcnt']]
    # Creating Object
    kmeans = KMeans(n_clusters=4)
    # Fitting to Train Only
    kmeans.fit(X_7)
    # Predicting to add column to train
    train_scaled['cluster_rooms'] = kmeans.predict(X_7)
    # Predicting to add column to validate
    validate_scaled['cluster_rooms'] = kmeans.predict(X_8)
    # Predicting to add column to test
    test_scaled['cluster_rooms'] = kmeans.predict(X_9)

    return train_scaled, validate_scaled, test_scaled
