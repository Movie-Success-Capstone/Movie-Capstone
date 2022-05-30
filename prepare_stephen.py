import os 
import re
import pandas as pd
import numpy as np
import acquire_nick
from scipy import stats
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def get_dirty_data():
    '''
    This function calls acquire_nick.py module and acquires raw dataframe. 
    '''
    dirty_df = acquire_nick.acquire_data()
        
    return dirty_df

def data_summary(df):
    '''
    Function accepts dataframe and creates a summary statistics inclusing:
    - dataframe shape
    - df info
    - df describe 
    '''
    print('--- Shape: {}'.format(df.shape))
    print('--- Info')
    df.info()
    print('--- Column Descriptions')
    print(df.describe(include='all'))

def nulls_by_columns(df):
    '''
    Function return number & percent of nulls by column
    '''
    return pd.concat([
        df.isna().sum().rename('count'),
        df.isna().mean().rename('percent')
    ], axis=1)

def nulls_by_rows(df):
    '''
    Function return number & percent of nulls by rows
    '''
    return pd.concat([
        df.isna().sum(axis=1).rename('n_missing'),
        df.isna().mean(axis=1).rename('percent_missing'),
    ], axis=1).value_counts().sort_index()

def handle_missing_values(df, prop_required_column, prop_required_row):
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1, thresh=n_required_column)
    return df


def split_movies_data(df):
    '''
    Function splits dataframe into train, validate & test (56% : 24% : 20% respectively)
    NOTE--> Stratified on vote count
    '''
    
    train_validate, test = train_test_split(df, test_size = 0.2, 
                                            random_state = 123, 
#                                             stratify = df.vote_count
                                            )
    
    train, validate = train_test_split(train_validate, test_size = 0.3, 
                                       random_state = 123, 
#                                        stratity = train_validate.vote_count 
                                      )
    return train, validate, test
    
def wrangle_movies_data(df):
    '''
    Main Wrangle function. This function performs the following:
    - Replace any blank white spaces in the dataframe using regex function with nan
    - Shifts 'release_date' column to firs position
    - Shifts id column to the second position
    - Sets 'release_date' as datetime format
    - Drops 'Unnmed: 0' column from DF
    - Reset DF index 
    - Lowers all cases in entire DF
    - Splits DF into train, validate & test subsets
    - 
    - 
    '''
    # Set dataset range for movies after release date of year 2000-01-01 onwards
    df = df[df.release_date >= '2000-01-01']
    
    # Replace white space values with NaN values.
    df = df.replace(r'^\s*$', np.nan, regex = True)
    
    # shift 'id' column to first position
    first_column = df.pop('release_date')
    second_column = df.pop('id')

    # insert column using insert(position,column_name,first_column) function
    df.insert(0, 'release_date', first_column)
    df.insert(1, 'id', second_column)
    
    # Set 'release_date' from object data type to datetime format
    df.release_date = pd.to_datetime(df.release_date)
    
    # Drop unnecessary columns 
    df = df.drop(columns = 'Unnamed: 0')

    # Set 'release_date' as index & sort values
    df = df.set_index('release_date').sort_index()
    
    # Split words in columns ['cast', 'crew', 'production_companies', 'production_countries'] into regular englist expressions
    #-------------How?? IDK what delimeter to use-------------
    
    # Rename columns (NOT necessary)
    
    # Lower all data in dataframe
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)

    # Split the data
    train, validate, test = split_movies_data(df)
    
    return train, validate, test

    
# def get_modeling_data():
    
#     # Scale data
#     scaler = sklearn.preprocessing.MinMaxScaler()
#     # Note that we only call .fit with the training data,
#     # but we use .transform to apply the scaling to all the data splits.
#     scaler.fit(x_train)

#     x_train_scaled = scaler.transform(x_train)
#     x_validate_scaled = scaler.transform(x_validate)
#     x_test_scaled = scaler.transform(x_test)

#     plt.figure(figsize=(13, 6))
#     plt.subplot(121)
#     plt.hist(x_train, bins=25, ec ='black')
#     plt.title('Original')
#     plt.subplot(122)
#     plt.hist(x_train_scaled, bins=25, ec='black')
#     plt.title('Scaled')
      
#     return df
    
    
   

    
    
    
    