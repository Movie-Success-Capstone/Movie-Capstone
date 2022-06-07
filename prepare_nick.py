#      ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
#     ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
#     ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ 
#     ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          
#     ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ 
#     ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
#     ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀█░█▀▀ ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀█░█▀▀ ▐░█▀▀▀▀▀▀▀▀▀ 
#     ▐░▌          ▐░▌     ▐░▌  ▐░▌          ▐░▌          ▐░▌       ▐░▌▐░▌     ▐░▌  ▐░▌          
#     ▐░▌          ▐░▌      ▐░▌ ▐░█▄▄▄▄▄▄▄▄▄ ▐░▌          ▐░▌       ▐░▌▐░▌      ▐░▌ ▐░█▄▄▄▄▄▄▄▄▄ 
#     ▐░▌          ▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌
#      ▀            ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀            ▀         ▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀ 
#                                                                                                
#----------------------------------------------------------------------------------------------------------|
import pandas as pd 
import numpy as np
from collections import Counter
from datetime import date
import os
#----------------------------------------------------------------------------------------------------------|
def prep_data(df, use_cache=True):
    # If the cached parameter is True, read the csv file on disk in the same folder as this file 
    if os.path.exists('clean.csv') and use_cache:
        print('clean.csv detected. \n Dataframe available.')
        return pd.read_csv('clean.csv')

    # When there's no cached csv, read the following query from Codeup's SQL database.
    print('clean.csv not detected.')
    print('processing capstone.csv')
    
    # drop columns that do not inform our decision-making process
    df.drop(columns=['adult', 'belongs_to_collection',
                 'homepage', 'original_language', 'original_title',
                 'poster_path', 'spoken_languages', 'status', 'tagline', 'video', 'cast',
                 'crew'], inplace=True)
    # drop all na values (sought other method, but it wasn't more effective
    df.dropna(axis=0, inplace=True)
    # by first sorting df descending from budget, it will keep more valuable values in next step
    df = df.sort_values(by='budget', ascending=False, na_position='last')
    # keep only the first instances of duplicates, now that they are sorted. 
    df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)
    df['genres'] = df['genres'].apply(lambda x: ' '.join([i['name'] for i in eval(x)]))
    # fills values less than one million with the median value, 10000000
    df['budget'] = np.where(df['budget'].between(0,1000000), df['budget'].median(), df['budget'])
    # extract the release year before converting to date-time
    df['release_year'] = df.release_date.apply(lambda x : x.split('-')[0]).astype(int)
    # convert to a datetime format. Consider to_datetime as well. 
    df['release_date'] = df['release_date'].astype('datetime64[ns]')
    # movie is successful if (revenue >= budget * 2)
    for index, row in df.iterrows():
        try:
            budget = df.at[index, 'budget']
            revenue = df.at[index, 'revenue']
            if (revenue >= budget * 2):
                profitable = True
            else:
                profitable = False
        except:     # if budget or revenue is empty
            profitable = np.nan
    
        df.at[index, 'profitable'] = profitable
    # convert from object to bool. May set to binary later.  
    df['profitable'] = df['profitable'].astype(bool)
    # convert profitable to int, 0 = false, 1 = true
    df['profitable'] = df['profitable']*1
    
    df['success_rating'] = (df['revenue']/(df['budget'] * 2)) * df['vote_average'] 
    df['success'] = df['success_rating'] > 6.5
    # convert success to int, 0 = false, 1 = true
    df['success'] = df['success']*1
    
    df['profit_amount'] = df.revenue - df.budget
    
    df = df[['title', 'success', 'success_rating', 'genres', 'cast_actor_1',
             'cast_actor_2', 'cast_actor_3', 'total_n_cast','budget', 'revenue',
             'profit_amount', 'id', 'vote_average', 'vote_count', 'production_companies',
             'production_countries','overview', 'popularity', 'runtime',
             'profitable', 'release_date', 'release_year', 'imdb_id']]
    
    # extract first production company
    df['production_company'] = df['production_companies'].str.split().str.get(0)
    # One Hot Encode for Genres
    df['is_genre_adventure'] = df.genres.apply(lambda genre_list: 'Adventure' in genre_list) * 1
    df['is_genre_horror'] = df.genres.apply(lambda genre_list: 'Horror' in genre_list) * 1
    df['is_genre_drama'] = df.genres.apply(lambda genre_list: 'Drama' in genre_list) * 1
    df['is_genre_scifi'] = df.genres.apply(lambda genre_list: 'Science' in genre_list) * 1
    df['is_genre_romance'] = df.genres.apply(lambda genre_list: 'Romance' in genre_list) * 1
    df['is_genre_thriller'] = df.genres.apply(lambda genre_list: 'Thriller' in genre_list) * 1
    df['is_genre_crime'] = df.genres.apply(lambda genre_list: 'Crime' in genre_list) * 1
    df['is_genre_comedy'] = df.genres.apply(lambda genre_list: 'Comedy' in genre_list) * 1
    df['is_genre_animation'] = df.genres.apply(lambda genre_list: 'Animation' in genre_list) * 1
    df['is_genre_action'] = df.genres.apply(lambda genre_list: 'Action' in genre_list) * 1
    df['is_genre_mystery'] = df.genres.apply(lambda genre_list: 'Mystery' in genre_list) * 1
    df['is_genre_fantasy'] = df.genres.apply(lambda genre_list: 'Fantasy' in genre_list) * 1
    df['is_genre_documentary'] = df.genres.apply(lambda genre_list: 'Documentary' in genre_list) * 1
    
    df['release_year'] = df.release_date.dt.year
    df['release_month'] = df.release_date.dt.month
    df['release_day'] = df.release_date.dt.day
    df['release_weekday'] = df.release_date.dt.day_name()
    df['is_long_movie'] = df.runtime.transform(lambda x: int(x > 120))
    df['ROI'] = df.revenue / df.budget
    df['returns'] = pd.qcut(df.ROI, 4, labels=['low', 'avg', 'high', 'very high'])
    df['budget_range'] = pd.qcut(df.budget, 5, labels=['low', 'avg', 'high', 'very high'], duplicates='drop')
    
    # somewhere along the way, nulls were introduced
    # here are three different ways to handle them here.
    df = df[~df['genres'].isnull()]
    df = df[df['production_countries'].notnull()]
    df = df[~df['production_company'].isna()]
    
    df = df.set_index('id').sort_index()
    
    df.to_csv('clean.csv')
    print('clean.csv ready for future use')
    
    return df
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|    
def nulls_by_col(df):
    '''
    This function  takes in a dataframe of observations and attributes(or columns) and returns a dataframe where each row is an atttribute name, the first column is the 
    number of rows with missing values for that attribute, and the second column is percent of total rows that have missing values for that attribute.
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = (num_missing / rows * 100)
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 
                                 'percent_rows_missing': prcnt_miss})\
    .sort_values(by='percent_rows_missing', ascending=False)
    return cols_missing.applymap(lambda x: f"{x:0.1f}")
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|
def nulls_by_row(df):
    '''
    This function takes in a dataframe and returns a dataframe with 3 columns: the number of columns missing, percent of columns missing, 
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
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|
def describe_data(df):
    '''
    This function takes in a pandas dataframe and prints out the shape, datatypes, number of missing values, 
    columns and their data types, summary statistics of numeric columns in the dataframe, as well as the value counts for categorical variables.
    '''
    # Print out the "shape" of our dataframe - rows and columns
    print(f'This dataframe has {df.shape[0]} rows and {df.shape[1]} columns.')
    print('')
    print('--------------------------------------')
    print('--------------------------------------')
    
    # print the datatypes and column names with non-null counts
    print(df.info())
    print('')
    print('--------------------------------------')
    print('--------------------------------------')
    
    
    # print out summary stats for our dataset
    print('Here are the summary statistics of our dataset')
    print(df.describe().applymap(lambda x: f"{x:0.3f}"))
    print('')
    print('--------------------------------------')
    print('--------------------------------------')

    # print the number of missing values per column and the total
    print('Null Values by Column: ')
    missing_total = df.isnull().sum().sum()
    missing_count = df.isnull().sum() # the count of missing values
    value_count = df.isnull().count() # the count of all values
    missing_percentage = round(missing_count / value_count * 100, 2) # percentage of missing values
    missing_df = pd.DataFrame({'count': missing_count, 'percentage': missing_percentage})\
    .sort_values(by='percentage', ascending=False)
    
    print(missing_df.head(50))
    print(f' \n Total Number of Missing Values: {missing_total} \n')
    df_total = df.shape[0] * df.shape[1]
    proportion_of_nulls = round((missing_total / df_total), 4)
    print(f' Proportion of Nulls in Dataframe: {proportion_of_nulls}\n') 
    print('--------------------------------------')
    print('--------------------------------------')
    
    print('Row-by-Row Nulls')
    print(nulls_by_row(df))
    print('----------------------')
    
    print('Relative Frequencies: \n')
    ## Display top 5 values of each variable within reasonable limit
    limit = 25
    for col in df.columns:
        if df[col].nunique() < limit:
            print(f'Column: {col} \n {round(df[col].value_counts(normalize=True).nlargest(5), 3)} \n')
        else: 
            print(f'Column: {col} \n')
            print(f'Range of Values: [{df[col].min()} - {df[col].max()}] \n')
        print('------------------------------------------')
        print('--------------------------------------')
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|
def nulls(df):
    '''
    This function takes in a pandas dataframe and prints out the shape, datatypes, number of missing values, 
    columns and their data types, summary statistics of numeric columns in the dataframe, as well as the value counts for categorical variables.
    '''
    # print the number of missing values per column and the total
    print('Null Values by Column: ')
    missing_total = df.isnull().sum().sum()
    missing_count = df.isnull().sum() # the count of missing values
    value_count = df.isnull().count() # the count of all values
    missing_percentage = round(missing_count / value_count * 100, 2) # percentage of missing values
    missing_df = pd.DataFrame({'count': missing_count, 'percentage': missing_percentage})\
    .sort_values(by='percentage', ascending=False)
    
    print(missing_df.head(50))
    print(f' \n Total Number of Missing Values: {missing_total} \n')
    df_total = df.shape[0] * df.shape[1]
    proportion_of_nulls = round((missing_total / df_total), 4)
    print(f' Proportion of Nulls in Dataframe: {proportion_of_nulls}\n') 
    print('--------------------------------------')
    print('--------------------------------------')
    
    print('Row-by-Row Nulls')
    print(nulls_by_row(df))
    print('----------------------')
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|
#----------------------------------------------------------------------------------------------------------|
