
# IMPORTS
import pandas as pd
import numpy as np
import os 
import ast
from collections import Counter
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


#------------------------------------------------------------##------------------------------------------------------------#


def parse_cast(cast_string):
    # get a python data structure
    data = ast.literal_eval(cast_string)
    # sort by the "order" from the dataset
    data = sorted(data, key=lambda d: d['order'])
    top3 = [cast['name'] for cast in data[:3]]
    
    # add nulls where there's less than 3 cast members present
    if len(top3) < 3:
        top3 = top3 + [np.nan] * (3 - len(top3))

    return pd.Series(top3 + [len(data)], index=['cast_actor_1', 'cast_actor_2', 'cast_actor_3', 'total_n_cast'])


#------------------------------------------------------------##------------------------------------------------------------#


def acquire_data(use_cache=True):
    """
    Note, this docstring needs to be properly written, 
    but right now it is most important to know that links.csv is hashed out because 
    to me it seems useless. Maybe there is some utility though, which we can explore later. 
    As for ratings.csv, it is excluded due to the problems it will cause in my recent
    attempts to merge it. This is, as such, not necessarily a complete acquire, but one that will
    get us through Tuesday. 
    """
    # If the cached parameter is True, read the csv file on disk in the same folder as this file 
    if os.path.exists('capstone.csv') and use_cache:
        print('Using cached CSV')
        return pd.read_csv('capstone.csv')

    # When there's no cached csv, read the following query from Codeup's SQL database.
    print('Capstone CSV not detected.')
    print('Reading dirty CSVs: credits and movies_metadata')
    
    df = pd.read_csv('credits.csv')
    df2 = pd.read_csv('movies_metadata.csv')
    
    # set index as id
    df = df.set_index('id')
    # use UDF to get top 3 actors and the number of actors who appear 
    parsed_cast = df['cast'].apply(parse_cast)
    # merge those names with the original df
    df= df.merge(parsed_cast, on='id')
    # extract cast members (actors and actressess) from the nested dictionary.
    df['cast'] = df['cast'].apply(lambda x: ' '.join([i['name'].replace(' ', '') for i in eval(x)]))
    # extract crew members (directors et cetera) from the nested dictionary.
    df['crew'] = df['crew'].apply(lambda x: ' '.join([i['name'].replace(' ', '') for i in eval(x)]))
    
    # drop the nas in order to permit the subsequent numeric conversions
    df2 = df2.dropna(subset=['title'])
     # convert from object to a numeric data type
    df2['popularity'] = pd.to_numeric(df2['popularity'])
    # convert from object to a numeric data type
    df2['budget'] = pd.to_numeric(df2['budget'])
    # extract production companies from the nested dictionary.
    df2['production_companies'] = df2['production_companies']\
    .apply(lambda x: ' '.join([i['name'].replace(' ', '') for i in eval(x)]))
    # extract production countries from the nested dictionary.
    df2['production_countries'] = df2['production_countries']\
    .apply(lambda x: ' '.join([i['name'].replace(' ', '') for i in eval(x)]))
    # weird instances that make conversion impossible
    df2 = df2[df2['id'] != '1997-08-20']
    df2 = df2[df2['id'] != '2012-09-29']
    df2 = df2[df2['id'] != '2014-01-01']
    df2['id'] = df2['id'].astype(int)
    
    df_new = df2.merge(df,on='id')
    
    # fixes instances of id duplication
    cnt = df_new.id.value_counts()
    v = cnt[cnt == 1].index.values
    test = df_new.query("id in @v")
    data = test.copy()
    # removes instances of tv-movies and re-releases, whereby nothing was recorded for revenue.
    # consequently removes many duplicate releases (parts of collections, etc) that were barely reviewed
    data = data[data['revenue'] >=1]
    df = data.reset_index()
    # creates a csv 
    df.to_csv('capstone.csv')
    
    return df



#------------------------------------------------------------##------------------------------------------------------------#






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
    
    df['success_rating'] = (df['revenue']/(df['budget'] * 2)) * df['vote_average'] 
    df['success'] = df['success_rating'] > 6.5
    
    df['profit_amount'] = df.revenue - df.budget
    
    df = df[['title', 'success', 'success_rating', 'genres', 'cast_actor_1',
             'cast_actor_2', 'cast_actor_3', 'total_n_cast','budget', 'revenue',
             'profit_amount', 'id', 'vote_average', 'vote_count', 'production_companies',
             'production_countries','overview', 'popularity', 'runtime',
             'profitable', 'release_date', 'release_year', 'imdb_id']]
    
    df = df.set_index('id').sort_index()
    
    df.to_csv('clean.csv')
    print('clean.csv ready for future use')
    
    return df
    
    
    
    
    
#------------------------------------------------------------##------------------------------------------------------------#    
    
    
    
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



#------------------------------------------------------------##------------------------------------------------------------#




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



#------------------------------------------------------------##------------------------------------------------------------#



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
        
        
#------------------------------------------------------------##------------------------------------------------------------#



        
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



#------------------------------------------------------------##------------------------------------------------------------#


def wrangle_df(use_cache=True): 
    
    if os.path.exists('clean.csv') and use_cache:
        print('Using cached CSV')
        return pd.read_csv('clean.csv', index_col='id')

    print('clean.csv not detected.')
    print('Acquiring and Preparing Data')
    
    df = prep_data(acquire_data())
    
    return df



#------------------------------------------------------------##------------------------------------------------------------#



def train_validate_test_split(df):
    ''' 
    This function takes in a dataframe and splits it 80:20.  The 20% will be our testing datafrme for our final model.  The 80% will be split a second time (70:30), creating our final training dataframe and a dataframe to validate our model with before testing.  Leaving us we a Train (56%), Validate(24%) and Test (20%) Dataframe from our original data (100%)
    '''
    
    train, test = train_test_split(df, 
                               train_size = 0.8,
                               random_state=1313)
    
    
    train, validate = train_test_split(train,
                                  train_size = 0.7,
                                  random_state=1313)
    
    
    return train, validate, test



#------------------------------------------------------------##------------------------------------------------------------#




# other useful functions for metrics

def get_metrics(tp, fn, fp, tn):
    '''
    This function takes the True Positive, False Negative, False Positive, and True Negatives from a confusion matrix and uses them to give us the metrics of the model used for the matrix.
    '''
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = (2 * (precision * recall) / (precision + recall))
    true_pos = recall
    true_neg = tn / (tn + fp)
    false_pos = fp / (tn + fp)
    false_neg = fn / (tp + fn)
    support_pos = tp + fn
    support_neg = tn + fp

    print(f'Accuracy: {accuracy: .2%}')
    print(f'---------------')
    print(f'Recall: {recall: .2%}')
    print(f'---------------')
    print(f'Precision: {precision: .2%}')
    print(f'---------------')
    print(f'F1 Score: {f1_score: .2%}')
    print(f'---------------')
    print(f'True Positive Rate: {true_pos: .2%}')
    print(f'---------------')
    print(f'True Negative Rate: {true_neg: .2%}')
    print(f'---------------')
    print(f'False Positive Rate: {false_pos: .2%}')
    print(f'---------------')
    print(f'False Negative Rate: {false_neg: .2%}')
    print(f'---------------')
    print(f'Support (Did Not Survive(0)): {support_pos}')
    print(f'---------------')
    print(f'Support (Survived(1)): {support_neg}')