import pandas as pd 
import os
import ast
import numpy as np

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
    if os.path.exists('dirty_df.csv') and use_cache:
        print('Using cached CSV')
        return pd.read_csv('dirty_df.csv')

    # When there's no cached csv, read the following query from Codeup's SQL database.
    print('CSV not detected.')
    print('Checking to s.')
    
    df = pd.read_csv('keywords.csv')
    df2 = pd.read_csv('credits.csv')
    #df3 = pd.read_csv('links.csv')
    df4 = pd.read_csv('movies_metadata.csv')
    #df5 = pd.read_csv('ratings.csv')
    
    # extract keywords from the nested dictionary.
    df['keywords'] = df['keywords'].apply(lambda x: ' '.join([i['name'] for i in eval(x)]))
    
    # extract cast members (actors and actressess) from the nested dictionary.
    df2['cast'] = df2['cast'].apply(lambda x: ' '.join([i['name'].replace(' ', '') for i in eval(x)]))
    # extract crew members (directors et cetera) from the nested dictionary.
    df2['crew'] = df2['crew'].apply(lambda x: ' '.join([i['name'].replace(' ', '') for i in eval(x)]))
    
    # drop the nas in order to permit the subsequent numeric conversions
    df4 = df4.dropna(subset=['title'])
     # convert from object to a numeric data type
    df4['popularity'] = pd.to_numeric(df4['popularity'])
    # convert from object to a numeric data type
    df4['budget'] = pd.to_numeric(df4['budget'])
    # convert from object to a numeric data type (not necessary, but makes some later operations easier)
    df4['id'] = pd.to_numeric(df4['id'])
    # extract production companies from the nested dictionary.
    df4['production_companies'] = df4['production_companies']\
    .apply(lambda x: ' '.join([i['name'].replace(' ', '') for i in eval(x)]))
    # extract production countries from the nested dictionary.
    df4['production_countries'] = df4['production_countries']\
    .apply(lambda x: ' '.join([i['name'].replace(' ', '') for i in eval(x)]))
    
    # presumed utility of each column worth merging (adjustable based on groups demands)
    data = pd.merge(df2, df4[['id', 'title', 'genres', 'budget', 'overview', 'popularity',
                              'production_companies', 'production_countries','revenue',
                              'runtime', 'vote_average', 'vote_count', 'release_date']],
                    left_on='id', right_on='id')
    # fixes instances of id duplication
    cnt = data.id.value_counts()
    v = cnt[cnt == 1].index.values
    test = data.query("id in @v")
    data = test.copy()
    # the only two columns are id and merge, so this is simply done on 'id' with no other parameters
    data = data.merge(df, on='id')
    # removes instances of tv-movies and re-releases, whereby nothing was recorded for revenue.
    # consequently removes many duplicate releases (parts of collections, etc) that were barely reviewed
    data = data[data['revenue'] >=1]
    # creates a csv 
    data.to_csv('dirty_df.csv')
    
    return data
    

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

def acquire_data_new(use_cache=True):
    """
    Note, this docstring needs to be properly written, 
    but right now it is most important to know that links.csv is hashed out because 
    to me it seems useless. Maybe there is some utility though, which we can explore later. 
    As for ratings.csv, it is excluded due to the problems it will cause in my recent
    attempts to merge it. This is, as such, not necessarily a complete acquire, but one that will
    get us through Tuesday. 
    """
    # If the cached parameter is True, read the csv file on disk in the same folder as this file 
    if os.path.exists('dirty_df.csv') and use_cache:
        print('Using cached CSV')
        return pd.read_csv('dirty_df.csv')

    # When there's no cached csv, read the following query from Codeup's SQL database.
    print('CSV not detected.')
    print('Checking to s.')
    
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
    df4['production_countries'] = df4['production_countries']\
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
    # creates a csv 
    data.to_csv('dirty_df.csv')
    
    return data