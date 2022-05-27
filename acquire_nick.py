import pandas as pd 

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
    
    