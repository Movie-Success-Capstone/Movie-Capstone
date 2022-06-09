import os 
import pandas as pd
import acquire_nick

def get_dirty_data():
    '''
    This function calls acquire_nick.py module and acquires raw dataframe. 
    '''
    dirty_df = acquire_nick.acquire_data()
        
    return dirty_df

def data_summary(df):
    print('--- Shape: {}'.format(df.shape))
    print('--- Info')
    df.info()
    print('--- Column Descriptions')
    print(df.describe(include='all'))

def nulls_by_columns(df):
    return pd.concat([
        df.isna().sum().rename('count'),
        df.isna().mean().rename('percent')
    ], axis=1)

def nulls_by_rows(df):
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

def acquire():
    if os.path.exists('zillow.csv'):
        df = pd.read_csv('zillow.csv')
    else:
        database = 'zillow'
        url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{database}'
        df = pd.read_sql(query, url)
        df.to_csv('zillow.csv', index=False)
    return df

    
    