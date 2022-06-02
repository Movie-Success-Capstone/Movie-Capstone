import wrangle as w 
import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def create_modeling_df():
    df = w.wrangle_df()
    # obtain ten most frequently occuring companies
    threshold1 = 101
    df.loc[df['production_company'].value_counts()\
           [df['production_company']].values < threshold1, 'production_company'] = "other_company"
    # obtain ten actors who appear the most 
    threshold2 = 26
    df.loc[df['cast_actor_1'].value_counts()\
           [df['cast_actor_1']].values < threshold2, 'cast_actor_1'] = "other_actor"
    # create dummies based on those newly created columns
    dummy_group = ['cast_actor_1', 'production_company']
    dummy_df = pd.get_dummies(df.loc[:,dummy_group])
    # subset the data frame. Will retain even less columns after feature selection
    keep =  ['budget','runtime', 'vote_average','vote_count', 'success', 
         'release_year', 'is_genre_adventure', 'is_genre_horror', 
         'is_genre_drama', 'is_genre_scifi', 'is_genre_romance',
         'is_genre_thriller', 'is_genre_crime', 'is_genre_comedy',
         'is_genre_animation', 'is_genre_action', 'is_genre_mystery',
         'is_genre_fantasy', 'is_genre_documentary', 'total_n_cast']
    modeling_df = df.loc[:,keep]
    modeling_df = pd.concat([modeling_df, dummy_df], axis=1)
    print('the shape of this modeling df should be (6893, 46)')
    print(f'the current shape is {modeling_df.shape}')
    print('please split and then scale this dataframe')
    return modeling_df
          
def split_and_scale(modeling_df):
    train, validate, test = w.train_validate_test_split(modeling_df)
    X_train = train.drop(columns=['success'])
    y_train = train['success']
    
    X_validate = validate.drop(columns=['success'])
    y_validate = validate['success']
    
    X_test = test.drop(columns=['success'])
    y_test = test['success']
    
    scaler = MinMaxScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns = X_train.columns)
    X_validate = pd.DataFrame(X_validate_scaled, index=X_validate.index, columns = X_validate.columns)
    X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns = X_test.columns)
    
    X_train['baseline_prediction'] = 0
    X_validate['baseline_prediction'] = 0
    X_test['baseline_prediction'] = 0
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test
    
    
        
    
    