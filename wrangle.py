from acquire_nick import *
from prepare_nick import *
import os 
from sklearn.model_selection import train_test_split

def wrangle_df(use_cache=True): 
    
    if os.path.exists('clean.csv') and use_cache:
        print('Using cached CSV')
        return pd.read_csv('clean.csv', index_col='id')

    print('clean.csv not detected.')
    print('Acquiring and Preparing Data')
    
    df = prep_data(acquire_data())
    
    return df

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