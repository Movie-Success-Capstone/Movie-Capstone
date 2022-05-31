from acquire_nick import *
from prepare_nick import *
import os 

def wrangle_df(use_cache=True): 
    
    if os.path.exists('clean.csv') and use_cache:
        print('Using cached CSV')
        return pd.read_csv('clean.csv')
    
    print('clean.csv not detected.')
    print('Acquiring and Preparing Data')
    
    df = prep_data(acquire_data())
    
    return df