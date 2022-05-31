from acquire_nick import *
from prepare_nick import *

def wrangle_df(): 
    
    df = prep_data(acquire_data())
    
    return df