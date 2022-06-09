# Basics
import pandas as pd
import numpy as np
from scipy import math
from scipy import stats

# Visuals
import matplotlib.pyplot as plt
import seaborn as sns

# SKLearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Custom
from wrangle import*
from modeling import*
from pre_processing import*


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Make sure we can see the full scale of the data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Set default graph size
plt.rc('figure', figsize=(13.0, 6.0))
sns.set(rc = {'figure.figsize':(13,6.0)})

# Set dataframe display options
pd.options.display.max_colwidth = 25
pd.options.display.max_seq_items = 25