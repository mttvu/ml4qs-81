import numpy as np
import pandas as pd
import os
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.special import erfc
import scipy
warnings.simplefilter('ignore')
plt.style.use('seaborn')
for package in [pd, np, scipy, sns]:
    print(package.__name__, 'version:', package.__version__)

def chauvenet(array):
    mean = array.mean()           # Mean of incoming array
    stdv = array.std()            # Standard deviation
    N = len(array)                # Lenght of incoming array
    criterion = 1.0/(2*N)         # Chauvenet's criterion
    d = abs(array-mean)/stdv      # Distance of a value to mean in stdv's
    prob = erfc(d)                # Area normal dist.
    return prob < criterion       # Use boolean array outside this function

train_outliers = dict()
for col in [col for col in train.columns if 'var_' in col]:
    train_outliers[col] = train[chauvenet(train[col].values)].shape[0]
train_outliers = pd.Series(train_outliers)

train_outliers.sort_values().plot(figsize=(14, 40), kind='barh').set_xlabel('Number of outliers');