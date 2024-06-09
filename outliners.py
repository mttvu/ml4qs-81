import numpy as np
import pandas as pd
import os
import multiprocessing
import seaborn as sns
import warnings
from scipy.special import erfc
import scipy
import matplotlib.pyplot as plt

def chauvenet(array):
    mean = array.mean()           # Mean of incoming array
    stdv = array.std()            # Standard deviation
    N = len(array)                # Lenght of incoming array
    criterion = 1.0/(2*N)         # Chauvenet's criterion
    d = abs(array-mean)/stdv      # Distance of a value to mean in stdv's
    prob = erfc(d)                # Area normal dist.
    return prob < criterion       # Use boolean array outside this function

train_outliers = dict()
train = pd.read_csv("./data/my/per_second_data.csv")
train.fillna(0, inplace=True)


for lable in [lable for lable in train.columns if 'label_'  in lable]:
    one_lable = train[train[lable]==1.0]
    for col in [col for col in train.columns if 'label_' not in col and 'time' not in col]:
        train_outliers[col] = one_lable[chauvenet(one_lable[col].values)].shape[0]
    train_outliers = pd.Series(train_outliers).sort_values()
    percentage_outliners = (train_outliers/one_lable.shape[0])*100
    print(train_outliers)
    print(one_lable.shape[0])
    plt.barh(range(len(train_outliers.to_dict())), list(train_outliers.to_dict().values()), align='center')
    plt.yticks(range(len(train_outliers.to_dict())), list(train_outliers.to_dict().keys()))
    plt.title("Outliners for lable "+ lable)
    plt.show()
    #train_outliers.sort_values().plot(kind='hist', title="My plot").show()
    #train_outliers.sort_values().plot(figsize=(14, 40), kind='barh').set_xlabel('Number of outliers')
    print('Total number of outliers in training set: {} ({:.2f}%)'.format(sum(train_outliers.values), np.max(percentage_outliners)))
    print("this is lable" + lable)

