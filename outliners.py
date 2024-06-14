import numpy as np
import pandas as pd
from scipy.special import erfc
import matplotlib.pyplot as plt

def chauvenet(array):
    mean = np.nanmean(array)
    stdv = np.nanstd(array)
    #mean = array.mean()           # Mean of incoming array
    #stdv = array.std()            # Standard deviation
    array[np.isnan(array)] = mean
    N = len(array)                # Lenght of incoming array
    criterion = 1.0/(2*N)         # Chauvenet's criterion
    d = abs(array-mean)/stdv      # Distance of a value to mean in stdv's
    prob = erfc(d)                # Area normal dist.
    return  prob < criterion, mean      # Use boolean array outside this function

def clean_outliners(train):
    for col in [col for col in train.columns if 'label_' not in col and 'time' not in col and 'participant' not in col]:
        tmp_array, mean = chauvenet(train[col].values)
        for i in range(tmp_array.size):
            if tmp_array[i]:
                print(i, col)
                train.loc[i, col] = np.nan
    return train


train_outliers = dict()
train = pd.read_csv("./data/all_data_per_second.csv")
#train.fillna(0, inplace=True)


# for lable in [lable for lable in train.columns if 'label_'  in lable]:
#     one_lable = train[train[lable]==1.0]
# for col in [col for col in train.columns if 'label_' not in col and 'time' not in col]:
#     train_outliers[col] = one_lable[chauvenet(one_lable[col].values)].shape[0]
# train_outliers = pd.Series(train_outliers).sort_values()
# percentage_outliners = (train_outliers/one_lable.shape[0])*100
# plt.barh(range(len(train_outliers.to_dict())), list(train_outliers.to_dict().values()), align='center')
# plt.yticks(range(len(train_outliers.to_dict())), list(train_outliers.to_dict().keys()))
# plt.title("Outliners for lable "+ lable)
# plt.show()
for col in [col for col in train.columns if 'label_' not in col and 'time' not in col and 'participant' not in col]:
    tmp_array, mean = chauvenet(train[col].values)
    train_outliers[col] = train[tmp_array].shape[0]
train_outliers = pd.Series(train_outliers).sort_values()
percentage_outliners = (train_outliers/train.shape[0])*100
plt.barh(range(len(train_outliers.to_dict())), list(train_outliers.to_dict().values()), align='center')
plt.yticks(range(len(train_outliers.to_dict())), list(train_outliers.to_dict().keys()))
plt.show()
print('Total number of outliers in training set: {} ({:.2f}%)'.format(sum(train_outliers.values), np.max(percentage_outliners)))
# display(train.head())
for col in [col for col in train.columns if 'label_' not in col and 'time' not in col and 'participant' not in col]:
    tmp_array, mean = chauvenet(train[col].values)
    for i in range(tmp_array.size):
        if tmp_array[i]:
            train.loc[i, col] = np.NaN
# for col in [col for col in train.columns if 'label_' not in col and 'time' not in col and 'participant' not in col]:
#     tmp_array, mean = chauvenet(train[col].values)
#     for i in range(tmp_array.size):
#         if tmp_array[i]:
#             train.loc[i,col] = np.nan
train.to_csv("./data/eliminated_outliners.csv")
#print("this is lable" + lable)

# train = clean_outliners(train)

# for lable in [lable for lable in train.columns if 'label_'  in lable]:
#     one_lable = train[train[lable]==1.0]
#     for col in [col for col in train.columns if 'label_' not in col and 'time' not in col]:
#         train_outliers[col] = one_lable[chauvenet(one_lable[col].values)].shape[0]
#     train_outliers = pd.Series(train_outliers).sort_values()
#     percentage_outliners = (train_outliers/one_lable.shape[0])*100
#     plt.barh(range(len(train_outliers.to_dict())), list(train_outliers.to_dict().values()), align='center')
#     plt.yticks(range(len(train_outliers.to_dict())), list(train_outliers.to_dict().keys()))
#     plt.title("Outliners for lable "+ lable)
#     plt.show()
#     #train_outliers.sort_values().plot(kind='hist', title="My plot").show(),
#     #train_outliers.sort_values().plot(figsize=(14, 40), kind='barh').set_xlabel('Number of outliers')
#     print('Total number of outliers in training set: {} ({:.2f}%)'.format(sum(train_outliers.values), np.max(percentage_outliners)))
#     print("this is lable" + lable)
