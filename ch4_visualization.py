##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4 - Exemplary graphs                            #
#                                                            #
##############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from scipy.stats import norm
from sklearn.decomposition import PCA
from Chapter4.FrequencyAbstraction import FourierTransformation
import re
from eda import split_df_by_unique_dates

np.random.seed(0)

# Figure 4.1

# Sample frequency (Hz)
fs = 10

# Create time points....
df = pd.DataFrame(np.arange(0, 16.1, float(1)/fs), columns=list('X'))
c1 = 3 * np.sin(2 * math.pi * 0.2 * df['X'])
c2 = 2 * np.sin(2 * math.pi * 0.25 * (df['X']-2)) + 5
df['Y'] = c1 + c2

df = pd.read_csv('data/data_fe.csv')
df.filter(regex='gyro').columns
df['date_time'] = pd.to_datetime(df['date_time'])
dfs = split_df_by_unique_dates(df, 'date_time')
for date, d in dfs.items():
    print(f'{date}')

start_date = pd.to_datetime('2024-06-06')
end_date = pd.to_datetime('2024-06-08')

filtered_df = df[(df['date_time'] > start_date) & (df['date_time'] < end_date) & (df['participant'] == 'my')]
filtered_df.reset_index(inplace=True)

df_first_3 = df.iloc[:3] 


plt.plot(filtered_df.index, filtered_df['gyroscope_x'], 'b-')
plt.legend(['gyroscope_x'], loc=3, fontsize='small')
plt.xlabel('time')
plt.ylabel('$X_{1}$')
plt.show()

frequencies = []
values = []
for col in filtered_df.filter(regex='gyroscope_x').columns:

    val = re.findall(r'freq_\d+\.\d+_Hz', col)
    if len(val) > 0:
        frequency = float((val[0])[5:len(val)-4])
        frequencies.append(frequency)
        values.append(filtered_df.loc[filtered_df.index, col])

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlim([0, 1])
ax1.plot(frequencies, values, 'b+')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('$a$')
plt.show()

