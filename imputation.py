import numpy as np
import pandas as pd
from util.VisualizeDataset import VisualizeDataset  # Adjust import as per your project structure
import matplotlib.pyplot as plt

# Function to split data by unique dates
def split_df_by_unique_dates(df, date_column):
    df['date_only'] = df[date_column].dt.date
    grouped = df.groupby('date_only')
    date_dfs = {str(date): group.drop(columns=['date_only']) for date, group in grouped}
    return date_dfs

# Define the imputation function
def impute_interpolate(dataset, col):
    dataset[col] = dataset[col].interpolate()
    dataset[col] = dataset[col].fillna(method='bfill')
    return dataset

# Load your dataset
df = pd.read_csv('./data/eliminated_outliners.csv')
df['date_time'] = pd.to_datetime(df['date_time'])

# Apply imputation to all columns in the dataset
for col in df.columns:
    df = impute_interpolate(df, col)

df.to_csv("./data/imputed.csv")

# Initialize an instance of VisualizeDataset
DataViz = VisualizeDataset(__file__)  # Adjust as per your project structure

# Split the dataset by unique dates
dfs = split_df_by_unique_dates(df, 'date_time')

# Iterate over each date's DataFrame and visualize
for date, d in dfs.items():
    d.set_index('date_time', inplace=True)

    # Visualize the dataset for the current date
    DataViz.plot_dataset(d, ['acceleration_', 'gyroscope_', 'linear_acceleration_', 'location_',
                             'magnetic_field_', 'pressure', 'proximity_', 'label_'],
                         ['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
                         ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])

    # plt.show()  # Show or save plots as needed
