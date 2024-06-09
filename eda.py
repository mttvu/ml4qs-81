from util.VisualizeDataset import VisualizeDataset
import pandas as pd

def split_df_by_unique_dates(df, date_column):
    df['date_only'] = df[date_column].dt.date
    grouped = df.groupby('date_only')
    date_dfs = {str(date): group.drop(columns=['date_only']) for date, group in grouped}

    return date_dfs

df = pd.read_csv('data/my/per_second_data.csv')
# df = pd.read_csv('data/kirsty/per_second_data.csv')
df['date_time'] = pd.to_datetime(df['date_time'])

DataViz = VisualizeDataset(__file__)

dfs = split_df_by_unique_dates(df, 'date_time')

for date, d in dfs.items():
    d.set_index('date_time', inplace=True)
    DataViz.plot_dataset(d, ['acceleration_', 'gyroscope_', 'linear_acceleration_', 'location_', 
                          'magnetic_field_', 'pressure', 'proximity_', 'label_'],
                              ['like', 'like', 'like', 'like', 'like', 'like', 'like','like'],
                              ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])