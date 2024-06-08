import pandas as pd
from util.VisualizeDataset import VisualizeDataset


def split_df_by_unique_dates(df, date_column):
    df['date_only'] = df[date_column].dt.date
    grouped = df.groupby('date_only')
    date_dfs = {str(date): group.drop(columns=['date_only']) for date, group in grouped}
    return date_dfs


df = pd.read_csv('data/kirsty/per_second_data.csv')
df['date_time'] = pd.to_datetime(df['date_time'])

DataViz = VisualizeDataset(__file__)

dfs = split_df_by_unique_dates(df, 'date_time')

for date, d in dfs.items():
    d.set_index('date_time', inplace=True)

    columns_to_plot = ['acceleration_', 'gyroscope_', 'linear_acceleration_', 'location_',
                       'magnetic_field_', 'pressure', 'proximity_', 'label_']

    non_empty_columns = []
    for col in columns_to_plot:
        filtered_cols = d.filter(regex=col)
        if not filtered_cols.empty and not filtered_cols.isnull().all().all():
            non_empty_columns.append(col)

    if non_empty_columns:
        DataViz.plot_dataset(d, non_empty_columns,
                             ['like' for _ in non_empty_columns],
                             ['line' if col != 'label_' else 'points' for col in non_empty_columns])
    else:
        print(f"No data to plot for date {date}")

