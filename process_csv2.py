import os
import pandas as pd
import re
from datetime import datetime, timedelta

# Define the base path to your data directory
base_path = 'data/kirsty'


def get_date_time(time_file_path):
    time_df = pd.read_csv(time_file_path)
    start_time_str = time_df.loc[time_df['event'] == 'START', 'system time'].iloc[0]
    end_time_str = time_df.loc[time_df['event'] == 'PAUSE', 'system time'].iloc[-1]  # Gets the last pause time
    start_time = datetime.fromtimestamp(start_time_str)
    end_time = datetime.fromtimestamp(end_time_str)
    return start_time, end_time


def process_csv(file_path, item):
    df = pd.read_csv(file_path)
    sensor_type = item.split('.')[0].lower()
    if sensor_type == 'location':
        df.columns = [
            'time' if col == 'Time (s)' else 'location_' + re.sub(r'\s*\([^)]*\)', '', col).strip().lower().replace(' ',
                                                                                                                    '_')
            for col in df.columns]
    elif sensor_type == 'proximity':
        df.columns = [
            'time' if col == 'Time (s)' else 'proximity_' + re.sub(r'\s*\([^)]*\)', '', col).strip().lower().replace(
                ' ', '_') for col in df.columns]
    else:
        df.columns = [
            'time' if col == 'Time (s)' else re.sub(r'\s*\([^)]*\)', '', col).strip().lower().replace(' ', '_') for col
            in df.columns]
    return df


def process_subdirectory(dir_path, category_label, start_time, end_time):
    combined_df = None
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path) and item_path.endswith('.csv'):
            df = process_csv(item_path, item)
            if combined_df is None:
                combined_df = df
            else:
                # Print debug information
                print("Merging DataFrames with columns:")
                print("Combined DataFrame columns:", combined_df.columns.tolist())
                print("Current DataFrame columns:", df.columns.tolist())

                combined_df = pd.merge(combined_df, df, on='time', how='outer', suffixes=('_left', '_right'))

                # Rename columns to ensure uniqueness
                combined_df.columns = pd.io.parsers.ParserBase({'names': combined_df.columns})._maybe_dedup_names(
                    combined_df.columns)

    # Add a column for the category label (one-hot encoding)
    combined_df[category_label] = 1

    combined_df['start_time'] = start_time
    combined_df['end_time'] = end_time
    combined_df['date_time'] = [start_time + timedelta(seconds=sec) for sec in combined_df['time']]

    return combined_df


def combine_all_data(base_path):
    all_data = pd.DataFrame()
    for category in os.listdir(base_path):  # 'bus', 'metro'
        category_path = os.path.join(base_path, category)

        if os.path.isdir(category_path):
            for sub_dir in os.listdir(category_path):
                sub_dir_path = os.path.join(category_path, sub_dir)
                time_file_path = os.path.join(sub_dir_path, 'meta', 'time.csv')

                if os.path.isfile(time_file_path):
                    start_time, end_time = get_date_time(time_file_path)
                    if os.path.isdir(sub_dir_path):
                        df = process_subdirectory(sub_dir_path, 'label_' + category, start_time, end_time)
                        if all_data.empty:
                            all_data = df
                        else:
                            # Ensure all expected columns are present in both DataFrames
                            missing_cols_all = set(all_data.columns) - set(df.columns)
                            for col in missing_cols_all:
                                df[col] = pd.NA
                            missing_cols_df = set(df.columns) - set(all_data.columns)
                            for col in missing_cols_df:
                                all_data[col] = pd.NA

                            # Print debug information
                            print("Concatenating DataFrames with columns:")
                            print("All DataFrame columns:", all_data.columns.tolist())
                            print("Current DataFrame columns:", df.columns.tolist())

                            all_data = pd.concat([all_data, df], ignore_index=True)

    # fill label column nans with 0
    label_cols = [col for col in all_data.columns if col.startswith('label_')]
    all_data[label_cols] = all_data[label_cols].fillna(0)
    return all_data


def aggregate_data_per_second(data_path):
    df = pd.read_csv(data_path)

    if not pd.api.types.is_datetime64_any_dtype(df['date_time']):
        df['date_time'] = pd.to_datetime(df['date_time'])

    df['date_time'] = df['date_time'].dt.floor('S')

    aggregated_df = df.groupby('date_time').mean()

    # Fill NaNs in category label columns which might occur during aggregation
    label_cols = [col for col in df.columns if col.startswith('label_')]
    for col in label_cols:
        if col in aggregated_df.columns:
            aggregated_df[col] = aggregated_df[col].fillna(0)

    # Reset index to keep 'date_time' as a column
    aggregated_df.reset_index(inplace=True)

    return aggregated_df


combined_data = combine_all_data(base_path)
combined_data.to_csv(f'{base_path}/combined_data.csv', index=False)

agg_data = aggregate_data_per_second(f'{base_path}/combined_data.csv')
agg_data.to_csv(f'{base_path}/per_second_data.csv', index=False)
