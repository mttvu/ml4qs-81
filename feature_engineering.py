import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation

def extract_date_time_info(df):
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['weekday'] = df['date_time'].dt.weekday
    df['hour'] = df['date_time'].dt.hour
    df['minute'] = df['date_time'].dt.minute
    return df

# for example my -> 1, kirsty -> 2, keye -> 3
def participant_label_encoding(df):
    label_encoder = LabelEncoder()
    df['participant_encoded'] = label_encoder.fit_transform(df['participant'])
    return df

def frequency_domain_transformation(df):
    sensor_columns = [
    'acceleration_x', 'acceleration_y', 'acceleration_z',
    'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
    'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
    'magnetic_field_x','magnetic_field_y','magnetic_field_z']

    df = FourierTransformation().abstract_frequency(data_table=df, columns=sensor_columns, window_size=30, sampling_rate=2)
    
    return df

def location_transformation(df, window_size):
    # difference in consecutive rows
    df['delta_latitude'] = df['location_latitude'].diff()
    df['delta_longitude'] = df['location_longitude'].diff()
    df['delta_height'] = df['location_height'].diff()
    df['delta_velocity'] = df['location_velocity'].diff()
    df['delta_direction'] = df['location_direction'].diff().abs()

    df['time_diff'] = df['date_time'].diff().dt.total_seconds()  # Time difference in seconds
    df['rate_delta_velocity'] = df['delta_velocity'] / df['time_diff']
    df['rate_delta_height'] = df['delta_height'] / df['time_diff']

    df['rolling_cumulative_distance'] = df.apply(
        lambda row: np.sqrt((df.loc[:row.name, 'delta_latitude']**2 + 
                            df.loc[:row.name, 'delta_longitude']**2).sum()), axis=1).rolling(window=window_size).sum()

    df['rolling_cumulative_elevation'] = df['delta_height'].rolling(window=window_size).sum().fillna(0)
    df['rolling_avg_velocity'] = df['location_velocity'].rolling(window=window_size).mean()
    df['rolling_avg_direction'] = df['location_direction'].rolling(window=window_size).mean()

    return df

def time_domain_transformation(df, window_size):
    sensor_columns = [
    'acceleration_x', 'acceleration_y', 'acceleration_z',
    'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
    'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
    'magnetic_field_x','magnetic_field_y','magnetic_field_z', 'pressure']
    
    NumericalAbstraction().abstract_numerical(df, sensor_columns, window_size, 'mean')
    NumericalAbstraction().abstract_numerical(df, sensor_columns, window_size, 'std')

    return df

def standardize(df):
    exclude_cols = ['index','participant', 'date_time'] + [col for col in df.columns if col.startswith('label_')]
    cols_to_standardize = df.columns.difference(exclude_cols)
    scaler = StandardScaler()
    df[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])
    return df



df = pd.read_csv('data/outliners_eliminated.csv')
df = df.drop(['index','Unnamed: 0','time'], errors='ignore', axis=1)
df = standardize(df)
df = extract_date_time_info(df)
df = participant_label_encoding(df)
df = frequency_domain_transformation(df)
df = location_transformation(df, 10)
# df = time_domain_transformation(df, 10)


df.to_csv('data/data_fe.csv')

