import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 
from sklearn.metrics import classification_report

def prep_df(partial=False):
    df = pd.read_csv('data/data_fe.csv')
    # selected_features = np.load('featureselection/svm_partial_dataset.npy')
    
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(by='date_time')
    
    # if partial:
    df = df[(df['date_time'] < "2024-06-06")]
    
    # drop useless columns
    unnamed = [col for col in df.columns if col.startswith('Unnamed')]
    df = df.drop(columns=unnamed)

    # combine labels into one column
    label_encoder = LabelEncoder()
    label_columns = [col for col in df.columns if col.startswith('label_')]
    df['activity'] = df[label_columns].idxmax(axis=1)
    df['activity'] = df['activity'].str.replace('label_', '')
    df['activity'] = label_encoder.fit_transform(df['activity'])
    df = df.drop(columns=label_columns)
    print(label_encoder.classes_)
    df['activity'].unique()
    # fill na
    df = df.fillna(method='ffill').fillna(method='bfill')

    df = df.select_dtypes(include=[np.number])
    return df

def feature_selection():
    df = prep_df(partial=True)
    selected_features = np.load('featureselection/svm_partial_dataset.npy', allow_pickle=True)

    X = df[['acceleration_y',
 'location_latitude',
 'location_longitude',
 'magnetic_field_x',
 'magnetic_field_y',
 'magnetic_field_z',
 'proximity_distance',
 'acceleration_x_pse',
 'acceleration_y_max_freq',
 'acceleration_z_max_freq',
 'acceleration_z_freq_1.0_Hz_ws_30',
 'linear_acceleration_x_pse',
 'linear_acceleration_y_pse',
 'magnetic_field_x_pse',
 'magnetic_field_x_freq_0.333_Hz_ws_30',
 'magnetic_field_x_freq_0.467_Hz_ws_30',
 'magnetic_field_x_freq_0.6_Hz_ws_30',
 'magnetic_field_x_freq_0.667_Hz_ws_30',
 'magnetic_field_x_freq_0.733_Hz_ws_30',
 'magnetic_field_x_freq_0.867_Hz_ws_30',
 'magnetic_field_y_pse',
 'magnetic_field_z_pse',
 'magnetic_field_z_freq_0.733_Hz_ws_30',
 'magnetic_field_z_freq_0.867_Hz_ws_30',
 'delta_latitude',
 'delta_longitude',
 'delta_velocity',
 'rate_delta_velocity']]
    # X = df.drop('activity', axis=1)
    y = df['activity']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4,shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    svm = SVC()
    sfs = SFS(
    svm,
    k_features = 10,
    forward = True,
    floating = False,
    verbose= 2,
    scoring= "f1_weighted",
    n_jobs=-1)
    sfs.fit(X_train, y_train)

    np.save('featureselection/svm_partial_dataset.npy', np.array(sfs.k_feature_names_))

if __name__ == '__main__':
    feature_selection()
    df = prep_df()
    X = df.drop('activity', axis=1)
    y = df['activity']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2,shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

    selected_features = np.load('featureselection/svm_partial_dataset.npy', allow_pickle=True)
    svm = SVC()
    svm.fit(X_train[selected_features], y_train)
    
    y_val_pred = svm.predict(X_val[selected_features])
    y_test_pred = svm.predict(X_test[selected_features])
    selected_features
    train_accuracy = svm.score(X_train[selected_features], y_train)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(classification_report(y_test, y_test_pred, target_names=['biking', 'bus', 'car', 'metro', 'sitting', 'train', 'tram', 'walking']))

    print("Selected Features:", selected_features)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")


