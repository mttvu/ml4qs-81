import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

df = pd.read_csv('data/data_fe.csv')
unnamed = [col for col in df.columns if col.startswith('Unnamed')]
df = df.drop(columns=unnamed)
df.sort_values(by=['participant_encoded', 'date_time'], inplace=True)

label_columns = [col for col in df.columns if col.startswith('label_')]
df['activity'] = df[label_columns].idxmax(axis=1)
df['activity'] = df['activity'].str.replace('label_', '')
label_encoder = LabelEncoder()
df['activity'] = label_encoder.fit_transform(df['activity'])

df = df.fillna(method='ffill').fillna(method='bfill')
df['date_time'] = pd.to_datetime(df['date_time'])

sequence_length = 10
cutoff_date = pd.Timestamp('2024-06-08')  

# Split data by date
train_df = df[df['date_time'] < cutoff_date]
test_df = df[df['date_time'] >= cutoff_date]

train_df = train_df.select_dtypes(include=[np.number])
test_df = test_df.select_dtypes(include=[np.number])

# create sequences from dataframes
def create_sequences(df, sequence_length):
    X_list, y_list = [], []
    # Group data by participant and create sequences
    for _, group in df.groupby('participant_encoded'):
        data = group.drop(columns=['activity']).values
        labels = group['activity'].values
        for i in range(len(data) - sequence_length + 1):
            X_list.append(data[i:i + sequence_length])
            y_list.append(labels[i + sequence_length - 1])
    return np.array(X_list), to_categorical(LabelEncoder().fit_transform(np.array(y_list)))

X_train, y_train = create_sequences(train_df, sequence_length)
X_test, y_test = create_sequences(test_df, sequence_length)

model = Sequential([
    LSTM(50, input_shape=(sequence_length, X_train.shape[2])),
    Dropout(0.2),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))