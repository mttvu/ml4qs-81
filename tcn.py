import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df = pd.read_csv('data/data_fe.csv')

# Drop useless columns
unnamed = [col for col in df.columns if col.startswith('Unnamed')]
df = df.drop(columns=unnamed)

# Combine labels into one column
label_encoder = LabelEncoder()
label_columns = [col for col in df.columns if col.startswith('label_')]
df['activity'] = df[label_columns].idxmax(axis=1)
df['activity'] = df['activity'].str.replace('label_', '')
df['activity'] = label_encoder.fit_transform(df['activity'])
df = df.drop(columns=label_columns)

# Fill NaN values
df = df.fillna(method='ffill').fillna(method='bfill')

# Select numerical features only
df = df.select_dtypes(include=[np.number])

# Standardize the features
scaler = StandardScaler()
X = df.drop('activity', axis=1)
y = df['activity']
X = scaler.fit_transform(X)

# Reshape data for TCN: (samples, time_steps, features)
time_steps = 10  # value needs to be chosen
num_samples = len(X) // time_steps
X = X[:num_samples * time_steps].reshape((num_samples, time_steps, -1))
y = np.array(y[:num_samples * time_steps]).reshape((num_samples, time_steps))[:, 0]

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the TCN model
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(time_steps, X.shape[2])))
model.add(Dropout(0.2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(len(np.unique(y)), activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Predictions
pred_train_y = model.predict(X_train)
pred_val_y = model.predict(X_val)
pred_test_y = model.predict(X_test)

pred_train_y = np.argmax(pred_train_y, axis=1)
pred_val_y = np.argmax(pred_val_y, axis=1)
pred_test_y = np.argmax(pred_test_y, axis=1)

# Evaluation
train_accuracy = accuracy_score(y_train, pred_train_y)
val_accuracy = accuracy_score(y_val, pred_val_y)
test_accuracy = accuracy_score(y_test, pred_test_y)

train_f1_score = f1_score(y_train, pred_train_y, average='weighted')
val_f1_score = f1_score(y_val, pred_val_y, average='weighted')
test_f1_score = f1_score(y_test, pred_test_y, average='weighted')

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

print(f"Train F1 Score: {train_f1_score}")
print(f"Validation F1 Score: {val_f1_score}")
print(f"Test F1 Score: {test_f1_score}")

print("Classification Report for the Best Model:\n")
print(classification_report(y_test, pred_test_y, target_names=label_encoder.classes_))

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
