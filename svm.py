import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 

df = pd.read_csv('data/data_fe.csv')

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

# fill na
df = df.fillna(method='ffill').fillna(method='bfill')

df = df.select_dtypes(include=[np.number])

# for col in df.columns:
#     print(col)

X = df.drop('activity', axis=1)
y = df['activity']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

svm = SVC()
sfs = SFS(
svm,
k_features = 28,
forward = True,
floating = False,
verbose= 2,
scoring= "f1_weighted",
n_jobs=-1)
sfs.fit(X_train, y_train)

selected_features = X_train.columns[sfs.get_support()]

svm.fit(X_train[selected_features], y_train)

y_val_pred = svm.predict(X_val[selected_features])
y_test_pred = svm.predict(X_test[selected_features])

train_accuracy = svm.score(X_train[selected_features], y_train)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Selected Features:", selected_features)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")