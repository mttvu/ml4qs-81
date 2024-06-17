import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def get_integer_mapping(le):
    '''
    Return a dict mapping labels to their integer values
    from an SKlearn LabelEncoder
    le = a fitted SKlearn LabelEncoder
    '''
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})

    return res

df = pd.read_csv('data/data_fe.csv')
# drop useless columns
participant = [col for col in df.columns if col.startswith('participant')]
unnamed = [col for col in df.columns if col.startswith('Unnamed')]
df = df.drop(columns=unnamed)
df = df.drop(columns=participant)

# combine labels into one column
label_encoder = LabelEncoder()
label_columns = [col for col in df.columns if col.startswith('label_')]
df['activity'] = df[label_columns].idxmax(axis=1)

df['activity'] = df['activity'].str.replace('label_', '')

df['activity'] = label_encoder.fit_transform(df['activity'])
integerMapping = get_integer_mapping(label_encoder)
print(integerMapping)
df = df.drop(columns=label_columns)

# fill na
df = df.fillna(method='ffill').fillna(method='bfill')

df = df.select_dtypes(include=[np.number])

# for col in df.columns:
#     print(col)

X = df.drop('activity', axis=1)
y = df['activity']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
for i in range(5, 11):
    print(i)
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
