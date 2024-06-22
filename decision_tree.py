import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt

def decision_tree(train_X, train_y, test_X, min_samples_leaf=100, max_depth=5, criterion='gini', print_model_details=False,
                  export_tree_path='./figures/decision_tree/', export_tree_name='tree.dot', gridsearch=False):
    if gridsearch:
        tuned_parameters = [{'min_samples_leaf': [1, 10, 50, 100, 500, 1000]}]
        dtree = GridSearchCV(DecisionTreeClassifier(max_depth=max_depth, criterion=criterion), tuned_parameters, cv=5, scoring='f1_weighted')
    else:
        dtree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth, criterion=criterion)

    dtree.fit(train_X, train_y)

    if gridsearch and print_model_details:
        print("Best parameters found:", dtree.best_params_)

    if gridsearch:
        dtree = dtree.best_estimator_

    pred_training_y = dtree.predict(train_X)
    pred_test_y = dtree.predict(test_X)

    if print_model_details:
        ordered_indices = [i[0] for i in sorted(enumerate(dtree.feature_importances_), key=lambda x: x[1], reverse=True)]
        print('Feature importance decision tree:')
        for i in range(len(dtree.feature_importances_)):
            print(f"{train_X.columns[ordered_indices[i]]} & {dtree.feature_importances_[ordered_indices[i]]}")

        if not os.path.exists(export_tree_path):
            os.makedirs(export_tree_path)
        export_graphviz(dtree, out_file=os.path.join(export_tree_path, export_tree_name),
                        feature_names=train_X.columns, class_names=dtree.classes_, filled=True)

    return dtree, pred_training_y, pred_test_y

if __name__ == "__main__":
    # Load the dataset
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

    # Print the feature names
    print("Features:", df.columns.tolist())

    X = df.drop('activity', axis=1)
    y = df['activity']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Feature selection
    decision_tree_classifier = DecisionTreeClassifier(min_samples_leaf=5, max_depth=5, random_state=42)
    sfs = SFS(
        decision_tree_classifier,
        k_features=28,
        forward=True,
        floating=False,
        verbose=2,
        scoring="f1_weighted",
        n_jobs=-1
    )
    sfs.fit(X_train, y_train)

    selected_features = X_train.columns[list(sfs.k_feature_idx_)]
    print("Selected Features:", selected_features)

    # Train the decision tree with the selected features and min_samples_leaf=100
    dtree, pred_train_y, pred_test_y = decision_tree(X_train[selected_features], y_train, X_test[selected_features], min_samples_leaf=500, max_depth=5)

    # Predictions
    pred_train_y = dtree.predict(X_train[selected_features])
    pred_val_y = dtree.predict(X_val[selected_features])
    pred_test_y = dtree.predict(X_test[selected_features])

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
    plt.plot([100], [train_f1_score], marker='o', label='Train F1 Score')
    plt.plot([100], [val_f1_score], marker='o', label='Validation F1 Score')
    plt.plot([100], [test_f1_score], marker='o', label='Test F1 Score')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('F1 Score')
    plt.title('F1 Score at Min Samples Leaf = 500')
    plt.legend()
    plt.grid(True)
    plt.show()
