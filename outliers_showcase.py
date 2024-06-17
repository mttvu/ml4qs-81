import numpy as np
import pandas as pd
from scipy.special import erfc
import matplotlib.pyplot as plt

# Define the Chauvenet's criterion function
def chauvenet(array):
    mean = np.nanmean(array)
    stdv = np.nanstd(array)
    array[np.isnan(array)] = mean
    N = len(array)
    criterion = 1.0 / (2 * N)
    d = abs(array - mean) / stdv
    prob = erfc(d)
    return prob < criterion, mean

# Function to plot data points and highlight outliers in subplots
def plot_outliers_in_subplots(label_data, label_col, feature_groups):
    for feature_type, features in feature_groups.items():
        num_features = len(features)
        fig, axs = plt.subplots(num_features, 1, figsize=(10, 6 * num_features))

        if num_features == 1:
            axs = [axs]  # Ensure axs is iterable if there's only one subplot

        for i, col in enumerate(features):
            data = label_data[col].values
            outliers, mean = chauvenet(data)

            axs[i].plot(data, 'b.', label='Data Points')
            axs[i].plot(np.where(outliers)[0], data[outliers], 'ro', label='Outliers')
            axs[i].set_title(f'Outliers in {col} for {label_col} According to Chauvenet\'s Criterion')
            axs[i].set_xlabel('Index')
            axs[i].set_ylabel(col)
            axs[i].legend()

        plt.suptitle(f'{feature_type.capitalize()} Features for {label_col}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# Load the data
train = pd.read_csv("./data/all_data_per_second.csv")

# Identify label columns
label_columns = [col for col in train.columns if 'label_' in col]

# Group features by their type
feature_groups = {}
for col in train.columns:
    if col not in label_columns and 'time' not in col and 'participant' not in col:
        feature_type = col.split('_')[0]
        if feature_type not in feature_groups:
            feature_groups[feature_type] = []
        feature_groups[feature_type].append(col)

outlier_info = {}

# Process each label
for label in [col for col in train.columns if 'label_' in col]:
    label_data = train[train[label] == 1.0]
    print(f"Processing label: {label}")
    # plot_outliers_in_subplots(label_data, label, feature_groups)      # uncomment to show plots (it's a lot)
    total_outliers = 0
    total_data_points = 0

    # Process each feature column
    for col in [col for col in train.columns if 'label_' not in col and 'time' not in col and 'participant' not in col]:
        data = label_data[col].values
        outliers, mean = chauvenet(data)
        total_outliers += np.sum(outliers)
        total_data_points += len(outliers)

    # Calculate total percentage of outliers for the label
    total_percentage_outliers = (total_outliers / total_data_points) * 100

    # Store results in the dictionary
    outlier_info[label] = {
        'percentage': total_percentage_outliers,
        'count': total_outliers
    }

# Print the results
for label, info in outlier_info.items():
    print(f"Label: {label}")
    print(f"    Total Percentage of Outliers: {info['percentage']:.2f}%")
    print(f"    Number of Outliers: {info['count']}")
    print()