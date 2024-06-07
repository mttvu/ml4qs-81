import os
import pandas as pd

# Define the base path to your data directory
base_path = 'data/my'

# A helper function to process each CSV file
def process_csv(file_path, prefix):
    df = pd.read_csv(file_path)
    # Standardize the time column to 'Time (s)'
    if 'Time (s)' not in df.columns:
        # Assuming the time column is always the first column
        df.rename(columns={df.columns[0]: 'Time (s)'}, inplace=True)
    # Rename other columns to include the file prefix
    df.columns = ['Time (s)' if col == 'Time (s)' else f"{prefix}_{col}" for col in df.columns]
    return df

# A function to process each sub-directory (e.g., 'bus 4-6')
def process_subdirectory(dir_path, category_label):
    combined_df = None
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path) and item_path.endswith('.csv'):
            # Extract the sensor type from the file name, e.g., 'Accelerometer.csv' -> 'accelero'
            sensor_type = item.split('.')[0].lower()[:len(item.split('.')[0])-1]  # Remove last 'er' or 'or'
            df = process_csv(item_path, sensor_type)
            # Merge data frames on 'Time (s)'
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on='Time (s)', how='outer')
    
    # Add a column for the category label (one-hot encoding)
    combined_df[category_label] = 1
    return combined_df

# Main function to combine all data
def combine_all_data(base_path):
    all_data = pd.DataFrame()
    for category in os.listdir(base_path):  # 'bus', 'metro'
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            for sub_dir in os.listdir(category_path):
                sub_dir_path = os.path.join(category_path, sub_dir)
                if os.path.isdir(sub_dir_path):
                    df = process_subdirectory(sub_dir_path, category)
                    if all_data.empty:
                        all_data = df
                    else:
                        all_data = pd.concat([all_data, df], ignore_index=True)
    return all_data

# Combine the data and save to a new CSV file
combined_data = combine_all_data(base_path)
combined_data.to_csv('data/my/combined_data.csv', index=False)
print("Data combined successfully and saved to 'combined_data.csv'.")

# Optionally display the DataFrame
# print(combined_data.head())
