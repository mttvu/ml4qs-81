import os
import pandas as pd
import re

# Define the base path to your data directory
base_path = 'data/my'


# A helper function to process each CSV file
def process_csv(file_path):
    df = pd.read_csv(file_path)
    # Rename other columns to include the file prefix
    df.columns = ['time' if col == 'Time (s)' else re.sub(r'\s*\([^)]*\)', '', col).strip().lower().replace(' ', '_') for col in df.columns]
    return df

# A function to process each sub-directory (e.g., 'bus 4-6')
def process_subdirectory(dir_path, category_label):
    combined_df = None
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path) and item_path.endswith('.csv'):
            df = process_csv(item_path)
            # Merge data frames on 'time'
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on='time', how='outer')
    
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
                    df = process_subdirectory(sub_dir_path, 'label_'+category)
                    if all_data.empty:
                        all_data = df
                    else:
                        all_data = pd.concat([all_data, df], ignore_index=True)
    return all_data

combined_data = combine_all_data(base_path)
combined_data.to_csv(f'{base_path}/combined_data.csv', index=False)
