import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/data_fe.csv')

# Plot histograms for selected features
df[['rolling_cumulative_elevation', 'rolling_avg_direction', 'rolling_avg_velocity']].hist(bins=50, figsize=(12, 8))
# plt.show()

# Define the features and their corresponding names
features = ['rolling_cumulative_elevation', 'rolling_avg_direction', 'rolling_avg_velocity']
names = ['Elevation', 'Direction', 'Velocity']

# Create subplots
fig, axes = plt.subplots(1, len(features), figsize=(18, 6))

# Plot histograms for each feature
for ax, feature, name in zip(axes, features, names):
    ax.hist(df[feature], bins=50)
    ax.set_title(f'Distribution of {name}')

plt.tight_layout()
plt.show()
