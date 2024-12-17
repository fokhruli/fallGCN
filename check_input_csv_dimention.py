import pandas as pd
import numpy as np

# Load the data
data_x = pd.read_csv("fall_dataset_preprocessed/X_train_fall.csv", header=None)
data_y = pd.read_csv("fall_dataset_preprocessed/y_train_fall.csv", header=None)
data_mechanical_feature = pd.read_csv("fall_dataset_preprocessed/X_train_fall_biomechanical_features.csv", header=None)

# Get total rows
total_rows_x = len(data_x)
total_rows_y = len(data_y)
total_rows_mechanical_feature = len(data_mechanical_feature)

print(data_x.head(6))
print(data_y.head(5))
print(data_mechanical_feature.head(5))


print(f"Total rows in X_train_fall_90: {total_rows_x}")
print(f"Total rows in y_train_fall_90: {total_rows_y}")
print(f"Total row of length mechanical feature {total_rows_mechanical_feature}")
# Check if they match
print(f"window size: {total_rows_x/total_rows_y}")



# Use value_counts to count occurrences of each unique value
counts = data_y[0].value_counts()

# Display the counts
print(counts)

