# import cv2

# def get_video_info(video_path):
#     """Get total number of frames and total duration in minutes of a video"""
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         raise ValueError(f"Error opening video file: {video_path}")
    
#     # Get total number of frames
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # Get frames per second (FPS)
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # Calculate total duration in seconds
#     total_duration_seconds = total_frames / fps
    
#     # Convert total duration to minutes
#     # total_duration_minutes = total_duration_seconds 
    
#     cap.release()
    
#     return total_frames, total_duration_seconds

# # Example usage
# for i in range(1,9):
#     print("camera number: ", i)
#     video_path = f"dataset/chute24/cam{i}.avi"
#     total_frames, total_duration_sec = get_video_info(video_path)
#     # print(f"Total frames: {total_frames}")
#     # print(f"Total duration (seconds): {total_duration_sec:.4f}")
#     print(f"FPS?: {total_frames/total_duration_sec}")

import pandas as pd
import numpy as np

# Load the data
data_x = pd.read_csv("X_train_fall_90.csv", header=None)
data_y = pd.read_csv("y_train_fall_90.csv", header=None)

# Get total rows
total_rows_x = len(data_x)
total_rows_y = len(data_y)

print(data_x.head(6))
print(data_y.head(5))

print(f"Total rows in X_train_fall_90: {total_rows_x}")
print(f"Total rows in y_train_fall_90: {total_rows_y}")

# Check if they match
print(f"window size: {total_rows_x/total_rows_y}")



# Use value_counts to count occurrences of each unique value
counts = data_y[0].value_counts()

# Display the counts
print(counts)



#         columns = []
#     for i in range(33):  # 33 landmarks
#         columns.extend([f'landmark{i}_x', f'landmark{i}_y', f'landmark{i}_z'])
    
#     print("\nSaving data to CSV files...")
#     # Save to CSV with proper column names
#     # X_df = pd.DataFrame(X_data, columns=None)
# # Create DataFrame and save without headers
#     X_df = pd.DataFrame(X_data)
#     X_df.to_csv(output_x, index=False, header=False)
#     y_df = pd.DataFrame(y_data, columns=None)
    
#     # X_df.to_csv(output_x, index=False)

#     y_df.to_csv(output_y, index=False)
#     print(f"Data saved to {output_x} and {output_y}")
#     print(f"X shape: {X_df.shape}, y shape: {y_df.shape}")
