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

print(data_x.head(1))

print(f"Total rows in X_train_fall_90: {total_rows_x}")
print(f"Total rows in y_train_fall_90: {total_rows_y}")

# Check if they match
if total_rows_x == total_rows_y:
    print("\nX and y have matching number of rows!")
    
    # Calculate how many rows per window (90 frames)
    num_windows = total_rows_x // 90
    print(f"\nTotal number of windows: {num_windows}")
    print(f"Each window has 90 frames")
    
    # Show example of first window
    print("\nFirst window ranges:")
    print(f"Rows 0-89: Window 1")
    print(f"Rows 90-179: Window 2")
    print(f"And so on...")
    
    # Show a few labels from first window
    print("\nLabels in first window (should all be the same):")
    print(data_y[0:90])
else:
    print("\nWARNING: X and y have different numbers of rows!")