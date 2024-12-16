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

data_x = pd.read_csv("Train_X_fall.csv", header=None)
data_y = pd.read_csv("Train_y_fall.csv", header=None)

print(data_x.head(1))
print(data_y[0])
