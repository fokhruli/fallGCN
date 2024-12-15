import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 1: Set up the PoseLandmarker object.
base_options = python.BaseOptions(
    model_asset_path='pose_landmarker.task',
    num_threads=4  # Specify the number of CPU threads to use
)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 2: Specify the path to your dataset folder.
dataset_path = "dataset"

# STEP 3: Iterate over each folder in the dataset.
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    
    # Skip files and process only folders.
    if not os.path.isdir(folder_path):
        continue
    
    # Create a folder to store the landmark data for the current folder.
    landmarks_folder = os.path.join(folder_path, "landmarks")
    os.makedirs(landmarks_folder, exist_ok=True)
    
    # STEP 4: Iterate over each video file in the current folder.
    for video_file in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_file)
        
        # Skip directories and process only video files.
        if not os.path.isfile(video_path):
            continue
        
        # Open the video file.
        video = cv2.VideoCapture(video_path)
        
        # Get the video name without the extension.
        video_name = os.path.splitext(video_file)[0]
        
        # Create a file to store the landmark data for the current video.
        landmarks_file = os.path.join(landmarks_folder, video_name + ".txt")
        
        # STEP 5: Process each frame of the video.
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Convert the frame to RGB.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create an MP Image object from the frame.
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Detect pose landmarks from the frame.
            detection_result = detector.detect(image)
            
            # Save the landmark data to the file.
            with open(landmarks_file, "a") as file:
                file.write(f"Frame {frame_count}:\n")
                for idx, landmark in enumerate(detection_result.pose_landmarks[0]):
                    file.write(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, z={landmark.z}\n")
                file.write("\n")
            
            frame_count += 1
        
        video.release()
        print(f"Processed video: {video_file}")

print("Landmark data collected successfully.")