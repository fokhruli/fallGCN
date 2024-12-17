import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests


def download_model():
    """Download the pose landmarker model if it doesn't exist."""
    model_path = 'pose_landmarker.task'
    if not os.path.exists(model_path):
        print("Downloading pose landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task" #"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")
    return model_path

def create_detector():
    """Create a new instance of PoseLandmarker."""
    base_options = python.BaseOptions(
        model_asset_path=model_path
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.VIDEO)
    return vision.PoseLandmarker.create_from_options(options)

# Download the model
model_path = download_model()

# STEP 2: Specify the path to your dataset folder.
dataset_path = "dataset"

# Create dataset directory if it doesn't exist
os.makedirs(dataset_path, exist_ok=True)

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
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):  # Only process video files
            continue
            
        video_path = os.path.join(folder_path, video_file)
        
        # Skip directories and process only video files.
        if not os.path.isfile(video_path):
            continue
        
        try:
            # Create a new detector instance for each video
            detector = create_detector()
            
            # Open the video file.
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                print(f"Error: Could not open video file {video_file}")
                continue

            # Get video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_interval_ms = int(1000.0 / fps)
            
            # Get the video name without the extension.
            video_name = os.path.splitext(video_file)[0]
            
            # Create a file to store the landmark data for the current video.
            landmarks_file = os.path.join(landmarks_folder, f"{video_name}_landmarks.txt")
            
            # Clear the file if it exists
            with open(landmarks_file, "w") as file:
                file.write(f"Video: {video_file}\nFPS: {fps}\n\n")
            
            # STEP 5: Process each frame of the video.
            frame_count = 0
            timestamp_ms = 0
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                try:
                    # Convert the frame to RGB.
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Create an MP Image object from the frame.
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    
                    # Detect pose landmarks from the frame.
                    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
                    
                    # Only write data if landmarks were detected
                    if detection_result.pose_landmarks:
                        with open(landmarks_file, "a") as file:
                            file.write(f"Frame {frame_count} (timestamp: {timestamp_ms}ms):\n")
                            for idx, landmark in enumerate(detection_result.pose_landmarks[0]):
                                file.write(f"Landmark {idx}: x={landmark.x:.6f}, y={landmark.y:.6f}, z={landmark.z:.6f}\n")
                            file.write("\n")
                    
                    frame_count += 1
                    timestamp_ms += frame_interval_ms

                except Exception as e:
                    print(f"Error processing frame {frame_count} of {video_file}: {str(e)}")
                    continue
            
            video.release()
            # Close and delete the detector
            detector.close()
            del detector
            print(f"Processed video: {video_file} ({frame_count} frames)")
            
        except Exception as e:
            print(f"Error processing video {video_file}: {str(e)}")
            if 'video' in locals():
                video.release()
            if 'detector' in locals():
                detector.close()
                del detector
            continue

print("Landmark data collection completed.")