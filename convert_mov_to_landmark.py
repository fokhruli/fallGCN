import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# List of supported video formats
VIDEO_FORMATS = ['.mov', '.avi', '.mp4']

def process_video_to_landmarks(video_path, output_path):
    """Process a video file and save landmarks to a formatted txt file."""
    try:
        # Initialize MediaPipe Pose detector
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            running_mode=vision.RunningMode.VIDEO)
        detector = vision.PoseLandmarker.create_from_options(options)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return False
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Open output file and write header
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            # Write video info header
            video_name = os.path.basename(video_path)
            f.write(f"Video: {video_name}\n")
            f.write(f"FPS: {fps}\n\n")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                timestamp_ms = int(frame_count * (1000 / fps))
                detection_result = detector.detect_for_video(image, timestamp_ms)
                
                if detection_result.pose_landmarks:
                    # Write frame header
                    f.write(f"Frame {frame_count} (timestamp: {timestamp_ms}ms):\n")
                    
                    # Write landmarks
                    landmarks = detection_result.pose_landmarks[0]
                    for i, landmark in enumerate(landmarks):
                        f.write(f"Landmark {i}: x={landmark.x:.6f}, y={landmark.y:.6f}, z={landmark.z:.6f}\n")
                    f.write("\n")  # Add blank line between frames
                
                frame_count += 1
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"\rProgress: {progress:.1f}% ({frame_count}/{total_frames} frames)", end="")
            
        cap.release()
        detector.close()
        print()  # New line after progress
        print(f"Processed {frame_count} frames, saved landmarks to: {output_path}")
        return True
            
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False

def validate_path(path):
    """Validate if the path exists and contains supported video files."""
    if not os.path.exists(path):
        return False, "Directory does not exist"
    
    has_video_files = False
    for root, _, files in os.walk(path):
        if any(file.lower().endswith(tuple(VIDEO_FORMATS)) for file in files):
            has_video_files = True
            break
    
    if not has_video_files:
        return False, f"No video files ({', '.join(VIDEO_FORMATS)}) found in the directory or its subdirectories"
    
    return True, "Valid path"

def process_dataset(input_root):
    """Crawl through dataset folder and process all video files."""
    output_root = 'dataset_landmark'
    
    # Validate input path
    is_valid, message = validate_path(input_root)
    if not is_valid:
        print(f"Error: {message}")
        return
        
    # Create output root directory
    os.makedirs(output_root, exist_ok=True)
    
    # Count total videos to process
    total_videos = sum(1 for root, _, files in os.walk(input_root) 
                      for file in files if file.lower().endswith(tuple(VIDEO_FORMATS)))
    processed_videos = 0
    
    print(f"\nFound {total_videos} video files to process")
    
    # Walk through all directories
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(tuple(VIDEO_FORMATS)):
                processed_videos += 1
                # Get relative path from input root
                rel_path = os.path.relpath(root, input_root)
                
                # Construct input and output paths
                input_path = os.path.join(root, file)
                output_path = os.path.join(
                    output_root,
                    rel_path,
                    os.path.splitext(file)[0] + '.txt'
                )
                
                print(f"\nProcessing video {processed_videos}/{total_videos}")
                print(f"Input: {input_path}")
                print(f"Output: {output_path}")
                
                process_video_to_landmarks(input_path, output_path)

def main():
    print(f"Supported video formats: {', '.join(VIDEO_FORMATS)}")
    while True:
        # Get input path from user
        input_path = input("\nEnter the path to your dataset folder [example: /home/rmedu/Fokhrul/fallGCN/dataset] (or 'q' to quit): ").strip()
        
        if input_path.lower() == 'q':
            print("Exiting program...")
            break
            
        # Remove quotes if present
        input_path = input_path.strip("'\"")
        
        # Validate and process
        is_valid, message = validate_path(input_path)
        if is_valid:
            print(f"\nProcessing dataset from: {input_path}")
            print(f"Output will be saved to: dataset_landmark")
            process_dataset(input_path)
        else:
            print(f"Error: {message}")
            print("Please enter a valid path or 'q' to quit")

if __name__ == "__main__":
    main()