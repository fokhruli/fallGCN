import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def load_label_file(label_path, chute_number):
    """Load and parse the label file for specific chute."""
    try:
        df = pd.read_csv(label_path)
        filtered_df = df[df['chute'] == chute_number]
        print(f"Found {len(filtered_df)} entries for chute {chute_number}")
        return filtered_df
    except Exception as e:
        print(f"Error loading label file: {e}")
        return pd.DataFrame()

def process_video_segment(video_path, start_frame, end_frame, detector):
    """Process a video segment and return pose landmarks for each frame, skipping undetected frames."""
    landmarks_list = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return landmarks_list
        
    print(f"Processing video: {video_path}")
    print(f"Frames to process: {start_frame} to {end_frame}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp_ms = int(start_frame * (1000 / fps))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames_processed = 0
    landmarks_detected = 0
    current_frame = start_frame
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {current_frame}")
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        detection_result = detector.detect_for_video(image, timestamp_ms)
        
        frames_processed += 1
        
        if detection_result.pose_landmarks:
            frame_landmarks = []
            for landmark in detection_result.pose_landmarks[0]:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks_list.append(frame_landmarks)
            landmarks_detected += 1
            
        current_frame += 1
        timestamp_ms = int(current_frame * (1000 / fps))
    
    cap.release()
    print(f"Processed {frames_processed} frames")
    print(f"Detected landmarks in {landmarks_detected} frames")
    return landmarks_list

def create_sliding_windows(landmarks, label, window_size=240, stride=120):
    """Create sliding windows from landmarks, saving all frames in each window."""
    X_windows = []
    y_labels = []
    
    print(f"Creating windows from {len(landmarks)} frames")
    
    for i in range(0, len(landmarks) - window_size + 1, stride):
        window = landmarks[i:i + window_size]
        if len(window) == window_size:
            # Add all frames in the window
            for frame in window:
                X_windows.append(frame)
                y_labels.append(label)
    
    print(f"Created {len(X_windows)} total frame entries")
    return X_windows, y_labels

def process_chute(chute_number):
    """Process a specific chute number."""
    model_path = 'pose_landmarker.task'
    all_X = []
    all_y = []
    
    print(f"\nStarting processing for chute {chute_number}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    # Load label file for specific chute
    labels_df = load_label_file('fall_detection_labels_combines.csv', chute_number)
    
    if labels_df.empty:
        print("No data found for this chute")
        return
        
    for _, row in labels_df.iterrows():
        print("\nProcessing new segment:")
        print(f"Camera: {row['cam']}")
        print(f"Frames: {row['start']} to {row['end']}")
        print(f"Label: {row['label']}")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            running_mode=vision.RunningMode.VIDEO)
        detector = vision.PoseLandmarker.create_from_options(options)
        
        cam = row['cam']
        start_frame = row['start']
        end_frame = row['end']
        label = row['label']
        
        video_path = os.path.join('dataset', f'chute{chute_number:02d}', f'cam{cam}.avi')
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found - {video_path}")
            continue
            
        landmarks = process_video_segment(video_path, start_frame, end_frame, detector)
        
        if landmarks:
            X_windows, y_labels = create_sliding_windows(landmarks, label)
            all_X.extend(X_windows)
            all_y.extend(y_labels)
        
        detector.close()
    
    print(f"\nTotal frames collected: {len(all_X)}")
    
    if not all_X:
        print("No data was collected! Check the errors above.")
        return
        
    # Create column names for landmarks
    landmark_columns = []
    for i in range(33):
        landmark_columns.extend([f'x{i+1}', f'y{i+1}', f'z{i+1}'])
    
    X_df = pd.DataFrame(all_X, columns=landmark_columns)
    y_df = pd.DataFrame(all_y, columns=['label'])
    
    # Save with chute number in filename
    output_X = f'Train_X_fall_chute{chute_number:02d}.csv'
    output_y = f'Train_y_fall_chute{chute_number:02d}.csv'
    
    X_df.to_csv(output_X, index=False)
    y_df.to_csv(output_y, index=False)
    
    print(f"\nFiles saved:")
    print(f"X data: {output_X} ({len(X_df)} rows)")
    print(f"y data: {output_y} ({len(y_df)} rows)")

if __name__ == "__main__":
    # Process chute 1 by default
    chute_number = 1
    process_chute(chute_number)