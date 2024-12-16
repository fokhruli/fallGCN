import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def load_label_file(label_path):
    """Load and parse the label file."""
    return pd.read_csv(label_path)

def process_video_segment(video_path, start_frame, end_frame, detector):
    """Process a video segment and return pose landmarks for each frame."""
    landmarks_list = []
    cap = cv2.VideoCapture(video_path)
    
    # Get video FPS for accurate timestamp calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp_ms = int(start_frame * (1000 / fps))  # Convert frame number to milliseconds
    
    # Set video to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect pose landmarks
        detection_result = detector.detect_for_video(image, timestamp_ms)
        
        # Check if pose was detected
        if detection_result.pose_landmarks:
            # Flatten landmarks into a single list [x1,y1,z1,x2,y2,z2,...]
            frame_landmarks = []
            for landmark in detection_result.pose_landmarks[0]:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks_list.append(frame_landmarks)
        else:
            # If no pose detected, fill with zeros
            landmarks_list.append([0.0] * 99)  # 33 landmarks * 3 coordinates
            
        current_frame += 1
        timestamp_ms = int(current_frame * (1000 / fps))  # Update timestamp based on frame number
    
    cap.release()
    return landmarks_list

def create_sliding_windows(landmarks, label, window_size=240, stride=120):
    """Create sliding windows from landmarks with corresponding labels."""
    X_windows = []
    y_labels = []
    
    for i in range(0, len(landmarks) - window_size + 1, stride):
        window = landmarks[i:i + window_size]
        if len(window) == window_size:  # Only add complete windows
            # Calculate mean of each feature across the window
            window_mean = np.mean(window, axis=0)
            X_windows.append(window_mean)
            y_labels.append(label)
    
    return X_windows, y_labels

def main():
    model_path = 'pose_landmarker.task'
    all_X = []
    all_y = []
    
    # Load label file
    labels_df = load_label_file('fall_detection_labels_combines.csv')
    
    # Process each entry in the label file
    for _, row in labels_df.iterrows():
        # Create a new detector instance for each video segment
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            running_mode=vision.RunningMode.VIDEO)
        detector = vision.PoseLandmarker.create_from_options(options)
        
        chute = row['chute']
        cam = row['cam']
        start_frame = row['start']
        end_frame = row['end']
        label = row['label']
        
        # Construct video path
        video_path = os.path.join('dataset', f'chute{chute:02d}', f'cam{cam}.avi')
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found - {video_path}")
            continue
            
        print(f"Processing chute {chute}, camera {cam}, frames {start_frame}-{end_frame}")
        
        # Process video segment
        landmarks = process_video_segment(video_path, start_frame, end_frame, detector)
        
        # Create windows from the segment
        X_windows, y_labels = create_sliding_windows(landmarks, label)
        
        # Add to overall lists
        all_X.extend(X_windows)
        all_y.extend(y_labels)
        
        # Clean up detector
        detector.close()
    
    # Create column names for landmarks
    landmark_columns = []
    for i in range(33):  # 33 landmarks
        landmark_columns.extend([f'x{i+1}', f'y{i+1}', f'z{i+1}'])
    
    # Convert to DataFrames
    X_df = pd.DataFrame(all_X, columns=landmark_columns)
    y_df = pd.DataFrame(all_y, columns=['label'])
    
    # Save to CSV
    X_df.to_csv('Train_X_fall.csv', index=False)
    y_df.to_csv('Train_y_fall.csv', index=False)
    
    print("Processing completed successfully!")



if __name__ == "__main__":
    main()