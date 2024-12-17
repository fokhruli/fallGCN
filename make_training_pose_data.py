import os
import pandas as pd
import numpy as np
from pathlib import Path

def read_landmarks_file(file_path):
    """Read landmarks data from text file and return as dictionary of frame numbers and landmarks."""
    print(f"\nReading landmarks file: {file_path}")
    landmarks_data = {}
    current_frame = None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    frame_count = 0    
    for line in lines:
        line = line.strip()
        if line.startswith('Frame'):
            current_frame = int(line.split()[1])
            landmarks_data[current_frame] = []
            frame_count += 1
            
        elif line.startswith('Landmark'):
            parts = line.split()
            x = float(parts[2].split('=')[1].rstrip(','))
            y = float(parts[3].split('=')[1].rstrip(','))
            z = float(parts[4].split('=')[1])
            landmarks_data[current_frame].append((x, y, z))
    
    print(f"Total frames read: {frame_count}")        
    return landmarks_data

def check_continuous_frames(frames):
    """Check if there are any gaps of 5 or more continuous frames."""
    frames = sorted(frames)
    for i in range(len(frames)-1):
        if frames[i+1] - frames[i] >= 5:
            return False
    return True

def create_window_data(landmarks_data, start_frame, window_size=90):
    """Create a single window of data if all frames are available."""
    frames = list(range(start_frame, start_frame + window_size))
    
    # Check if all frames exist and are continuous
    all_frames_exist = all(frame in landmarks_data for frame in frames)
    if not all_frames_exist:
        print(f"  Skipping window at frame {start_frame}: Missing frames")
        return None
        
    if not check_continuous_frames(frames):
        print(f"  Skipping window at frame {start_frame}: Gap of 5+ frames detected")
        return None
        
    # Create feature vector for the window
    window_data = []
    for frame in frames:
        frame_landmarks = landmarks_data[frame]
        # Flatten the landmarks into a single row
        frame_data = [coord for landmark in frame_landmarks for coord in landmark]
        window_data.append(frame_data)
        
    return window_data

def get_label_for_window(labels_df, chute, cam, start_frame, next_frames=60):
    """Get the label for the next 60 frames after the window."""
    window_end = start_frame + 90  # Current window end
    label_range = labels_df[
        (labels_df['chute'] == chute) & 
        (labels_df['cam'] == cam) &
        (labels_df['start'] <= window_end + next_frames) &
        (labels_df['end'] >= window_end)
    ]
    
    if len(label_range) == 0:
        print(f"  No label found for window at frame {start_frame}")
        return None
        
    # Return the majority label for the next 60 frames
    return label_range.iloc[0]['label']

def process_data(data_dir, labels_file, output_x, output_y, start_chute=1, end_chute=1, window_size=90):
    """Process data files for specified chute range and create training datasets."""
    print(f"Starting data processing...")
    print(f"Processing chutes {start_chute} to {end_chute}")
    print(f"Reading labels from: {labels_file}")
    
    # Read labels
    labels_df = pd.read_csv(labels_file)
    print(f"Found {len(labels_df)} label entries")
    
    X_data = []
    y_data = []
    
    total_windows_processed = 0
    total_windows_saved = 0
    
    # Filter labels for specified chute range
    labels_df = labels_df[
        (labels_df['chute'] >= start_chute) & 
        (labels_df['chute'] <= end_chute)
    ]
    
    # Get unique chute-cam combinations from filtered labels
    chute_cam_pairs = labels_df[['chute', 'cam']].drop_duplicates().values
    print(f"\nFound {len(chute_cam_pairs)} unique chute-camera pairs in labels for chutes {start_chute}-{end_chute}")
    
    # Process each chute-cam pair that exists in labels
    for chute, cam in chute_cam_pairs:
        landmarks_file = os.path.join(
            data_dir, 
            f'chute{chute:02d}', 
            'landmarks',
            f'cam{cam}_landmarks.txt'
        )
        
        if not os.path.exists(landmarks_file):
            print(f"\nSkipping non-existent file: {landmarks_file}")
            continue
            
        print(f"\nProcessing Chute {chute:02d}, Camera {cam}")
        
        # Get frame ranges for this chute-cam from labels
        chute_labels = labels_df[
            (labels_df['chute'] == chute) & 
            (labels_df['cam'] == cam)
        ]
        
        min_frame = chute_labels['start'].min()
        max_frame = chute_labels['end'].max()
        print(f"Label frame range: {min_frame} to {max_frame}")
        
        # Read landmarks data
        landmarks_data = read_landmarks_file(landmarks_file)
        
        # Filter landmarks to only include frames within label range
        valid_frames = sorted([f for f in landmarks_data.keys() 
                             if min_frame <= f <= max_frame])
        
        if not valid_frames:
            print(f"No valid frames found in range for Chute {chute:02d}, Camera {cam}")
            continue
            
        print(f"Processing frames {valid_frames[0]} to {valid_frames[-1]}")
        print(f"Total valid frames: {len(valid_frames)}")
        
        # Process each possible window within the valid frame range
        windows_saved = 0
        for start_frame in valid_frames[:-window_size]:
            total_windows_processed += 1
            window_data = create_window_data(landmarks_data, start_frame, window_size)
            if window_data is None:
                continue
                
            label = get_label_for_window(chute_labels, chute, cam, start_frame)
            if label is not None:
                X_data.extend(window_data)
                y_data.extend([label] * window_size)
                windows_saved += 1
                total_windows_saved += 1
        
        print(f"Saved {windows_saved} valid windows for Chute {chute:02d}, Camera {cam}")
    
    print("\nProcessing complete!")
    print(f"Total windows processed: {total_windows_processed}")
    print(f"Total windows saved: {total_windows_saved}")
    print(f"Total frames saved: {len(X_data)}")
    
    if len(X_data) == 0:
        print("No data to save. Exiting.")
        return
    
    # Create column names for X data
    columns = []
    for i in range(33):  # 33 landmarks
        columns.extend([f'landmark{i}_x', f'landmark{i}_y', f'landmark{i}_z'])
    
    print("\nSaving data to CSV files...")
    # Save to CSV with proper column names
    # X_df = pd.DataFrame(X_data, columns=None)
# Create DataFrame and save without headers
    X_df = pd.DataFrame(X_data)
    X_df.to_csv(output_x, index=False, header=False)
    y_df = pd.DataFrame(y_data, columns=None)
    
    # X_df.to_csv(output_x, index=False)

    y_df.to_csv(output_y, index=False)
    print(f"Data saved to {output_x} and {output_y}")
    print(f"X shape: {X_df.shape}, y shape: {y_df.shape}")

# Example usage
if __name__ == "__main__":
    data_dir = "dataset"
    labels_file = "fall_detection_labels_combines.csv"
    output_x = "X_train_fall_90.csv"
    output_y = "y_train_fall_90.csv"
    
    # Process only chute 1 (can be modified to process more chutes)
    START_CHUTE = 1
    END_CHUTE = 1  # Change this to process more chutes (up to 24)
    
    process_data(
        data_dir=data_dir,
        labels_file=labels_file,
        output_x=output_x,
        output_y=output_y,
        start_chute=START_CHUTE,
        end_chute=END_CHUTE
    )