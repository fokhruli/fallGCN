import pandas as pd
import numpy as np
import math

# Define the mapping from landmark index to name
LANDMARK_NAMES = {
    0: 'nose',
    1: 'left_eye_inner',
    2: 'left_eye',
    3: 'left_eye_outer',
    4: 'right_eye_inner',
    5: 'right_eye',
    6: 'right_eye_outer',
    7: 'left_ear',
    8: 'right_ear',
    9: 'mouth_left',
    10: 'mouth_right',
    11: 'left_shoulder',
    12: 'right_shoulder',
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
    17: 'left_pinky',
    18: 'right_pinky',
    19: 'left_index',
    20: 'right_index',
    21: 'left_thumb',
    22: 'right_thumb',
    23: 'left_hip',
    24: 'right_hip',
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle',
    29: 'left_heel',
    30: 'right_heel',
    31: 'left_foot_index',
    32: 'right_foot_index'
}

# Anthropometric Data (Plagenhoef, 1983) - mass ratios for males
MASS_RATIOS = {
    'head': 0.0826,
    'trunk': 0.4684,
    'upper_arm': 0.0325,
    'forearm': 0.0187,
    'hand': 0.0065,
    'thigh': 0.1050,
    'shank': 0.0475,
    'foot': 0.0143
}

# Define the features to calculate (45 features total)
FEATURE_NAMES = [
    # 1. Body Segment Angles (27 features)
    'trunk_flex_ext_angle',
    'trunk_abd_add_angle',
    'trunk_rot_angle',
    
    'upper_arm_left_flex_ext_angle',
    'upper_arm_left_abd_add_angle',
    'upper_arm_left_rot_angle',
    'upper_arm_right_flex_ext_angle',
    'upper_arm_right_abd_add_angle',
    'upper_arm_right_rot_angle',
    
    'forearm_left_flex_ext_angle',
    'forearm_left_abd_add_angle',
    'forearm_left_rot_angle',
    'forearm_right_flex_ext_angle',
    'forearm_right_abd_add_angle',
    'forearm_right_rot_angle',
    
    'thigh_left_flex_ext_angle',
    'thigh_left_abd_add_angle',
    'thigh_left_rot_angle',
    'thigh_right_flex_ext_angle',
    'thigh_right_abd_add_angle',
    'thigh_right_rot_angle',
    
    'shank_left_flex_ext_angle',
    'shank_left_abd_add_angle',
    'shank_left_rot_angle',
    'shank_right_flex_ext_angle',
    'shank_right_abd_add_angle',
    'shank_right_rot_angle',
    
    # 2. Centroid Locations (9 features)
    'C_upper_x', 'C_upper_y', 'C_upper_z',
    'C_lower_x', 'C_lower_y', 'C_lower_z',
    'C_total_x', 'C_total_y', 'C_total_z',
    
    # 3. Hip and Shoulder Coordinates (8 features)
    'left_hip_x', 'left_hip_y',
    'right_hip_x', 'right_hip_y',
    'left_shoulder_x', 'left_shoulder_y',
    'right_shoulder_x', 'right_shoulder_y'
]

def load_landmark_data(csv_file):
    """Load landmark data from a CSV file."""
    df = pd.read_csv(csv_file, header=None)
    return df

def extract_landmarks(row):
    """Extract landmark coordinates from a row."""
    coords = {}
    for i in range(33):
        landmark_name = LANDMARK_NAMES.get(i)
        if landmark_name:
            x = row[i*3]
            y = row[i*3 + 1]
            z = row[i*3 + 2]
            coords[landmark_name] = np.array([x, y, z])
    return coords

def calculate_joint_angle(p1, p2, p3):
    """Calculate the angle at joint p2 formed by points p1, p2, p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def calculate_segment_angle(p1, p2, vertical=np.array([0, 1, 0])):
    """Calculate the angle between a body segment and the vertical axis."""
    segment = p2 - p1
    norm_segment = np.linalg.norm(segment)
    norm_vertical = np.linalg.norm(vertical)
    if norm_segment == 0 or norm_vertical == 0:
        return 0.0
    cos_angle = np.dot(segment, vertical) / (norm_segment * norm_vertical)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def calculate_centroid(points):
    """Calculate the centroid of a set of points."""
    return np.mean(points, axis=0)

def calculate_yaw_angle(shoulder_l, shoulder_r):
    """Calculate the yaw angle of the trunk."""
    delta_y = shoulder_r[1] - shoulder_l[1]
    delta_x = shoulder_r[0] - shoulder_l[0]
    return np.degrees(np.arctan2(delta_y, delta_x))

def find_waist_point(coords, alpha=0.5):
    """Calculate the waist point for body segmentation."""
    hip_center = (coords['left_hip'] + coords['right_hip']) / 2
    shoulder_center = (coords['left_shoulder'] + coords['right_shoulder']) / 2
    return alpha * hip_center + (1 - alpha) * shoulder_center

def calculate_centroids(coords):
    """Calculate upper body, lower body, and total body centroids with x,y,z coordinates."""
    # Upper Body Centroid
    upper_points = [
        coords['left_shoulder'], coords['right_shoulder'],
        coords['left_elbow'], coords['right_elbow'],
        coords['left_wrist'], coords['right_wrist']
    ]
    C_upper = calculate_centroid(upper_points)
    
    # Lower Body Centroid
    lower_points = [
        coords['left_hip'], coords['right_hip'],
        coords['left_knee'], coords['right_knee'],
        coords['left_ankle'], coords['right_ankle']
    ]
    C_lower = calculate_centroid(lower_points)
    
    # Total Body Centroid
    all_points = list(coords.values())
    C_total = calculate_centroid(all_points)
    
    return {
        'C_upper': C_upper,
        'C_lower': C_lower,
        'C_total': C_total
    }

def extract_biomechanical_features(coords):
    """Extract all 45 biomechanical features."""
    features = {}
    
    # 1. Body Segment Angles (27 features)
    # a. Trunk Angles
    hip_center = (coords['left_hip'] + coords['right_hip']) / 2
    shoulder_center = (coords['left_shoulder'] + coords['right_shoulder']) / 2
    
    trunk_flex_ext = calculate_segment_angle(hip_center, shoulder_center)
    trunk_abd_add = calculate_segment_angle(shoulder_center, hip_center)
    trunk_rot = calculate_yaw_angle(coords['left_shoulder'], coords['right_shoulder'])
    
    features['trunk_flex_ext_angle'] = trunk_flex_ext
    features['trunk_abd_add_angle'] = trunk_abd_add
    features['trunk_rot_angle'] = trunk_rot
    
    # b. Upper Arm Angles (Left and Right)
    for side in ['left', 'right']:
        shoulder = coords[f'{side}_shoulder']
        elbow = coords[f'{side}_elbow']
        wrist = coords[f'{side}_wrist']
        
        flex_ext = calculate_joint_angle(shoulder, elbow, wrist)
        abd_add = calculate_segment_angle(shoulder, elbow)
        rot = calculate_segment_angle(elbow, wrist)
        
        features[f'upper_arm_{side}_flex_ext_angle'] = flex_ext
        features[f'upper_arm_{side}_abd_add_angle'] = abd_add
        features[f'upper_arm_{side}_rot_angle'] = rot
    
    # c. Forearm Angles (Left and Right)
    for side in ['left', 'right']:
        elbow = coords[f'{side}_elbow']
        wrist = coords[f'{side}_wrist']
        pinky = coords[f'{side}_pinky']
        
        flex_ext = calculate_joint_angle(elbow, wrist, pinky)
        abd_add = calculate_segment_angle(elbow, wrist)
        rot = calculate_segment_angle(wrist, pinky)
        
        features[f'forearm_{side}_flex_ext_angle'] = flex_ext
        features[f'forearm_{side}_abd_add_angle'] = abd_add
        features[f'forearm_{side}_rot_angle'] = rot
    
    # d. Thigh Angles (Left and Right)
    for side in ['left', 'right']:
        hip = coords[f'{side}_hip']
        knee = coords[f'{side}_knee']
        ankle = coords[f'{side}_ankle']
        
        flex_ext = calculate_joint_angle(hip, knee, ankle)
        abd_add = calculate_segment_angle(hip, knee)
        rot = calculate_segment_angle(knee, ankle)
        
        features[f'thigh_{side}_flex_ext_angle'] = flex_ext
        features[f'thigh_{side}_abd_add_angle'] = abd_add
        features[f'thigh_{side}_rot_angle'] = rot
    
    # e. Shank Angles (Left and Right)
    for side in ['left', 'right']:
        knee = coords[f'{side}_knee']
        ankle = coords[f'{side}_ankle']
        heel = coords[f'{side}_heel']
        
        flex_ext = calculate_joint_angle(knee, ankle, heel)
        abd_add = calculate_segment_angle(knee, ankle)
        rot = calculate_segment_angle(ankle, heel)
        
        features[f'shank_{side}_flex_ext_angle'] = flex_ext
        features[f'shank_{side}_abd_add_angle'] = abd_add
        features[f'shank_{side}_rot_angle'] = rot
    
    # 2. Centroid Locations (9 features)
    centroids = calculate_centroids(coords)
    
    # Store x,y,z coordinates for each centroid
    for centroid_name in ['C_upper', 'C_lower', 'C_total']:
        centroid_coords = centroids[centroid_name]
        features[f'{centroid_name}_x'] = centroid_coords[0]
        features[f'{centroid_name}_y'] = centroid_coords[1]
        features[f'{centroid_name}_z'] = centroid_coords[2]
    
    # 3. Hip and Shoulder Coordinates (8 features)
    key_joints = [
        'left_hip', 'right_hip',
        'left_shoulder', 'right_shoulder'
    ]
    for joint in key_joints:
        features[f'{joint}_x'] = coords[joint][0]
        features[f'{joint}_y'] = coords[joint][1]
    
    return features

def process_csv(input_csv, output_csv):
    """Process the input CSV to calculate biomechanical features and save to output CSV."""
    # Load data
    df = load_landmark_data(input_csv)
    num_frames = df.shape[0]
    print(f"Processing {num_frames} frames... ")
    
    # Initialize list to hold features for each frame
    features_list = []
    for index, row in df.iterrows():
        print(f"Processing frame {index + 1}/{num_frames}")
        
        # Extract landmarks
        coords = extract_landmarks(row)
        
        # Skip frames with missing landmarks
        missing_landmarks = [name for name, coord in coords.items() if np.isnan(coord).any()]
        if missing_landmarks:
            print(f"Frame {index}: Missing landmarks {missing_landmarks}. Skipping frame.")
            continue
        
        # Extract biomechanical features
        features = extract_biomechanical_features(coords)
        features_list.append(features)
    
    # Create DataFrame from features
    features_df = pd.DataFrame(features_list, columns=FEATURE_NAMES)
    
    # Save to CSV
    features_df.to_csv(output_csv, index=False, header=None)
    print(f"Features saved to {output_csv}")

if __name__ == "__main__":
    input_csv_path = 'fall_dataset_preprocessed/X_train_fall.csv'  # Replace with your input CSV file path
    output_csv_path = 'fall_dataset_preprocessed/X_train_fall_biomechanical_features.csv'  # Desired output CSV file path
    process_csv(input_csv_path, output_csv_path)