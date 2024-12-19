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

# Define the features to calculate
FEATURE_NAMES = [
    # Body Segment Angles (27 features)
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
    
    # Centroid Locations (3 features)
    'C_upper',
    'C_lower',
    'C_total',
    
    # Hip and Shoulder Coordinates (14 features)
    'left_hip_x', 'left_hip_y',
    'right_hip_x', 'right_hip_y',
    'left_shoulder_x', 'left_shoulder_y',
    'right_shoulder_x', 'right_shoulder_y',
    'left_elbow_x', 'left_elbow_y',
    'right_elbow_x', 'right_elbow_y',
    'left_knee_x', 'left_knee_y',
    
    # Note: Yaw Trunk Angle is included as 'trunk_rot_angle'
]

def load_landmark_data(csv_file):
    """
    Load landmark data from a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing landmark data.
    """
    df = pd.read_csv(csv_file, header=None)
    return df

def extract_landmarks(row):
    """
    Extract landmark coordinates from a row.
    
    Args:
        row (pd.Series): Row from the DataFrame.
    
    Returns:
        dict: Dictionary mapping landmark names to their (x, y, z) coordinates.
    """
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
    """
    Calculate the angle at joint p2 formed by points p1, p2, p3.
    
    Args:
        p1 (np.array): Coordinates of the first point.
        p2 (np.array): Coordinates of the joint point.
        p3 (np.array): Coordinates of the third point.
    
    Returns:
        float: Angle in degrees at joint p2.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Prevent numerical errors
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def calculate_segment_angle(p1, p2, vertical=np.array([0, 1, 0])):
    """
    Calculate the angle between a body segment and the vertical axis.
    
    Args:
        p1 (np.array): Starting point of the segment.
        p2 (np.array): Ending point of the segment.
        vertical (np.array): Vertical axis vector.
    
    Returns:
        float: Angle in degrees.
    """
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
    """
    Calculate the centroid of a set of points.
    
    Args:
        points (list): List of (x, y, z) coordinates.
    
    Returns:
        np.array: Centroid coordinates [x, y, z].
    """
    return np.mean(points, axis=0)

def calculate_yaw_angle(shoulder_l, shoulder_r):
    """
    Calculate the yaw angle of the trunk.
    
    Args:
        shoulder_l (np.array): Left shoulder coordinates [x, y, z].
        shoulder_r (np.array): Right shoulder coordinates [x, y, z].
    
    Returns:
        float: Yaw angle in degrees.
    """
    delta_y = shoulder_r[1] - shoulder_l[1]
    delta_x = shoulder_r[0] - shoulder_l[0]
    return np.degrees(np.arctan2(delta_y, delta_x))

def calculate_frontal_plane_normal(coords):
    """
    Calculate the normal vector of the frontal plane.
    
    Args:
        coords (dict): Dictionary of landmark coordinates.
    
    Returns:
        np.array: Normalized frontal plane normal vector.
    """
    chest_vector = coords['right_shoulder'] - coords['left_shoulder']
    pelvis_vector = coords['right_hip'] - coords['left_hip']
    normal = np.cross(chest_vector, pelvis_vector)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.array([0, 0, 1])  # Default normal if vectors are parallel
    return normal / norm

def find_waist_point(coords, alpha=0.5):
    """
    Calculate the waist point for body segmentation.
    
    Args:
        coords (dict): Dictionary of landmark coordinates.
        alpha (float): Weighting factor for waist level.
    
    Returns:
        np.array: Waist point coordinates [x, y, z].
    """
    hip_center = (coords['left_hip'] + coords['right_hip']) / 2
    shoulder_center = (coords['left_shoulder'] + coords['right_shoulder']) / 2
    return alpha * hip_center + (1 - alpha) * shoulder_center

def calculate_centroids(coords, waist_point):
    """
    Calculate upper body, lower body, and total body centroids as distances from the waist point.
    
    Args:
        coords (dict): Dictionary of landmark coordinates.
        waist_point (np.array): Coordinates of the waist point.
    
    Returns:
        dict: Dictionary containing centroid distances.
    """
    # Upper Body Centroid
    upper_points = [
        coords['left_shoulder'], coords['right_shoulder'],
        coords['left_elbow'], coords['right_elbow'],
        coords['left_wrist'], coords['right_wrist']
    ]
    C_upper = calculate_centroid(upper_points)
    distance_upper = np.linalg.norm(C_upper - waist_point)
    
    # Lower Body Centroid
    lower_points = [
        coords['left_hip'], coords['right_hip'],
        coords['left_knee'], coords['right_knee'],
        coords['left_ankle'], coords['right_ankle']
    ]
    C_lower = calculate_centroid(lower_points)
    distance_lower = np.linalg.norm(C_lower - waist_point)
    
    # Total Body Centroid
    all_points = list(coords.values())
    C_total = calculate_centroid(all_points)
    distance_total = np.linalg.norm(C_total - waist_point)
    
    return {
        'C_upper': distance_upper,
        'C_lower': distance_lower,
        'C_total': distance_total
    }

def calculate_segment_mass(total_mass, segment, gender='male'):
    """
    Calculate segment mass based on Plagenhoef (1983) data.
    
    Args:
        total_mass (float): Total body mass in kg.
        segment (str): Segment name.
        gender (str): 'male' or 'female'.
    
    Returns:
        float: Mass of the specified segment in kg.
    """
    return total_mass * MASS_RATIOS[segment] # MASS_RATIOS[gender][segment]

def extract_biomechanical_features(coords, total_mass, gender='male'):
    """
    Extract all 45 biomechanical features.
    
    Args:
        coords (dict): Dictionary of landmark coordinates.
        total_mass (float): Total body mass in kg.
        gender (str): 'male' or 'female'.
    
    Returns:
        dict: Dictionary containing all 45 biomechanical features.
    """
    features = {}
    
    # 1. Body Segment Angles (27 features)
    # a. Trunk Angles
    hip_center = (coords['left_hip'] + coords['right_hip']) / 2
    shoulder_center = (coords['left_shoulder'] + coords['right_shoulder']) / 2
    spine_vector = shoulder_center - hip_center
    
    trunk_flex_ext = calculate_segment_angle(hip_center, shoulder_center)
    trunk_abd_add = calculate_segment_angle(shoulder_center, hip_center)  # Example
    trunk_rot = calculate_yaw_angle(coords['left_shoulder'], coords['right_shoulder'])
    features['trunk_flex_ext_angle'] = trunk_flex_ext
    features['trunk_abd_add_angle'] = trunk_abd_add
    features['trunk_rot_angle'] = trunk_rot
    
    # b. Upper Arm Angles (Left and Right)
    for side in ['left', 'right']:
        shoulder = coords[f'{side}_shoulder']
        elbow = coords[f'{side}_elbow']
        wrist = coords[f'{side}_wrist']
        
        # Flexion/Extension
        flex_ext = calculate_joint_angle(shoulder, elbow, wrist)
        # Abduction/Adduction
        abd_add = calculate_segment_angle(shoulder, elbow)
        # Rotation
        rot = calculate_segment_angle(elbow, wrist)
        
        features[f'upper_arm_{side}_flex_ext_angle'] = flex_ext
        features[f'upper_arm_{side}_abd_add_angle'] = abd_add
        features[f'upper_arm_{side}_rot_angle'] = rot
    
    # c. Forearm Angles (Left and Right)
    for side in ['left', 'right']:
        elbow = coords[f'{side}_elbow']
        wrist = coords[f'{side}_wrist']
        pinky = coords[f'{side}_pinky']
        
        # Flexion/Extension
        flex_ext = calculate_joint_angle(coords[f'{side}_elbow'], coords[f'{side}_wrist'], coords[f'{side}_pinky'])
        # Abduction/Adduction
        abd_add = calculate_segment_angle(coords[f'{side}_elbow'], coords[f'{side}_wrist'])
        # Rotation
        rot = calculate_segment_angle(coords[f'{side}_wrist'], coords[f'{side}_pinky'])
        
        features[f'forearm_{side}_flex_ext_angle'] = flex_ext
        features[f'forearm_{side}_abd_add_angle'] = abd_add
        features[f'forearm_{side}_rot_angle'] = rot
    
    # d. Thigh Angles (Left and Right)
    for side in ['left', 'right']:
        hip = coords[f'{side}_hip']
        knee = coords[f'{side}_knee']
        
        # Flexion/Extension
        flex_ext = calculate_joint_angle(hip, knee, coords[f'{side}_ankle'])
        # Abduction/Adduction
        abd_add = calculate_segment_angle(hip, knee)
        # Rotation
        rot = calculate_segment_angle(knee, coords[f'{side}_ankle'])
        
        features[f'thigh_{side}_flex_ext_angle'] = flex_ext
        features[f'thigh_{side}_abd_add_angle'] = abd_add
        features[f'thigh_{side}_rot_angle'] = rot
    
    # e. Shank Angles (Left and Right)
    for side in ['left', 'right']:
        knee = coords[f'{side}_knee']
        ankle = coords[f'{side}_ankle']
        heel = coords[f'{side}_heel']
        
        # Flexion/Extension
        flex_ext = calculate_joint_angle(knee, ankle, heel)
        # Abduction/Adduction
        abd_add = calculate_segment_angle(knee, ankle)
        # Rotation
        rot = calculate_segment_angle(ankle, heel)
        
        features[f'shank_{side}_flex_ext_angle'] = flex_ext
        features[f'shank_{side}_abd_add_angle'] = abd_add
        features[f'shank_{side}_rot_angle'] = rot
    
    # Repeat similar calculations for other body segments if applicable
    # Ensure all 9 segments have 3 angles each
    
    # 2. Centroid Locations (3 features)
    waist_point = find_waist_point(coords, alpha=0.5)
    centroids = calculate_centroids(coords, waist_point)
    features['C_upper'] = centroids['C_upper']
    features['C_lower'] = centroids['C_lower']
    features['C_total'] = centroids['C_total']
    
    # 3. Yaw Trunk Angle (Already included as 'trunk_rot_angle')
    
    # 4. Hip and Shoulder Coordinates (14 features)
    key_joints = [
        'left_hip', 'right_hip',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_knee'
    ]
    for joint in key_joints:
        features[f'{joint}_x'] = coords[joint][0]
        features[f'{joint}_y'] = coords[joint][1]
    
    # 5. Joint Angles (Removed to prevent redundancy)
    # Ensure all necessary angles are included in Body Segment Angles
    
    return features

def process_csv(input_csv, output_csv, total_mass=70.0, gender='male'):
    """
    Process the input CSV to calculate biomechanical features and save to output CSV.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the output CSV file with features.
        total_mass (float): Total body mass in kg.
        gender (str): 'male' or 'female'.
    """
    # Load data
    df = load_landmark_data(input_csv)
    num_frames = df.shape[0]
    print(f"Processing {num_frames} frames... ")
    
    # Initialize list to hold features for each frame
    features_list = []
    count_frame = 0
    for index, row in df.iterrows():
        print(f"Frame number {count_frame} processing right now")
        coords = extract_landmarks(row)
        
        # Handle cases where certain landmarks might be missing or have NaN values
        missing_landmarks = [name for name, coord in coords.items() if np.isnan(coord).any()]
        if missing_landmarks:
            print(f"Frame {index}: Missing landmarks {missing_landmarks}. Skipping frame.")
            continue
        
        # Extract biomechanical features
        features = extract_biomechanical_features(coords, total_mass, gender)
        features_list.append(features)
        count_frame += 1
    
    # Create DataFrame from features
    features_df = pd.DataFrame(features_list, columns=FEATURE_NAMES)
    
    # Save to CSV
    features_df.to_csv(output_csv, index=False, header=None)
    print(f"Features saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    input_csv_path = 'fall_dataset_preprocessed/X_train_fall.csv'  # Replace with your input CSV file path
    output_csv_path = 'fall_dataset_preprocessed/X_train_fall_biomechanical_features.csv'  # Desired output CSV file path
    total_body_mass = 70.0  # Example total body mass in kg
    gender = 'male'          # 'male' or 'female'
    
    process_csv(input_csv_path, output_csv_path, total_body_mass, gender)
