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
    # Body Segment Angles (9 features)
    'trunk_angle',
    'upper_arm_left_angle',
    'upper_arm_right_angle',
    'forearm_left_angle',
    'forearm_right_angle',
    'thigh_left_angle',
    'thigh_right_angle',
    'shank_left_angle',
    'shank_right_angle',
    
    # Centroid Locations (9 features)
    'C_upper_x', 'C_upper_y', 'C_upper_z',
    'C_lower_x', 'C_lower_y', 'C_lower_z',
    'C_total_x', 'C_total_y', 'C_total_z',
    
    # Yaw Trunk Angle (1 feature)
    'yaw_trunk_angle',
    
    # Hip and Shoulder Coordinates (12 features)
    'left_hip_x', 'left_hip_y', 'left_hip_z',
    'right_hip_x', 'right_hip_y', 'right_hip_z',
    'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
    'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z',
    
    # Joint Angles (14 features)
    'elbow_left_angle',
    'elbow_right_angle',
    'shoulder_left_angle',
    'shoulder_right_angle',
    'hip_left_angle',
    'hip_right_angle',
    'knee_left_angle',
    'knee_right_angle',
    'ankle_left_angle',
    'ankle_right_angle',
    'wrist_left_angle',
    'wrist_right_angle',
    'foot_left_angle',
    'foot_right_angle'
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

def calculate_segment_angle(point1, point2, vertical=np.array([0, 1, 0])):
    """
    Calculate the angle between a body segment and the vertical axis.
    
    Args:
        point1 (np.array): Starting point of the segment.
        point2 (np.array): Ending point of the segment.
        vertical (np.array): Vertical axis vector.
    
    Returns:
        float: Angle in degrees.
    """
    segment = point2 - point1
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
    Calculate upper body, lower body, and total body centroids.
    
    Args:
        coords (dict): Dictionary of landmark coordinates.
        waist_point (np.array): Coordinates of the waist point.
    
    Returns:
        dict: Dictionary containing centroids.
    """
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
    return total_mass * MASS_RATIOS[gender][segment]

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
    
    # 1. Body Segment Angles (9 features)
    # a. Trunk angle: angle between spine vector and vertical axis
    hip_center = (coords['left_hip'] + coords['right_hip']) / 2
    shoulder_center = (coords['left_shoulder'] + coords['right_shoulder']) / 2
    spine_vector = shoulder_center - hip_center
    trunk_angle = calculate_segment_angle(hip_center, shoulder_center)
    features['trunk_angle'] = trunk_angle
    
    # b. Upper arm angles (left and right)
    upper_arm_l = calculate_segment_angle(coords['left_shoulder'], coords['left_elbow'])
    upper_arm_r = calculate_segment_angle(coords['right_shoulder'], coords['right_elbow'])
    features['upper_arm_left_angle'] = upper_arm_l
    features['upper_arm_right_angle'] = upper_arm_r
    
    # c. Forearm angles (left and right)
    forearm_l = calculate_segment_angle(coords['left_elbow'], coords['left_wrist'])
    forearm_r = calculate_segment_angle(coords['right_elbow'], coords['right_wrist'])
    features['forearm_left_angle'] = forearm_l
    features['forearm_right_angle'] = forearm_r
    
    # d. Thigh angles (left and right)
    thigh_l = calculate_segment_angle(coords['left_hip'], coords['left_knee'])
    thigh_r = calculate_segment_angle(coords['right_hip'], coords['right_knee'])
    features['thigh_left_angle'] = thigh_l
    features['thigh_right_angle'] = thigh_r
    
    # e. Shank angles (left and right)
    shank_l = calculate_segment_angle(coords['left_knee'], coords['left_ankle'])
    shank_r = calculate_segment_angle(coords['right_knee'], coords['right_ankle'])
    features['shank_left_angle'] = shank_l
    features['shank_right_angle'] = shank_r
    
    # 2. Centroid Locations (9 features)
    waist_point = find_waist_point(coords, alpha=0.5)
    centroids = calculate_centroids(coords, waist_point)
    features['C_upper_x'] = centroids['C_upper'][0]
    features['C_upper_y'] = centroids['C_upper'][1]
    features['C_upper_z'] = centroids['C_upper'][2]
    
    features['C_lower_x'] = centroids['C_lower'][0]
    features['C_lower_y'] = centroids['C_lower'][1]
    features['C_lower_z'] = centroids['C_lower'][2]
    
    features['C_total_x'] = centroids['C_total'][0]
    features['C_total_y'] = centroids['C_total'][1]
    features['C_total_z'] = centroids['C_total'][2]
    
    # 3. Yaw Trunk Angle (1 feature)
    yaw_trunk = calculate_yaw_angle(coords['left_shoulder'], coords['right_shoulder'])
    features['yaw_trunk_angle'] = yaw_trunk
    
    # 4. Hip and Shoulder Coordinates (12 features)
    key_joints = ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']
    for joint in key_joints:
        features[f'{joint}_x'] = coords[joint][0]
        features[f'{joint}_y'] = coords[joint][1]
        features[f'{joint}_z'] = coords[joint][2]
    
    # 5. Joint Angles (14 features)
    # a. Elbow angles (left and right)
    elbow_l = calculate_joint_angle(coords['left_shoulder'], coords['left_elbow'], coords['left_wrist'])
    elbow_r = calculate_joint_angle(coords['right_shoulder'], coords['right_elbow'], coords['right_wrist'])
    features['elbow_left_angle'] = elbow_l
    features['elbow_right_angle'] = elbow_r
    
    # b. Shoulder angles (left and right)
    shoulder_l = calculate_joint_angle(hip_center, coords['left_shoulder'], coords['left_elbow'])
    shoulder_r = calculate_joint_angle(hip_center, coords['right_shoulder'], coords['right_elbow'])
    features['shoulder_left_angle'] = shoulder_l
    features['shoulder_right_angle'] = shoulder_r
    
    # c. Hip angles (left and right)
    hip_l = calculate_joint_angle(hip_center, coords['left_hip'], coords['left_knee'])
    hip_r = calculate_joint_angle(hip_center, coords['right_hip'], coords['right_knee'])
    features['hip_left_angle'] = hip_l
    features['hip_right_angle'] = hip_r
    
    # d. Knee angles (left and right)
    knee_l = calculate_joint_angle(coords['left_hip'], coords['left_knee'], coords['left_ankle'])
    knee_r = calculate_joint_angle(coords['right_hip'], coords['right_knee'], coords['right_ankle'])
    features['knee_left_angle'] = knee_l
    features['knee_right_angle'] = knee_r
    
    # e. Ankle angles (left and right)
    ankle_l = calculate_joint_angle(coords['left_knee'], coords['left_ankle'], coords['left_heel'])
    ankle_r = calculate_joint_angle(coords['right_knee'], coords['right_ankle'], coords['right_heel'])
    features['ankle_left_angle'] = ankle_l
    features['ankle_right_angle'] = ankle_r
    
    # f. Wrist angles (left and right)
    wrist_l = calculate_joint_angle(coords['left_elbow'], coords['left_wrist'], coords['left_pinky'])
    wrist_r = calculate_joint_angle(coords['right_elbow'], coords['right_wrist'], coords['right_pinky'])
    features['wrist_left_angle'] = wrist_l
    features['wrist_right_angle'] = wrist_r
    
    # g. Foot angles (left and right)
    foot_l = calculate_joint_angle(coords['left_ankle'], coords['left_heel'], coords['left_foot_index'])
    foot_r = calculate_joint_angle(coords['right_ankle'], coords['right_heel'], coords['right_foot_index'])
    features['foot_left_angle'] = foot_l
    features['foot_right_angle'] = foot_r
    
    # 6. Body Segment Masses (8 features)
    # Calculate mass for each body segment
    segments = ['head', 'trunk', 'upper_arm', 'forearm', 'hand', 'thigh', 'shank', 'foot']
    for segment in segments:
        mass = calculate_segment_mass(total_mass, segment, gender)
        features[f'{segment}_mass'] = mass
    
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
    print(f"Processing {num_frames} frames...")
    
    # Initialize list to hold features for each frame
    features_list = []
    
    for index, row in df.iterrows():
        coords = extract_landmarks(row)
        
        # Handle cases where certain landmarks might be missing or have NaN values
        missing_landmarks = [name for name, coord in coords.items() if np.isnan(coord).any()]
        if missing_landmarks:
            print(f"Frame {index}: Missing landmarks {missing_landmarks}. Skipping frame.")
            continue
        
        # Extract biomechanical features
        features = extract_biomechanical_features(coords, total_mass, gender)
        features_list.append(features)
    
    # Create DataFrame from features
    features_df = pd.DataFrame(features_list, columns=FEATURE_NAMES + [f'{seg}_mass' for seg in ['head', 'trunk', 'upper_arm', 'forearm', 'hand', 'thigh', 'shank', 'foot']])
    
    # Save to CSV
    features_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    input_csv_path = 'pose_landmarks.csv'          # Replace with your input CSV file path
    output_csv_path = 'biomechanical_features.csv' # Desired output CSV file path
    total_body_mass = 70.0  # Example total body mass in kg
    gender = 'male'          # 'male' or 'female'
    
    process_csv(input_csv_path, output_csv_path, total_body_mass, gender)
