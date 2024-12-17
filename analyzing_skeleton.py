import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import cv2

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

# Define landmark connections to form the skeleton
SKELETON_CONNECTIONS = [
    ('nose', 'left_eye_inner'),
    ('left_eye_inner', 'left_eye'),
    ('left_eye', 'left_eye_outer'),
    ('nose', 'right_eye_inner'),
    ('right_eye_inner', 'right_eye'),
    ('right_eye', 'right_eye_outer'),
    ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear'),
    ('nose', 'mouth_left'),
    ('nose', 'mouth_right'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
    ('left_ankle', 'left_heel'),
    ('left_heel', 'left_foot_index'),
    ('right_ankle', 'right_heel'),
    ('right_heel', 'right_foot_index'),
    ('left_wrist', 'left_pinky'),
    ('left_wrist', 'left_index'),
    ('left_wrist', 'left_thumb'),
    ('right_wrist', 'right_pinky'),
    ('right_wrist', 'right_index'),
    ('right_wrist', 'right_thumb')
]

def load_pose_data(csv_file):
    """
    Load pose landmark data from a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the pose data.
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

def plot_skeleton(coords, connections=SKELETON_CONNECTIONS, title="Skeleton"):
    """
    Plot the skeleton based on landmark coordinates.
    
    Args:
        coords (dict): Dictionary mapping landmark names to their (x, y, z) coordinates.
        connections (list): List of tuples defining connections between landmarks.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 10))
    
    # Extract x and y coordinates
    x = [coords[landmark][0] for landmark in coords]
    y = [coords[landmark][1] for landmark in coords]
    
    # Scatter plot of landmarks
    for landmark, coord in coords.items():
        plt.scatter(coord[0], coord[1], s=50)
        plt.text(coord[0], coord[1], landmark, fontsize=9)
    
    # Draw connections
    for connection in connections:
        start, end = connection
        if start in coords and end in coords:
            plt.plot(
                [coords[start][0], coords[end][0]],
                [coords[start][1], coords[end][1]],
                'r-'
            )
    
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.axis('equal')
    plt.show()

def initialize_plot():
    """
    Initialize the plot for animation.
    
    Returns:
        fig, ax: Matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 1)  # Assuming normalized coordinates; adjust as needed
    ax.set_ylim(1, 0)  # Inverted y-axis
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Skeleton Animation')
    return fig, ax

def animate_skeleton(frame_idx, pose_df, connections, ax):
    """
    Update function for animation.
    
    Args:
        frame_idx (int): Current frame index.
        pose_df (pd.DataFrame): DataFrame containing pose data.
        connections (list): List of tuples defining connections between landmarks.
        ax (matplotlib.axes.Axes): Matplotlib axes.
    
    Returns:
        None
    """
    ax.cla()  # Clear the axes
    ax.set_xlim(0, 1)  # Adjust based on data
    ax.set_ylim(1, 0)  # Inverted y-axis
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Skeleton Animation - Frame {frame_idx}')
    
    # Extract landmarks for the current frame
    frame_row = pose_df.iloc[frame_idx]
    landmarks = extract_landmarks(frame_row)
    
    # Plot landmarks
    for landmark, coord in landmarks.items():
        ax.scatter(coord[0], coord[1], s=50)
        ax.text(coord[0], coord[1], landmark, fontsize=8)
    
    # Plot connections
    for connection in connections:
        start, end = connection
        if start in landmarks and end in landmarks:
            x_values = [landmarks[start][0], landmarks[end][0]]
            y_values = [landmarks[start][1], landmarks[end][1]]
            ax.plot(x_values, y_values, 'r-')

def animate_multiple_frames(pose_df, connections=SKELETON_CONNECTIONS, interval=100):
    """
    Animate the skeleton across multiple frames.
    
    Args:
        pose_df (pd.DataFrame): DataFrame containing pose data.
        connections (list): List of tuples defining connections between landmarks.
        interval (int): Time between frames in milliseconds.
    
    Returns:
        None
    """
    fig, ax = initialize_plot()
    
    num_frames = len(pose_df)
    
    ani = animation.FuncAnimation(
        fig, 
        animate_skeleton, 
        frames=num_frames, 
        fargs=(pose_df, connections, ax),
        interval=interval,
        repeat=False
    )
    
    plt.show()

def draw_skeleton_opencv(pose_df, connections=SKELETON_CONNECTIONS, image_size=(640, 480), save_video=False, output_file='skeleton_visualization.avi'):
    """
    Draw the skeleton on video frames using OpenCV.
    
    Args:
        pose_df (pd.DataFrame): DataFrame containing pose data.
        connections (list): List of tuples defining connections between landmarks.
        image_size (tuple): Size of the image (width, height).
        save_video (bool): Whether to save the visualization as a video file.
        output_file (str): Output video file name.
    
    Returns:
        None
    """
    frame_width, frame_height = image_size
    fps = 10  # Adjust as needed
    
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    for index, row in pose_df.iterrows():
        # Create a blank image
        image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Extract landmarks
        coords = extract_landmarks(row)
        
        # Scale landmarks to image size
        scaled_coords = {}
        for landmark, coord in coords.items():
            # Assuming the coordinates are normalized between 0 and 1
            x = int(coord[0] * frame_width)
            y = int(coord[1] * frame_height)
            scaled_coords[landmark] = (x, y)
        
        # Draw landmarks
        for landmark, point in scaled_coords.items():
            cv2.circle(image, point, 5, (0, 255, 0), -1)
            cv2.putText(image, landmark, point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw connections
        for connection in connections:
            start, end = connection
            if start in scaled_coords and end in scaled_coords:
                cv2.line(image, scaled_coords[start], scaled_coords[end], (0, 0, 255), 2)
        
        # Write the frame to the video
        if save_video:
            out.write(image)
        
        # Optional: Display the frame in real-time
        cv2.imshow('Skeleton', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if save_video:
        out.release()
        print(f"Skeleton visualization video saved as '{output_file}'.")
    
    cv2.destroyAllWindows()

# Example Usage
if __name__ == "__main__":
    # Load the pose data
    pose_df = load_pose_data("fall_dataset_preprocessed/X_train_fall.csv")
    
    # Choose a mode: 'static', 'animation', 'opencv'
    mode = 'animation'  # Change to 'animation' or 'opencv' as needed
    
    if mode == 'static':
        # Select a frame to visualize (e.g., the first frame)
        frame_index = 0  # Change this index to visualize different frames
        frame_row = pose_df.iloc[frame_index]
        
        # Extract landmarks
        landmarks = extract_landmarks(frame_row)
        
        # Plot the skeleton
        plot_skeleton(landmarks, title=f"Skeleton Frame {frame_index}")
    
    elif mode == 'animation':
        # Animate the skeleton across frames
        animate_multiple_frames(pose_df, connections=SKELETON_CONNECTIONS, interval=100)
    
    elif mode == 'opencv':
        # Draw the skeleton on video frames using OpenCV
        draw_skeleton_opencv(
            pose_df, 
            connections=SKELETON_CONNECTIONS, 
            image_size=(640, 480), 
            save_video=True, 
            output_file='skeleton_visualization.avi'
        )
    
    else:
        print("Invalid mode selected. Choose 'static', 'animation', or 'opencv'.")
