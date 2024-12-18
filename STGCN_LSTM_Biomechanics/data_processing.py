import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pdb

# Keep all the original body part indices
nose = 0
left_eye_inner = 3
left_eye = 6
left_eye_outer = 9
right_eye_inner = 12
right_eye = 15
right_eye_outer = 18
left_ear = 21
right_ear = 24
mouth_left = 27
mouth_right = 30
left_shoulder = 33
right_shoulder = 36
left_elbow = 39
right_elbow = 42
left_wrist = 45
right_wrist = 48
left_pinky = 51
right_pinky = 54
left_index = 57
right_index = 60
left_thumb = 63
right_thumb = 66
left_hip = 69
right_hip = 72
left_knee = 75
right_knee = 78
left_ankle = 81
right_ankle = 84
left_heel = 87
right_heel = 90
left_foot_index = 93
right_foot_index = 96

# Keep original body_parts list
body_parts = [
    nose, left_eye_inner, left_eye, left_eye_outer, right_eye_inner, right_eye, 
    right_eye_outer, left_ear, right_ear, mouth_left, mouth_right, left_shoulder, 
    right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_pinky, 
    right_pinky, left_index, right_index, left_thumb, right_thumb, left_hip, 
    right_hip, left_knee, right_knee, left_ankle, right_ankle, left_heel, 
    right_heel, left_foot_index, right_foot_index
]

class Data_Loader():
    def __init__(self):
        self.num_repitation = 5
        self.num_channel = 3
        self.body_part = self.body_parts()       
        self.sequence_length = []
        self.num_timestep = 90  # Number of timesteps per sequence
        self.train_x, self.train_biomech, self.train_y = self.import_dataset()
        self.num_joints = len(self.body_part)
        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()  # Additional scaler for biomechanical features
        self.scaled_pose_x, self.scaled_biomech_x, self.scaled_y = self.preprocessing()
                
    def body_parts(self):
        return body_parts
    
    def import_dataset(self):
        # Load features and labels from CSV files
        train_x = pd.read_csv("/home/rmedu/Fokhrul/fallGCN/fall_dataset_preprocessed/X_train_fall.csv", header=None).values
        train_biomech = pd.read_csv("/home/rmedu/Fokhrul/fallGCN/fall_dataset_preprocessed/X_train_fall_biomechanical_features.csv", header=None).values
        train_y = pd.read_csv("/home/rmedu/Fokhrul/fallGCN/fall_dataset_preprocessed/y_train_fall.csv", header=None).values
        return train_x, train_biomech, train_y
            
    def preprocessing(self):
        # Process pose data
        X_train = np.zeros((self.train_x.shape[0], self.num_joints * self.num_channel), dtype='float32')
        
        # Populate X_train with joint coordinates
        for row in range(self.train_x.shape[0]):
            counter = 0
            for part in self.body_part:
                for i in range(self.num_channel):
                    X_train[row, counter + i] = self.train_x[row, part + i]
                counter += self.num_channel 
        
        # Process biomechanical features
        X_biomech = self.train_biomech
        
        # Process labels
        y_train = self.train_y.flatten()
        unique_labels = np.unique(y_train)
        print("Unique labels before processing:", unique_labels)
        
        # Ensure labels are binary: 0 and 1
        if not set(unique_labels).issubset({0, 1}):
            y_train = np.where(y_train > 0, 1, 0)
            print("Unique labels after binarization:", np.unique(y_train))
        else:
            print("Labels are already binary.")
        
        # Scale features
        X_train = self.sc1.fit_transform(X_train)
        X_biomech = self.sc2.fit_transform(X_biomech)
        
        # Calculate total_samples as the number of full sequences
        total_samples = X_train.shape[0] // self.num_timestep
        print(f"Total samples: {total_samples}")
        
        # Slice data to include only full sequences
        X_train = X_train[:total_samples * self.num_timestep]
        X_biomech = X_biomech[:total_samples * self.num_timestep]
        y_train = y_train[:total_samples * self.num_timestep]
        
        print(f"Sliced X_train shape: {X_train.shape}")
        print(f"Sliced X_biomech shape: {X_biomech.shape}")
        print(f"Sliced y_train shape: {y_train.shape}")
        
        # Reshape X_train to [samples, timesteps, joints, channels]
        X_train = X_train.reshape(total_samples, self.num_timestep, self.num_joints, self.num_channel)
        
        # Reshape X_biomech to [samples, timesteps, biomech_features]
        num_biomech_features = X_biomech.shape[1]
        X_biomech = X_biomech.reshape(total_samples, self.num_timestep, num_biomech_features)
        
        # Reshape y_train to [samples, timesteps]
        y_train = y_train.reshape(total_samples, 1)
        
        print(f"Reshaped X_train shape: {X_train.shape}")
        print(f"Reshaped X_biomech shape: {X_biomech.shape}")
        print(f"Reshaped y_train shape: {y_train.shape}")
        
        return X_train, X_biomech, y_train