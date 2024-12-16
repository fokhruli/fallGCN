import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import signal
# from IPython.core.debugger import set_trace
# import ipdb 

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
   
    
body_parts = [nose, left_eye_inner, left_eye, left_eye_outer, right_eye_inner, right_eye, right_eye_outer, left_ear, right_ear, mouth_left, mouth_right, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_pinky, right_pinky, left_index, right_index, left_thumb, right_thumb, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_heel, right_heel, left_foot_index, right_foot_index]

class Data_Loader():
    def __init__(self):
        self.num_repitation = 5
        self.num_channel = 3
        self.dir = dir
        self.body_part = self.body_parts()       
        self.dataset = []
        self.sequence_length = []
        self.num_timestep = 100
        self.new_label = []
        self.train_x, self.train_y= self.import_dataset()
        self.batch_size = self.train_y.shape[0]
        self.num_joints = len(self.body_part)
        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()
        self.scaled_x, self.scaled_y = self.preprocessing()
                
    def body_parts(self):
        body_parts = [nose, left_eye_inner, left_eye, left_eye_outer, right_eye_inner, right_eye, right_eye_outer, left_ear, right_ear, mouth_left, mouth_right, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_pinky, right_pinky, left_index, right_index, left_thumb, right_thumb, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_heel, right_heel, left_foot_index, right_foot_index]
        return body_parts
    
    def import_dataset(self):
        train_x = pd.read_csv("Train_X_openpose.csv", header = None).iloc[:,:].values
        train_y = pd.read_csv("Train_Y_openpose.csv", header = None).iloc[:,:].values
        return train_x, train_y
            
    def preprocessing(self):
        X_train = np.zeros((self.train_x.shape[0],self.num_joints*self.num_channel)).astype('float32')
        for row in range(self.train_x.shape[0]):
            counter = 0
            for parts in self.body_part:
                for i in range(self.num_channel):
                    X_train[row, counter+i] = self.train_x[row, parts+i]
                counter += self.num_channel 
        
        y_train = np.reshape(self.train_y,(-1,1))
        X_train = self.sc1.fit_transform(X_train)         
        y_train = self.sc2.fit_transform(y_train)
        
        X_train_ = np.zeros((self.batch_size, self.num_timestep, self.num_joints, self.num_channel))
        
        for batch in range(X_train_.shape[0]):
            for timestep in range(X_train_.shape[1]):
                for node in range(X_train_.shape[2]):
                    for channel in range(X_train_.shape[3]):
                        X_train_[batch,timestep,node,channel] = X_train[timestep+(batch*self.num_timestep),channel+(node*self.num_channel)]
        
                        
        X_train = X_train_                
        return X_train, y_train