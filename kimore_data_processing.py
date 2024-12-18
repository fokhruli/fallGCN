import pandas as pd
import numpy as np
#import tensorflow as tf
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from scipy import signal
#from sklearn.model_selection import train_test_split
#from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM, Lambda, UpSampling1D
#from tensorflow.keras.models import Model
#from tensorflow.keras.optimizers import *
#from sklearn.model_selection import train_test_split
#from IPython.core.debugger import set_trace
#import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from math import sqrt
#from scipy import signal
#from sklearn.metrics import mean_squared_error

index_Spine_Base=0
index_Spine_Mid=4
index_Neck=8
index_Head=12   # no orientation
index_Shoulder_Left=16
index_Elbow_Left=20
index_Wrist_Left=24
index_Hand_Left=28
index_Shoulder_Right=32
index_Elbow_Right=36
index_Wrist_Right=40
index_Hand_Right=44
index_Hip_Left=48
index_Knee_Left=52
index_Ankle_Left=56
index_Foot_Left=60  # no orientation    
index_Hip_Right=64
index_Knee_Right=68
index_Ankle_Right=72
index_Foot_Right=76   # no orientation
index_Spine_Shoulder=80
index_Tip_Left=84     # no orientation
index_Thumb_Left=88   # no orientation
index_Tip_Right=92    # no orientation
index_Thumb_Right=96  # no orientation

body_parts = [index_Spine_Base, index_Spine_Mid, index_Neck, index_Head, index_Shoulder_Left, index_Elbow_Left, index_Wrist_Left, index_Hand_Left, index_Shoulder_Right, index_Elbow_Right, index_Wrist_Right, index_Hand_Right, index_Hip_Left, index_Knee_Left, index_Ankle_Left, index_Foot_Left, index_Hip_Right, index_Knee_Right, index_Ankle_Right, index_Ankle_Right, index_Spine_Shoulder, index_Tip_Left, index_Thumb_Left, index_Tip_Right, index_Thumb_Right
]

num_repitation = 5        
dataset = []
sequence_length = []

for i in range(1,18):  # expert subject
    if i == 13:        
        continue
    else:
        po_features = pd.read_csv("E:/4-1/final_project/dataset/kimore/Exercise05/PO_feature_E_ID"+str(i)+".csv", header = None).iloc[:,:].values
        cf_features = pd.read_csv("E:/4-1/final_project/dataset/kimore/Exercise05/CF_feature_E_ID"+str(i)+".csv", header = None).iloc[:,:].values
        po_features = po_features[:,0:-1]
        cf_features = cf_features[:,0:-1]
        features = np.concatenate((po_features,cf_features),axis=1)
        label1 = pd.read_excel("E:/4-1/final_project/dataset/kimore/Exercise05/ClinicalAssessment_E_ID"+str(i)+".xlsx", sheet_name=None)
        label1 = label1['Foglio1'].iloc[:,:].values[0,1]
        label2 = np.zeros((num_repitation,1))
        label2[0:num_repitation,:] = label1
        label2_sum = np.sum(label2)
        if np.isnan(label2_sum) == True:
            continue
        else:
            dataset.append([features, label2])
            
for i in range(1,28):  # not expert subject
    if i == 11 or i == 17:
        continue
    else:
        po_features = pd.read_csv("E:/4-1/final_project/dataset/kimore/Exercise05/PO_feature_NE_ID"+str(i)+".csv", header = None).iloc[:,:].values
        cf_features = pd.read_csv("E:/4-1/final_project/dataset/kimore/Exercise05//CF_feature_NE_ID"+str(i)+".csv", header = None).iloc[:,:].values
        po_features = po_features[:,0:-1]
        cf_features = cf_features[:,0:-1]
        features = np.concatenate((po_features,cf_features),axis=1)
        label1 = pd.read_excel("E:/4-1/final_project/dataset/kimore/Exercise05/ClinicalAssessment_NE_ID"+str(i)+".xlsx", sheet_name=None)
        label1 = label1['Foglio1'].iloc[:,:].values[0,1]
        label2 = np.zeros((num_repitation,1))
        label2[0:num_repitation,:] = label1
        label2_sum = np.sum(label2)
        if np.isnan(label2_sum) == True:
            continue
        else:
            dataset.append([features, label2])

for i in range(1,17):  # parkinson patient
    if i == 7 or i == 15:
        continue
    else:
        po_features = pd.read_csv("E:/4-1/final_project/dataset/kimore/Exercise05/PO_feature_P_ID"+str(i)+".csv", header = None).iloc[:,:].values
        cf_features = pd.read_csv("E:/4-1/final_project/dataset/kimore/Exercise05/CF_feature_P_ID"+str(i)+".csv", header = None).iloc[:,:].values
        po_features = po_features[:,0:-1]
        cf_features = cf_features[:,0:-1]
        features = np.concatenate((po_features,cf_features),axis=1)
        label1 = pd.read_excel("E:/4-1/final_project/dataset/kimore/Exercise05/ClinicalAssessment_P_ID"+str(i)+".xlsx", sheet_name=None)
        label1 = label1['Foglio1'].iloc[:,:].values[0,1]
        label2 = np.zeros((num_repitation,1))
        label2[0:num_repitation,:] = label1
        label2_sum = np.sum(label2)
    if np.isnan(label2_sum) == True:
        continue
    else:
        dataset.append([features, label2])

for i in range(1,11):  # stroke patient

    po_features = pd.read_csv("E:/4-1/final_project/dataset/kimore/Exercise05/PO_feature_S_ID"+str(i)+".csv", header = None).iloc[:,:].values
    cf_features = pd.read_csv("E:/4-1/final_project/dataset/kimore/Exercise05/CF_feature_S_ID"+str(i)+".csv", header = None).iloc[:,:].values
    po_features = po_features[:,0:-1]
    cf_features = cf_features[:,0:-1]
    features = np.concatenate((po_features,cf_features),axis=1)
    label1 = pd.read_excel("E:/4-1/final_project/dataset/kimore/Exercise05/ClinicalAssessment_S_ID"+str(i)+".xlsx", sheet_name=None)
    label1 = label1['Foglio1'].iloc[:,:].values[0,1]
    label2 = np.zeros((num_repitation,1))
    label2[0:num_repitation,:] = label1
    label2_sum = np.sum(label2)
    if np.isnan(label2_sum) == True:
        continue
    else:
        dataset.append([features, label2])
            
for i in range(1,7):  # backpain patient
    
    po_features = pd.read_csv("E:/4-1/final_project/dataset/kimore/Exercise05/PO_feature_B_ID"+str(i)+".csv", header = None).iloc[:,:].values
    cf_features = pd.read_csv("E:/4-1/final_project/dataset/kimore/Exercise05/CF_feature_B_ID"+str(i)+".csv", header = None).iloc[:,:].values
    po_features = po_features[:,0:-1]
    cf_features = cf_features[:,0:-1]
    features = np.concatenate((po_features,cf_features),axis=1)
    label1 = pd.read_excel("E:/4-1/final_project/dataset/kimore/Exercise05/ClinicalAssessment_B_ID"+str(i)+".xlsx", sheet_name=None)
    label1 = label1['Foglio1'].iloc[:,:].values[0,1]
    label2 = np.zeros((num_repitation,1))
    label2[0:num_repitation,:] = label1
    label2_sum = np.sum(label2)
    if np.isnan(label2_sum) == True:
        continue
    else:
        dataset.append([features, label2])
            
num_timestep = 100
new_label = []

for i in range(len(dataset)):
    seq_length = dataset[i][0].shape[0]//num_timestep
    for repitation in range(seq_length):
        data = dataset[i][0][(num_timestep*repitation):(num_timestep*(repitation+1))]
        if i == 0 and repitation == 0:
            new_dataset = data
        else:
            new_dataset = np.concatenate((new_dataset,data))
        new_label.append(dataset[i][1][0,0])
new_label = np.array(new_label)    

y_train = np.reshape(new_label,(-1,1))

np.savetxt("Train_X.csv", new_dataset, delimiter=",")
np.savetxt("Train_Y.csv", y_train, delimiter=",")
