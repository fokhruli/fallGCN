import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# Importing the dataset
dataset = pd.read_csv('/home/rmedu/Fokhrul/fallGCN/fall_dataset_preprocessed/X_train_fall_biomechanical_features.csv', header = None)
label = pd.read_csv('/home/rmedu/Fokhrul/fallGCN/fall_dataset_preprocessed/y_train_fall.csv', header = None)
X = dataset.iloc[:,:].values
y = label.iloc[:,:].values

# X_flatten = []
# for i in range(y.shape[0]):
#     temp = X[90*i:90*(i+1),:].reshape(-1)
#     X_flatten.append(temp)

# X = np.array(X_flatten)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# sc1 = StandardScaler()
X = sc.fit_transform(X)
# y = sc1.fit_transform(y)

# Data processing
X_ = np.zeros((y.shape[0], 90*X.shape[1]))
for repitation in range(X_.shape[0]):
    temp = X[repitation*90:(repitation*90)+90,:]
    temp = np.reshape(temp,(1,-1))
    X_[repitation,:] = temp
X = X_

print(f"Data X size : {X.shape}")
print(f"Data y size : {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# Convert labels to long type and flatten them
y_train_tensor = torch.tensor(y_train, dtype=torch.long).squeeze()  # This makes it (N,)

# Determine the device to use (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Move tensors to the device
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)


# from sklearn.svm import SVC

# # Reshape data for SVM (flatten the sequences)
# X_train_svm = X_train.reshape(X_train.shape[0], -1)
# X_test_svm = X_test.reshape(X_test.shape[0], -1)

# # Instantiate the SVM classifier
# svm_clf = SVC(
#     kernel='rbf',        # Radial Basis Function kernel
#     C=1.0,                # Regularization parameter
#     class_weight='balanced',  # Handle class imbalance
#     probability=True      # Enable probability estimates
# )

# # Train the SVM
# svm_clf.fit(X_train_svm, y_train)

# # Predict on the test set
# y_pred_svm = svm_clf.predict(X_test_svm)
# y_pred_prob_svm = svm_clf.predict_proba(X_test_svm)[:, 1]

# # Evaluate the SVM
# precision_svm = precision_score(y_test, y_pred_svm, zero_division=0)
# recall_svm = recall_score(y_test, y_pred_svm, zero_division=0)
# f1_svm = f1_score(y_test, y_pred_svm, zero_division=0)
# auprc_svm = average_precision_score(y_test, y_pred_prob_svm)

# print("=== Support Vector Machine (SVM) Performance ===")
# print(f"Precision: {precision_svm:.4f}")
# print(f"Recall: {recall_svm:.4f}")
# print(f"F1-score: {f1_svm:.4f}")
# print(f"AUPRC: {auprc_svm:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_svm))

# from sklearn.ensemble import RandomForestClassifier

# # Flatten the data for Random Forest
# X_train_rf = X_train.reshape(X_train.shape[0], -1)
# X_test_rf = X_test.reshape(X_test.shape[0], -1)

# # Instantiate the Random Forest classifier
# rf_clf = RandomForestClassifier(
#     n_estimators=100,       # Number of trees
#     random_state=0,
#     class_weight='balanced',  # Handle class imbalance
#     n_jobs=-1               # Utilize all available cores
# )

# # Train the Random Forest
# rf_clf.fit(X_train_rf, y_train)

# # Predict on the test set
# y_pred_rf = rf_clf.predict(X_test_rf)
# y_pred_prob_rf = rf_clf.predict_proba(X_test_rf)[:, 1]

# # Evaluate the Random Forest
# precision_rf = precision_score(y_test, y_pred_rf, zero_division=0)
# recall_rf = recall_score(y_test, y_pred_rf, zero_division=0)
# f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
# auprc_rf = average_precision_score(y_test, y_pred_prob_rf)

# print("=== Random Forest Performance ===")
# print(f"Precision: {precision_rf:.4f}")
# print(f"Recall: {recall_rf:.4f}")
# print(f"F1-score: {f1_rf:.4f}")
# print(f"AUPRC: {auprc_rf:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_rf))

# import xgboost as xgb
# from sklearn.metrics import roc_auc_score

# # Flatten the data for XGBoost
# X_train_xgb = X_train.reshape(X_train.shape[0], -1)
# X_test_xgb = X_test.reshape(X_test.shape[0], -1)

# # Instantiate the XGBoost classifier
# xgb_clf = xgb.XGBClassifier(
#     objective='binary:logistic',
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=6,
#     scale_pos_weight=(len(y_train) - np.sum(y_train)) / np.sum(y_train),  # Handle imbalance
#     use_label_encoder=False,
#     eval_metric='logloss',
#     n_jobs=-1
# )

# # Train the XGBoost classifier
# xgb_clf.fit(X_train_xgb, y_train)

# # Predict on the test set
# y_pred_xgb = xgb_clf.predict(X_test_xgb)
# y_pred_prob_xgb = xgb_clf.predict_proba(X_test_xgb)[:, 1]

# # Evaluate XGBoost
# precision_xgb = precision_score(y_test, y_pred_xgb, zero_division=0)
# recall_xgb = recall_score(y_test, y_pred_xgb, zero_division=0)
# f1_xgb = f1_score(y_test, y_pred_xgb, zero_division=0)
# auprc_xgb = average_precision_score(y_test, y_pred_prob_xgb)
# roc_auc_xgb = roc_auc_score(y_test, y_pred_prob_xgb)

# print("=== XGBoost Performance ===")
# print(f"Precision: {precision_xgb:.4f}")
# print(f"Recall: {recall_xgb:.4f}")
# print(f"F1-score: {f1_xgb:.4f}")
# print(f"AUPRC: {auprc_xgb:.4f}")
# print(f"ROC AUC: {roc_auc_xgb:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_xgb))


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Prepare Data for 1D CNN
# PyTorch expects (batch_size, channels, seq_length)
X_train_cnn = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)  # (N, features, time_steps)
X_test_cnn = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)

y_train_cnn = torch.tensor(y_train, dtype=torch.long)
y_test_cnn = torch.tensor(y_test, dtype=torch.long)

# Create Datasets and DataLoaders
batch_size = 32
train_dataset_cnn = TensorDataset(X_train_cnn, y_train_cnn)
train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=True)

# 2. Define the 1D CNN Architecture
class CNN1D(nn.Module):
    def __init__(self, num_features, num_classes=2):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128 * (num_time_steps // 2), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 3. Instantiate the Model, Define Loss and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cnn = CNN1D(num_features=num_features).to(device)

criterion_cnn = nn.CrossEntropyLoss()
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)

# 4. Training Loop for 1D CNN
epochs = 50
for epoch in range(1, epochs + 1):
    model_cnn.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader_cnn:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer_cnn.zero_grad()
        outputs = model_cnn(batch_X)
        loss = criterion_cnn(outputs, batch_y)
        loss.backward()
        optimizer_cnn.step()
        
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(train_loader_cnn.dataset)
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}')

# 5. Evaluation for 1D CNN
model_cnn.eval()
with torch.no_grad():
    X_test_cnn = X_test_cnn.to(device)
    outputs = model_cnn(X_test_cnn)
    probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    y_pred_cnn = torch.argmax(outputs, dim=1).cpu().numpy()

# Calculate Metrics
precision_cnn = precision_score(y_test, y_pred_cnn, zero_division=0)
recall_cnn = recall_score(y_test, y_pred_cnn, zero_division=0)
f1_cnn = f1_score(y_test, y_pred_cnn, zero_division=0)
auprc_cnn = average_precision_score(y_test, probabilities)

print("=== 1D CNN Performance ===")
print(f"Precision: {precision_cnn:.4f}")
print(f"Recall: {recall_cnn:.4f}")
print(f"F1-score: {f1_cnn:.4f}")
print(f"AUPRC: {auprc_cnn:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_cnn))



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Prepare Data for LSTM
# PyTorch expects (batch_size, seq_length, features)
X_train_lstm = torch.tensor(X_train, dtype=torch.float32)
X_test_lstm = torch.tensor(X_test, dtype=torch.float32)

y_train_lstm = torch.tensor(y_train, dtype=torch.long)
y_test_lstm = torch.tensor(y_test, dtype=torch.long)

# Create Datasets and DataLoaders
batch_size = 32
train_dataset_lstm = TensorDataset(X_train_lstm, y_train_lstm)
train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=batch_size, shuffle=True)

# 2. Define the LSTM Architecture
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)
        
        # Take the output from the last time step
        out = out[:, -1, :]  # (batch_size, hidden_size)
        out = self.dropout(out)
        out = self.fc(out)    # (batch_size, num_classes)
        return out

# 3. Instantiate the Model, Define Loss and Optimizer
input_size = num_features
hidden_size = 128
num_layers = 1
num_classes = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_lstm = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)

criterion_lstm = nn.CrossEntropyLoss()
optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.001)

# 4. Training Loop for LSTM
epochs = 50
for epoch in range(1, epochs + 1):
    model_lstm.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader_lstm:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer_lstm.zero_grad()
        outputs = model_lstm(batch_X)
        loss = criterion_lstm(outputs, batch_y)
        loss.backward()
        optimizer_lstm.step()
        
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(train_loader_lstm.dataset)
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}')

# 5. Evaluation for LSTM
model_lstm.eval()
with torch.no_grad():
    X_test_lstm = X_test_lstm.to(device)
    outputs = model_lstm(X_test_lstm)
    probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    y_pred_lstm = torch.argmax(outputs, dim=1).cpu().numpy()

# Calculate Metrics
precision_lstm = precision_score(y_test, y_pred_lstm, zero_division=0)
recall_lstm = recall_score(y_test, y_pred_lstm, zero_division=0)
f1_lstm = f1_score(y_test, y_pred_lstm, zero_division=0)
auprc_lstm = average_precision_score(y_test, probabilities)

print("=== 1 LSTM Performance ===")
print(f"Precision: {precision_lstm:.4f}")
print(f"Recall: {recall_lstm:.4f}")
print(f"F1-score: {f1_lstm:.4f}")
print(f"AUPRC: {auprc_lstm:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lstm))
