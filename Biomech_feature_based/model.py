import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Prepare Your Data
# Assuming X_train and y_train are already defined as NumPy arrays
# Replace the following lines with your actual data loading mechanism
# X_train = np.load('path_to_X_train.npy')
# y_train = np.load('path_to_y_train.npy')

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
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Ensure y has shape (N, 1)

# Determine the device to use (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Move tensors to the device
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)

# 2. Create a Dataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 32  # Adjust as needed
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 3. Define the Neural Network Architecture
class ANNRegressor(nn.Module):
    def __init__(self, input_dim):
        super(ANNRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 2)
            # No activation on output for regression
        )
    
    def forward(self, x):
        return self.network(x)

# 4. Instantiate the Model, Define Loss Function and Optimizer
input_dim = X_train.shape[1]
model = ANNRegressor(input_dim).to(device)

# Calculate class weights for weighted loss
class_counts = np.bincount(y_train.flatten())
num_classes = 2  # Adjust based on your problem
if len(class_counts) < num_classes:
    class_counts = np.append(class_counts, [0]*(num_classes - len(class_counts)))

# Calculate class weights: inverse frequency
class_weights = 1. / (class_counts + 1e-6)  # Add epsilon to prevent division by zero
class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1

class_weights = torch.FloatTensor(class_weights).to(device)
print(f'Class weights: {class_weights}')

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 5. Training Loop
epochs = 500
loss_history = []

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    
    avg_loss = epoch_loss / len(train_loader.dataset)
    loss_history.append(avg_loss)
    
    if epoch % 50 == 0 or epoch == 1:
        print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}')

# 6. Plotting the Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# 7. Making Predictions
model.eval()
with torch.no_grad():
    predictions = model(X_train_tensor).cpu().numpy()

# Example: Print first 5 predictions vs actual values
print("Predictions vs Actuals:")
for pred, actual in zip(predictions[:5], y_train[:5]):
    print(f'Predicted: {pred[0]:.4f}, Actual: {actual}')
