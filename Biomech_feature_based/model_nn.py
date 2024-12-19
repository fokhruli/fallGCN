import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import os
import logging

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
# Convert labels to long type and flatten them
y_train_tensor = torch.tensor(y_train, dtype=torch.long).squeeze()  # This makes it (N,)

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
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
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
epochs = 200
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

# # 6. Plotting the Loss Curve
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, epochs + 1), loss_history, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')
# plt.title('Training Loss Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.show()

# # 7. Making Predictions
# model.eval()
# with torch.no_grad():
#     predictions = model(X_train_tensor).cpu().numpy()

# # Example: Print first 5 predictions vs actual values
# print("Predictions vs Actuals:")
# for pred, actual in zip(predictions[:5], y_train[:5]):
#     print(f'Predicted: {pred[0]:.4f}, Actual: {actual}')



def evaluate_model(model, X_test_tensor, y_test, device, threshold=0.5):
    """
    Evaluate the model and return predictions and probabilities
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        y_pred_prob = probabilities[:, 1].cpu().numpy()  # Probability of positive class
        y_pred = (y_pred_prob >= threshold).astype(int)
        
    return y_pred, y_pred_prob

def calculate_metrics(y_true, y_pred, y_pred_prob):
    """
    Calculate various performance metrics
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auprc = average_precision_score(y_true, y_pred_prob)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auprc': auprc
    }

def save_and_show(fig_dir, filename, dpi=300):
    """
    Save and display the current figure
    """
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, filename), dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, fig_dir):
    """
    Plot and save confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Fall', 'Fall'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    save_and_show(fig_dir, 'confusion_matrix.png')

def plot_pr_curve(y_true, y_pred_prob, auprc, fig_dir):
    """
    Plot and save precision-recall curve
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, label=f'AUPRC = {auprc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    save_and_show(fig_dir, 'precision_recall_curve.png')

def plot_roc_curve(y_true, y_pred_prob, fig_dir):
    """
    Plot and save ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    save_and_show(fig_dir, 'roc_curve.png')

def plot_training_history(history, fig_dir):
    """
    Plot and save training history
    """
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    save_and_show(fig_dir, 'loss_over_epochs.png')

# Add this to your training loop to collect history
history = {
    'train_loss': [],
    'val_loss': [],
}

# After training, add this code for evaluation
fig_dir = 'figures_nn'
os.makedirs(fig_dir, exist_ok=True)

# Prepare validation data
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).squeeze()

# Get predictions
y_pred, y_pred_prob = evaluate_model(model, X_test_tensor, y_test, device)

# Calculate metrics
metrics = calculate_metrics(y_test.squeeze(), y_pred, y_pred_prob)

# Log results
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-score: {metrics['f1']:.4f}")
print(f"AUPRC: {metrics['auprc']:.4f}")

# Generate plots
plot_confusion_matrix(y_test.squeeze(), y_pred, fig_dir)
plot_pr_curve(y_test.squeeze(), y_pred_prob, metrics['auprc'], fig_dir)
plot_roc_curve(y_test.squeeze(), y_pred_prob, fig_dir)
plot_training_history(history, fig_dir)