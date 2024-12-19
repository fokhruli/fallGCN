import os
import sys
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from STGCN_LSTM_Biomechanics.data_processing import Data_Loader
from STGCN_LSTM_Biomechanics.graph import Graph
from STGCN_LSTM_Biomechanics.stgcn import SGCN_LSTM_Fused  # Import your model class
import joblib

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_filename = 'inference.log'
log_path = os.path.join(log_dir, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    logging.info(f'Using GPU: {gpu_name}')
else:
    logging.info('Using CPU')

# Initialize Data Loader
data_loader = Data_Loader()

# Load the graph adjacency matrices
graph = Graph(len(data_loader.body_part))


# Define model parameters
# Replace these placeholders with your actual model parameters
input_dim_pose = 100        # Example value; replace with actual
input_dim_biomech = 50      # Example value; replace with actual
hidden_dim = 128            # Example value; replace with actual
num_classes = 1             # Binary classification
adj = np.eye(10)            # Example adjacency matrix; replace with actual
adj2 = np.eye(10)           # Replace with actual
adj3 = np.eye(10)           # Replace with actual

# Instantiate the model
model = SGCN_LSTM_Fused(
    input_shape=(None,90,33,3),
    biomech_dim = 44,
    # hidden_dim=hidden_dim,
    # num_classes=2,
    adj_matrix=graph.AD,
    adj_matrix2=graph.AD2,
    adj_matrix3=graph.AD3,
    device=device
)

# Move the model to the appropriate device
model.to(device)
print(model)
# Load the saved state dictionary
best_model_path = os.path.join('models', 'best_model_AttFusion.pth')

try:
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded successfully from {best_model_path}')
except Exception as e:
    logging.error(f'Error loading the model: {e}')
    sys.exit(1)

# Set the model to evaluation mode
model.eval()

# Initialize Data Loader
data_loader = Data_Loader()

# Load the graph adjacency matrices
graph = Graph(len(data_loader.body_part))


# Split data into training and validation sets with stratification
train_pose_x, valid_pose_x, train_biomech_x, valid_biomech_x, train_y, valid_y = train_test_split(
    data_loader.scaled_pose_x,
    data_loader.scaled_biomech_x,
    data_loader.scaled_y,
    test_size=0.2,
    random_state=42,
    stratify=data_loader.scaled_y
)

logging.info(f"Validation instances: {len(valid_pose_x)}")

def predict(model, pose_data, biomech_data, device, batch_size=32):
    """
    Perform inference using the trained model.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        pose_data (np.ndarray): Pose features.
        biomech_data (np.ndarray): Biomechanical features.
        device (torch.device): Device to perform computation on.
        batch_size (int): Batch size for inference.

    Returns:
        np.ndarray: Predicted probabilities.
    """
    model.eval()  # Ensure model is in evaluation mode
    y_pred_prob = []

    # Convert data to tensors
    pose_tensor = torch.tensor(pose_data, dtype=torch.float32).to(device)
    biomech_tensor = torch.tensor(biomech_data, dtype=torch.float32).to(device)

    # Create DataLoader for batching
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(pose_tensor, biomech_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_pose, batch_biomech in dataloader:
            # Forward pass
            outputs = model(batch_pose, batch_biomech)
            # Assuming the model outputs logits; apply sigmoid for binary classification
            probs = torch.softmax(outputs, dim=1)[:, 1]
            y_pred_prob.extend(probs.cpu().numpy())
            # y_pred_prob.extend(probs.flatten())

    return np.array(y_pred_prob)

# Perform inference on validation data
y_pred_prob = predict(model, valid_pose_x, valid_biomech_x, device)

# Convert probabilities to binary predictions (threshold = 0.5)
y_pred = (y_pred_prob >= 0.5).astype(int)

# import pdb
# pdb.set_trace()
# Evaluation Metrics
precision = precision_score(valid_y, y_pred, zero_division=0)
recall = recall_score(valid_y, y_pred, zero_division=0)
f1 = f1_score(valid_y, y_pred, zero_division=0)
auprc = average_precision_score(valid_y, y_pred_prob)
fpr, tpr, thresholds = roc_curve(valid_y, y_pred_prob)
roc_auc = auc(fpr, tpr)

logging.info(f'Precision: {precision:.4f}')
logging.info(f'Recall: {recall:.4f}')
logging.info(f'F1-score: {f1:.4f}')
logging.info(f'AUPRC: {auprc:.4f}')
logging.info(f'ROC AUC: {roc_auc:.4f}')

def save_and_close(fig_dir, filename, dpi=300):
    plt.savefig(os.path.join(fig_dir, filename), dpi=dpi, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

# Create figures directory
fig_dir = 'figures'
os.makedirs(fig_dir, exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(valid_y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Fall', 'Fall'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
save_and_close(fig_dir, 'confusion_matrix.png')
logging.info('Confusion Matrix saved.')

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(valid_y, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f'AUPRC = {auprc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
save_and_close(fig_dir, 'precision_recall_curve.png')
logging.info('Precision-Recall Curve saved.')

# ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
save_and_close(fig_dir, 'roc_curve.png')
logging.info('ROC Curve saved.')
