import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
)
from data_processing import Data_Loader
from graph import Graph
from stgcn import SGCN_LSTM, SGCNLSTMTrainer
import pdb
import os
import logging
import sys

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_filename = 'training.log'
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
    logging.info(f'Found GPU: {gpu_name}')
else:
    logging.critical('GPU device not found. Exiting.')
    raise SystemError('GPU device not found')

# Load and prepare data
data_loader = Data_Loader()
graph = Graph(len(data_loader.body_part))

# Split data into training and validation sets with stratification
train_x, valid_x, train_y, valid_y = train_test_split(
    data_loader.scaled_x, 
    data_loader.scaled_y, 
    test_size=0.2, 
    random_state=RANDOM_SEED,
    stratify=data_loader.scaled_y  # Ensures balanced splits
)


logging.info(f"Training instances: {len(train_x)}")
logging.info(f"Validation instances: {len(valid_x)}")

# Check class distribution
unique, counts = np.unique(train_y, return_counts=True)
logging.info(f"Training class distribution: {dict(zip(unique, counts))}")
unique, counts = np.unique(valid_y, return_counts=True)
logging.info(f"Validation class distribution: {dict(zip(unique, counts))}")

# Initialize trainer with device
trainer = SGCNLSTMTrainer(
    train_x=train_x,
    train_y=train_y,
    valid_x=valid_x,
    valid_y=valid_y,
    adj=graph.AD,
    adj2=graph.AD2,
    adj3=graph.AD3,
    lr=0.0001, #0.0001,
    epochs=1000,
    batch_size=32,
    device=device.type
)

# Build and print model summary
logging.info("Model Architecture:")
logging.info(trainer.model)
total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
logging.info(f"Total trainable parameters: {total_params}")

# Train the modelfigures
history = trainer.train()

# Make predictions
y_pred_prob = trainer.predict(valid_x)

# Convert probabilities to binary predictions (threshold = 0.5)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Evaluation Metrics
precision = precision_score(valid_y, y_pred, zero_division=0)
recall = recall_score(valid_y, y_pred, zero_division=0)
f1 = f1_score(valid_y, y_pred, zero_division=0)
auprc = average_precision_score(valid_y, y_pred_prob)

logging.info(f'Precision: {precision:.4f}')
logging.info(f'Recall: {recall:.4f}')
logging.info(f'F1-score: {f1:.4f}')
logging.info(f'AUPRC: {auprc:.4f}')

def save_and_show(fig_dir, filename, dpi=300):
    plt.savefig(os.path.join(fig_dir, filename), dpi=dpi, bbox_inches='tight')
    plt.show()

# Create figures directory
fig_dir = 'figures'
os.makedirs(fig_dir, exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(valid_y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Fall', 'Fall'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
save_and_show(fig_dir, 'confusion_matrix.png')

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(valid_y, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, label=f'AUPRC = {auprc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
save_and_show(fig_dir, 'precision_recall_curve.png')

# ROC Curve
fpr, tpr, thresholds = roc_curve(valid_y, y_pred_prob)
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

# Plot AUPRC over epochs
plt.figure(figsize=(10, 5))
plt.plot(history['train_auprc'], label='Train AUPRC')
plt.plot(history['val_auprc'], label='Validation AUPRC')
plt.xlabel('Epoch')
plt.ylabel('AUPRC')
plt.title('AUPRC over Epochs')
plt.legend()
plt.grid(True)
save_and_show(fig_dir, 'auprc_over_epochs.png')

# Plot Loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)
save_and_show(fig_dir, 'loss_over_epochs.png')
