import os
import sys
import argparse
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from huggingface_hub import hf_hub_download

from STGCN_LSTM_Biomechanics.data_processing import Data_Loader as data_fuse
from STGCN_LSTM_Biomechanics.graph import Graph as graph_fuse
from STGCN_LSTM_Biomechanics.stgcn import SGCN_LSTM_Fused

from STGCN_LSTM.data_processing import Data_Loader as data_vanilla
from STGCN_LSTM.graph import Graph as graph_vanilla
from STGCN_LSTM.stgcn import SGCN_LSTM

# --------------------- Configuration ---------------------

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

# --------------------- Argument Parsing ---------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference Script for STGCN_LSTM_Biomechanics Models")
    parser.add_argument(
        '--model',
        type=str,
        choices=['vanilla', 'fusion'],
        required=True,
        help="Choose which model to use: 'vanilla' for skeleton data only, 'fusion' for skeleton and biomechanics data."
    )
    parser.add_argument(
        '--use_cuda',
        action='store_true',
        help="Flag to force the use of CUDA if available."
    )
    return parser.parse_args()

# --------------------- Device Configuration ---------------------

def get_device(use_cuda):
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f'Using GPU: {gpu_name}')
    else:
        logging.info('Using CPU')
    return device

# --------------------- Model Loading ---------------------

def download_model(model_filename, model_repo):
    """
    Downloads the model file from Hugging Face Hub.
    
    Args:
        model_filename (str): The filename of the model to download.
        model_repo (str): The Hugging Face repository name.
    
    Returns:
        str: The local path to the downloaded model file.
    """
    try:
        model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
        logging.info(f'Model {model_filename} downloaded successfully from {model_repo}.')
        return model_path
    except Exception as e:
        logging.error(f'Error downloading the model {model_filename} from {model_repo}: {e}')
        sys.exit(1)

def load_model(model_choice, device, graph, model_repo_mapping, ModelClass):
    """
    Loads the specified model based on user choice.
    
    Args:
        model_choice (str): 'vanilla' or 'fusion'.
        device (torch.device): The device to load the model onto.
        graph (Graph): The graph structure required by the model.
        model_repo_mapping (dict): Mapping from model_choice to model_repo.
        ModelClass (torch.nn.Module): The model class to instantiate.
    
    Returns:
        torch.nn.Module: The loaded model.
    """
    if model_choice == 'vanilla':
        model_filename = 'best_model_vanilla.pth'
    else:
        model_filename = 'best_model_With_fusion.pth'

    # Retrieve the corresponding repository for the selected model
    model_repo = model_repo_mapping.get(model_choice)
    if not model_repo:
        logging.error(f"No repository defined for model choice '{model_choice}'.")
        sys.exit(1)

    # Download the model from Hugging Face
    model_path = download_model(model_filename, model_repo)

    # Instantiate the model
    if model_choice == 'vanilla':
        model = ModelClass(
            input_shape=(None, 90, 33, 3),
            adj_matrix=graph.AD,
            adj_matrix2=graph.AD2,
            adj_matrix3=graph.AD3,
            device=device
        )
    else:
        model = ModelClass(
            input_shape=(None, 90, 33, 3),
            biomech_dim=44,
            adj_matrix=graph.AD,
            adj_matrix2=graph.AD2,
            adj_matrix3=graph.AD3,
            device=device
        )

    # Move the model to the appropriate device
    model.to(device)
    logging.info(f'Model architecture:\n{model}')

    # Load the saved state dictionary
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded successfully from {model_path}')
    except Exception as e:
        logging.error(f'Error loading the model: {e}')
        sys.exit(1)

    return model


# --------------------- Prediction Function ---------------------

def predict(model, pose_data, biomech_data, device, batch_size=32):
    """
    Perform inference using the trained model.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        pose_data (np.ndarray): Pose features.
        biomech_data (np.ndarray or None): Biomechanical features (if applicable).
        device (torch.device): Device to perform computation on.
        batch_size (int): Batch size for inference.

    Returns:
        np.ndarray: Predicted probabilities.
    """
    model.eval()  # Ensure model is in evaluation mode
    y_pred_prob = []

    # Convert pose data to tensors
    pose_tensor = torch.tensor(pose_data, dtype=torch.float32).to(device)

    if biomech_data is not None:
        # Convert biomechanics data to tensors
        biomech_tensor = torch.tensor(biomech_data, dtype=torch.float32).to(device)
        # Create DataLoader with both pose and biomechanics data
        dataset = TensorDataset(pose_tensor, biomech_tensor)
    else:
        # Create DataLoader with only pose data
        dataset = TensorDataset(pose_tensor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            if biomech_data is not None:
                batch_pose, batch_biomech = batch
                outputs = model(batch_pose, batch_biomech)
            else:
                batch_pose = batch[0]
                outputs = model(batch_pose)
            
            # Assuming the model outputs logits; apply softmax for binary classification
            probs = torch.softmax(outputs, dim=1)[:, 1]
            y_pred_prob.extend(probs.cpu().numpy())

    return np.array(y_pred_prob)

# --------------------- Main Function ---------------------

def main(args):
    device = get_device(args.use_cuda)

    # Define the mapping from model_choice to model_repo
    # Replace the repository names with your actual Hugging Face repository identifiers
    model_repo_mapping = {
        'vanilla': 'fokhrul006/fall_prediction',  # Replace with your actual repo
        'fusion': 'fokhrul006/fall_prediction'     # Replace with your actual repo
    }

    # Conditional Imports (If opting for conditional imports)
    if args.model == 'vanilla':
        from STGCN_LSTM.data_processing import Data_Loader as Data_Loader_Selected
        from STGCN_LSTM.graph import Graph as Graph_Selected
        from STGCN_LSTM.stgcn import SGCN_LSTM as Model_Selected
    else:
        from STGCN_LSTM_Biomechanics.data_processing import Data_Loader as Data_Loader_Selected
        from STGCN_LSTM_Biomechanics.graph import Graph as Graph_Selected
        from STGCN_LSTM_Biomechanics.stgcn import SGCN_LSTM_Fused as Model_Selected

    # Initialize Data Loader and Graph based on model choice
    if args.model == 'vanilla':
        data_loader = data_vanilla()
        graph = graph_vanilla(len(data_loader.body_part))
    else:
        data_loader = data_fuse()
        graph = graph_fuse(len(data_loader.body_part))


    # Load the selected model
    model = load_model(args.model, device, graph, model_repo_mapping, Model_Selected)

    # Rest of the code...

    # Load and split data
    if args.model == 'vanilla':
        # Use only pose data
        pose_x = data_loader.scaled_x
        biomech_x = None
        y = data_loader.scaled_y
    else:
        # Use both pose and biomechanics data
        pose_x = data_loader.scaled_pose_x
        biomech_x = data_loader.scaled_biomech_x
        y = data_loader.scaled_y

    # Split data into training and validation sets with stratification
    if args.model == 'vanilla':
        # For vanilla model, biomech_x is None; use dummy data for splitting
        train_pose_x, valid_pose_x, _, _, train_y, valid_y = train_test_split(
            pose_x,
            np.zeros((len(pose_x), 0)),  # Dummy array
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    else:
        # For fusion model, include biomechanics data
        train_pose_x, valid_pose_x, train_biomech_x, valid_biomech_x, train_y, valid_y = train_test_split(
            pose_x,
            biomech_x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )


    logging.info(f"Validation instances: {len(valid_pose_x)}")

    # Perform inference on validation data
    y_pred_prob = predict(model, valid_pose_x, valid_biomech_x if args.model == 'fusion' else None, device)

    # Convert probabilities to binary predictions (threshold = 0.5)
    y_pred = (y_pred_prob >= 0.5).astype(int)

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

    # Function to save and close plots
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

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
