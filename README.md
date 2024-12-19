#fallGCN: Fall Prediction Pipeline using Spatio-temporal Graph Convoulation Networks with Biomechanical Feature Fusion

A robust Python-based solution for predicting falls using skeletal (pose) data and biomechanics data. The pipeline supports two model variants:
- **Vanilla Model**: Uses only skeleton (pose) data process by GCN
- **Fusion Model**: Combines skeleton data with biomechanics data for enhanced prediction accuracy

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Data](#data)
- [Model Training](#model-training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Features

- Dual model support (vanilla and fusion)
- Hugging Face integration for model management
- Flexible inference pipeline
- Comprehensive evaluation metrics
- Structured logging system

## Installation

### Prerequisites
- Python 3.8 or higher
- pytorch == 2.1.0+cu121 
- Hugging Face account (for accessing model repositories)

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/fokhruli/fallGCN.git
   cd fallGCN
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   
   # Unix/MacOS
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

<!-- 4. **Configure Hugging Face Authentication**
   ```bash
   # Unix/MacOS
   export HUGGINGFACE_HUB_TOKEN=your_token_here
   
   # Windows (CMD)
   set HUGGINGFACE_HUB_TOKEN=your_token_here
   
   # Windows (PowerShell)
   $env:HUGGINGFACE_HUB_TOKEN="your_token_here" -->
   ```

## Data

### Data Download
1. Access the dataset from the provided source (contact repository maintainers for access)
2. Place the downloaded data in the `data/` directory

### Data Structure
- `pose_data/`: Contains skeletal data files
- `biomech_data/`: Contains biomechanics data files (required for fusion model only)

### Preprocessing
The pipeline includes automatic preprocessing steps:
- Scaling of pose data
- Normalization of biomechanics data
- Feature alignment for fusion model

## Model Training

### Model Architecture
The system uses STGCN_LSTM architecture with two variants:
- Vanilla model for pose-only prediction
- Fusion model for combined pose and biomechanics prediction

### Training Process
1. **Configure Training Parameters**
   ```bash
   # Edit config.yaml to set:
   - Learning rate
   - Batch size
   - Number of epochs
   - Model architecture parameters
   ```

2. **Start Training**
   ```bash
   python train.py --model [vanilla|fusion] --config config.yaml
   ```

## Inference

### Running Inference
1. **Using Vanilla Model**
   ```bash
   python inference_script.py --model vanilla --use_cuda
   ```

2. **Using Fusion Model**
   ```bash
   python inference_script.py --model fusion --use_cuda
   ```

### Model Selection
- Vanilla Model: `best_model_vanilla.pth`
- Fusion Model: `best_model_with_fusion.pth`

Both models are available on Hugging Face Hub: `fokhrul006/fall_prediction`

## Evaluation

### Metrics
The pipeline automatically calculates:
- Precision, Recall, F1-Score
- AUPRC and ROC AUC
- Confusion Matrix

### Visualization
Generated plots include:
- Precision-Recall curves
- ROC curves
- Confusion matrices

### Logging
- Logs are stored in `logs/inference.log`
- Each run overwrites previous logs
- Includes model performance metrics and execution details

## Contributing

1. Fork the repository
2. Create your feature branch
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. Commit your changes
   ```bash
   git commit -m "Add feature: YourFeatureName"
   ```
4. Push to your fork
   ```bash
   git push origin feature/YourFeatureName
   ```
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Contact

- Name: [Your Name]
- Email: your.email@example.com
- GitHub: your_username

## extract keypoints/landmarks from videos using mediapipe

[ConvLSTM](https://github.com/ndrplz/ConvLSTM_pytorch/tree/master)