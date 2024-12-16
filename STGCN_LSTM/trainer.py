import numpy as np
import torch
import torch.cuda
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_processing import Data_Loader
from graph import Graph
from stgcn import SGCN_LSTM, SGCNLSTMTrainer

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f'Found GPU at: {torch.cuda.get_device_name(0)}')
else:
    raise SystemError('GPU device not found')

# Load and prepare data
data_loader = Data_Loader()
graph = Graph(len(data_loader.body_part))

# Split data
train_x, valid_x, train_y, valid_y = train_test_split(
    data_loader.scaled_x, 
    data_loader.scaled_y, 
    test_size=0.2, 
    random_state=RANDOM_SEED
)

print("Training instances: ", len(train_x))
print("Validation instances: ", len(valid_x))


# Initialize trainer with device
trainer = SGCNLSTMTrainer(
    train_x=train_x,
    train_y=train_y,
    valid_x=valid_x,
    valid_y=valid_y,
    adj=graph.AD,
    adj2=graph.AD2,
    adj3=graph.AD3,
    lr=0.0001,
    epochs=3000,
    batch_size=32
)



# Build and print model summary
print("Model Architecture:")
print(trainer.model)
total_params = sum(p.numel() for p in trainer.model.parameters())
print(f"Total parameters: {total_params}")

# Train the model
history = trainer.train()

# Load pre-trained weights if they exist
try:
    trainer.model.load_state_dict(torch.load("best_model.pt"))
    print("Loaded pre-trained weights successfully")
except FileNotFoundError:
    print("No pre-trained weights found")

# Make predictions
y_pred = trainer.predict(valid_x)

# Inverse transform predictions and actual values
y_pred = data_loader.sc2.inverse_transform(y_pred)
valid_y = data_loader.sc2.inverse_transform(valid_y)

# Plotting
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(y_pred, 's', color='red', label='Prediction', linestyle='None', alpha=0.5, markersize=6)
plt.plot(valid_y, 'o', color='green', label='Actual Score', alpha=0.4, markersize=6)
plt.title('Validation Set', fontsize=18)
plt.xlabel('Sequence Number', fontsize=16)
plt.ylabel('Score', fontsize=16)
plt.legend()

# Evaluation metrics
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

test_dev = np.abs(valid_y - y_pred)
mean_abs_dev = np.mean(test_dev)
mae = mean_absolute_error(valid_y, y_pred)
rms_dev = sqrt(mean_squared_error(y_pred, valid_y))
mse = mean_squared_error(valid_y, y_pred)
mape = mean_absolute_percentage_error(valid_y, y_pred)

print('Mean absolute error:', mae)
print('RMS deviation:', rms_dev)
print('MSE:', mse)
print('MAPE:', mape)

plt.show()