import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os


class SGCN_LSTM(nn.Module):
    def __init__(self, adj_matrix, adj_matrix2, adj_matrix3, input_shape, device='cuda'):
        super(SGCN_LSTM, self).__init__()
        self.device = device
        
        # Ensure adj_matrices are tensors
        if not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.from_numpy(adj_matrix)
        if not isinstance(adj_matrix2, torch.Tensor):
            adj_matrix2 = torch.from_numpy(adj_matrix2)
        if not isinstance(adj_matrix3, torch.Tensor):
            adj_matrix3 = torch.from_numpy(adj_matrix3)
        
        # Use clone().detach() instead of torch.tensor()
        self.adj = adj_matrix.clone().detach().float().to(self.device)
        self.adj2 = adj_matrix2.clone().detach().float().to(self.device)
        self.adj3 = adj_matrix3.clone().detach().float().to(self.device)
        
        self.num_nodes = input_shape[2]
        self.num_features = input_shape[3]
        
        # First SGCN block layers
        self.temporal_conv1 = nn.Conv2d(self.num_features, 64, kernel_size=(9, 1), padding=(4, 0))
        self.gcn_conv1_1 = nn.Conv2d(64 + self.num_features, 64, kernel_size=(1, 1))
        self.gcn_conv1_2 = nn.Conv2d(64 + self.num_features, 64, kernel_size=(1, 1))
        self.temp_conv1_1 = nn.Conv2d(128, 16, kernel_size=(9, 1), padding=(4, 0))
        self.temp_conv1_2 = nn.Conv2d(16, 16, kernel_size=(9, 1), padding=(4, 0))
        self.temp_conv1_3 = nn.Conv2d(16, 16, kernel_size=(9, 1), padding=(4, 0))
        
        # Second SGCN block layers
        self.temporal_conv2 = nn.Conv2d(48, 64, kernel_size=(9, 1), padding=(4, 0))
        self.gcn_conv2_1 = nn.Conv2d(64 + 48, 64, kernel_size=(1, 1))
        self.gcn_conv2_2 = nn.Conv2d(64 + 48, 64, kernel_size=(1, 1))
        self.temp_conv2_1 = nn.Conv2d(128, 16, kernel_size=(9, 1), padding=(4, 0))
        self.temp_conv2_2 = nn.Conv2d(16, 16, kernel_size=(9, 1), padding=(4, 0))
        self.temp_conv2_3 = nn.Conv2d(16, 16, kernel_size=(9, 1), padding=(4, 0))
        
        # Third SGCN block layers
        self.temporal_conv3 = nn.Conv2d(48, 64, kernel_size=(9, 1), padding=(4, 0))
        self.gcn_conv3_1 = nn.Conv2d(64 + 48, 64, kernel_size=(1, 1))
        self.gcn_conv3_2 = nn.Conv2d(64 + 48, 64, kernel_size=(1, 1))
        self.temp_conv3_1 = nn.Conv2d(128, 16, kernel_size=(9, 1), padding=(4, 0))
        self.temp_conv3_2 = nn.Conv2d(16, 16, kernel_size=(9, 1), padding=(4, 0))
        self.temp_conv3_3 = nn.Conv2d(16, 16, kernel_size=(9, 1), padding=(4, 0))
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=48 * self.num_nodes, hidden_size=40, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=80, hidden_size=40, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=40, hidden_size=40, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=40, hidden_size=80, batch_first=True)
        
        #BiLSTM layers
        # self.lstm1 = nn.LSTM(input_size=48 * self.num_nodes, hidden_size=40, batch_first=True)
        # self.lstm2 = nn.LSTM(input_size=80, hidden_size=40, batch_first=True, bidirectional=True)
        # self.lstm3 = nn.LSTM(input_size=80, hidden_size=40, batch_first=True, bidirectional=True)
        # self.lstm4 = nn.LSTM(input_size=40, hidden_size=80, batch_first=True)
        self.final_dense = nn.Linear(80, 2)
        
        # self.lstm1 = nn.LSTM(input_size=48 * self.num_nodes, hidden_size=40, batch_first=True, bidirectional=True)
        # self.lstm2 = nn.LSTM(input_size=80, hidden_size=40, batch_first=True, bidirectional=True)
        # self.lstm3 = nn.LSTM(input_size=80, hidden_size=40, batch_first=True, bidirectional=True)
        # self.lstm4 = nn.LSTM(input_size=80, hidden_size=80, batch_first=True, bidirectional=True)
        # # Final Dense Layer for Binary Classification
        # self.final_dense = nn.Linear(80, 2)  # Changed to 2 outputs for CrossEntropyLoss
        
        self.dropout = nn.Dropout(0.25)

    def sgcn_block(self, x, temporal_conv, gcn_conv1, gcn_conv2, temp_conv1, temp_conv2, temp_conv3):
        # Input shape: [batch, channels, timesteps, nodes]
        # Temporal convolution
        k1 = F.relu(temporal_conv(x))
        k = torch.cat([x, k1], dim=1)
        
        # Graph convolution
        x1 = F.relu(gcn_conv1(k))
        gcn_x1 = torch.einsum('vw,ntwc->ntvc', self.adj, x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        y1 = F.relu(gcn_conv2(k))
        gcn_y1 = torch.einsum('vw,ntwc->ntvc', self.adj2, y1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        gcn_1 = torch.cat([gcn_x1, gcn_y1], dim=1)
        
        # Temporal convolution
        z1 = F.relu(temp_conv1(gcn_1))
        z1 = self.dropout(z1)
        
        z2 = F.relu(temp_conv2(z1))
        z2 = self.dropout(z2)
        
        z3 = F.relu(temp_conv3(z2))
        z3 = self.dropout(z3)
        
        return torch.cat([z1, z2, z3], dim=1)

    def forward(self, x):
        # Input shape: [batch, timesteps, nodes, features]
        # Convert to [batch, features, timesteps, nodes]
        x = x.permute(0, 3, 1, 2)
        
        # First SGCN block
        x1 = self.sgcn_block(x, self.temporal_conv1, self.gcn_conv1_1, self.gcn_conv1_2,
                            self.temp_conv1_1, self.temp_conv1_2, self.temp_conv1_3)
        
        # # Second SGCN block with residual
        # x2 = self.sgcn_block(x1, self.temporal_conv2, self.gcn_conv2_1, self.gcn_conv2_2,
        #                     self.temp_conv2_1, self.temp_conv2_2, self.temp_conv2_3)
        # x2 = x2 + x1

        x3 = x1        
        # # Third SGCN block with residual
        # x3 = self.sgcn_block(x2, self.temporal_conv3, self.gcn_conv3_1, self.gcn_conv3_2,
        #                     self.temp_conv3_1, self.temp_conv3_2, self.temp_conv3_3)
        # x3 = x3 + x2
        
        # Reshape for LSTM [batch, timesteps, nodes * features]
        x3 = x3.permute(0, 2, 3, 1)  # [batch, timesteps, nodes, features]
        batch_size, timesteps, nodes, features = x3.shape
        x3 = x3.reshape(batch_size, timesteps, nodes * features)  # [batch, timesteps, nodes*features]
        
        # LSTM layers
        x4, _ = self.lstm1(x3)  # [batch, timesteps, 80]
        x4 = self.dropout(x4)
        # x5, _ = self.lstm2(x4)  # [batch, timesteps, 40]
        # x5 = self.dropout(x5)
        
        # x6, _ = self.lstm3(x5)  # [batch, timesteps, 40]
        # x6 = self.dropout(x6)
        
        x7, _ = self.lstm4(x4)  # [batch, timesteps, 80]
        x7 = self.dropout(x7)
        
        # Get last timestep and predict
        x7 = x7[:, -1, :]  # [batch, 80]
        out = self.final_dense(x7)  # [batch, 2]
        
        return out



# The SGCNLSTMTrainer class remains the same

class SGCNLSTMTrainer:
    def __init__(self, train_x, train_y, valid_x, valid_y, adj, adj2, adj3, 
                 lr=0.0001, epochs=200, batch_size=10, device='cuda',
                 save_start_epoch=50):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # If labels are one-hot encoded, convert to class indices
        if len(train_y.shape) > 1 and train_y.shape[1] > 1:
            train_y = np.argmax(train_y, axis=1)
        if len(valid_y.shape) > 1 and valid_y.shape[1] > 1:
            valid_y = np.argmax(valid_y, axis=1)
        
        # Ensure train_y and valid_y are 1D
        train_y = train_y.flatten()
        valid_y = valid_y.flatten()
        
        # Convert data to PyTorch tensors and move to device
        self.train_x = torch.FloatTensor(train_x).to(self.device)
        self.train_y = torch.LongTensor(train_y).to(self.device)
        self.valid_x = torch.FloatTensor(valid_x).to(self.device)
        self.valid_y = torch.LongTensor(valid_y).to(self.device)
        
        # Print shapes for debugging
        print(f'Train labels shape: {self.train_y.shape}')
        print(f'Validation labels shape: {self.valid_y.shape}')
        
        # Create model and move to device
        self.model = SGCN_LSTM(adj, adj2, adj3, train_x.shape, device=self.device).to(self.device)
        
        # Calculate class weights for weighted loss
        class_counts = np.bincount(train_y.flatten())
        num_classes = 2  # Adjust based on your problem
        if len(class_counts) < num_classes:
            class_counts = np.append(class_counts, [0]*(num_classes - len(class_counts)))
        
        # Calculate class weights: inverse frequency
        class_weights = 1. / (class_counts + 1e-6)  # Add epsilon to prevent division by zero
        class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1
        
        self.class_weights = torch.FloatTensor(class_weights).to(self.device)
        print(f'Class weights: {self.class_weights}')
        
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_start_epoch = save_start_epoch
        
        # Create data loaders
        self.train_dataset = TensorDataset(self.train_x, self.train_y)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        
        self.valid_dataset = TensorDataset(self.valid_x, self.valid_y)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False)

    def calculate_metrics(self, outputs, targets):
        # Assuming outputs are raw logits; apply softmax
        probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class '1'
        
        # Detach tensors before converting to NumPy
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        probabilities = probabilities.detach().cpu().numpy()
        
        precision = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        auprc = average_precision_score(targets, probabilities)
        
        return precision, recall, f1, auprc

    def train(self):
        best_val_auprc = float('-inf')
        history = {
            'train_loss': [], 'val_loss': [], 
            'train_precision': [], 'train_recall': [], 
            'train_f1': [], 'train_auprc': [],
            'val_precision': [], 'val_recall': [], 
            'val_f1': [], 'val_auprc': []
        }
        
        # Initialize a learning rate scheduler
        scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=100, verbose=True)

        # Create a directory to save the best model
        os.makedirs('models', exist_ok=True)
        best_model_path = os.path.join('models', 'best_model_vanilla.pth')

        for epoch in range(self.epochs):
            # Training Phase
            self.model.train()
            train_loss = 0
            train_outputs_all = []
            train_targets_all = []
            
            for batch_x, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                train_outputs_all.append(outputs)
                train_targets_all.append(batch_y)
            
            train_loss /= len(self.train_loader)
            train_outputs = torch.cat(train_outputs_all)
            train_targets = torch.cat(train_targets_all)
            train_precision, train_recall, train_f1, train_auprc = self.calculate_metrics(train_outputs, train_targets)
            
            # Validation Phase
            self.model.eval()
            val_loss = 0
            val_outputs_all = []
            val_targets_all = []
            
            with torch.no_grad():
                for batch_x, batch_y in self.valid_loader:
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    val_outputs_all.append(outputs)
                    val_targets_all.append(batch_y)
            
            val_loss /= len(self.valid_loader)
            val_outputs = torch.cat(val_outputs_all)
            val_targets = torch.cat(val_targets_all)
            val_precision, val_recall, val_f1, val_auprc = self.calculate_metrics(val_outputs, val_targets)
            
            # Save best model based on validation AUPRC after save_start_epoch
            if (epoch + 1) >= self.save_start_epoch and val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                torch.save(self.model.state_dict(), best_model_path)
                print(f'New best model saved at epoch {epoch+1} with AUPRC: {val_auprc:.4f}')
            
            # Step the scheduler
            scheduler.step(val_auprc)

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['train_f1'].append(train_f1)
            history['train_auprc'].append(train_auprc)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1'].append(val_f1)
            history['val_auprc'].append(val_auprc)
            
            print(f'Epoch {epoch+1}/{self.epochs} - '
                  f'Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - '
                  f'Train AUPRC: {train_auprc:.4f} - Val AUPRC: {val_auprc:.4f}')
        
        print(f'Training completed. Best validation AUPRC: {best_val_auprc:.4f}')
        return history

    def predict(self, x):
        self.model.eval()
        x_tensor = torch.FloatTensor(x).to(self.device)
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0]
                outputs = self.model(batch_x)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class '1'
                predictions.extend(probabilities.cpu().numpy())
        
        return np.array(predictions)