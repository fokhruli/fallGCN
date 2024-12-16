import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SGCN_LSTM(nn.Module):
    def __init__(self, adj_matrix, adj_matrix2, adj_matrix3, input_shape, device='cuda'):
        super(SGCN_LSTM, self).__init__()
        self.device = device
        self.adj = torch.tensor(adj_matrix, dtype=torch.float32).to(self.device)
        self.adj2 = torch.tensor(adj_matrix2, dtype=torch.float32).to(self.device)
        self.adj3 = torch.tensor(adj_matrix3, dtype=torch.float32).to(self.device)
        
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
        self.lstm1 = nn.LSTM(input_size=48 * self.num_nodes, hidden_size=80, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=80, hidden_size=40, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=40, hidden_size=40, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=40, hidden_size=80, batch_first=True)
        
        self.final_dense = nn.Linear(80, 1)
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
        
        # Second SGCN block with residual
        x2 = self.sgcn_block(x1, self.temporal_conv2, self.gcn_conv2_1, self.gcn_conv2_2,
                            self.temp_conv2_1, self.temp_conv2_2, self.temp_conv2_3)
        x2 = x2 + x1
        
        # Third SGCN block with residual
        x3 = self.sgcn_block(x2, self.temporal_conv3, self.gcn_conv3_1, self.gcn_conv3_2,
                            self.temp_conv3_1, self.temp_conv3_2, self.temp_conv3_3)
        x3 = x3 + x2
        
        # Reshape for LSTM [batch, timesteps, nodes * features]
        x3 = x3.permute(0, 2, 3, 1)
        batch_size, timesteps, nodes, features = x3.shape
        x3 = x3.reshape(batch_size, timesteps, nodes * features)
        
        # LSTM layers
        x4, _ = self.lstm1(x3)
        x4 = self.dropout(x4)
        
        x5, _ = self.lstm2(x4)
        x5 = self.dropout(x5)
        
        x6, _ = self.lstm3(x5)
        x6 = self.dropout(x6)
        
        x7, _ = self.lstm4(x6)
        x7 = self.dropout(x7)
        
        # Get last timestep and predict
        x7 = x7[:, -1, :]
        out = self.final_dense(x7)
        
        return out

# The SGCNLSTMTrainer class remains the same

class SGCNLSTMTrainer:
    def __init__(self, train_x, train_y, valid_x, valid_y, adj, adj2, adj3, 
                 lr=0.0001, epochs=200, batch_size=10, device='cuda'):
        self.device = device
        
        # Convert data to PyTorch tensors and move to device
        self.train_x = torch.FloatTensor(train_x).to(self.device)
        self.train_y = torch.FloatTensor(train_y).to(self.device)
        self.valid_x = torch.FloatTensor(valid_x).to(self.device)
        self.valid_y = torch.FloatTensor(valid_y).to(self.device)
        
        # Create model and move to device
        self.model = SGCN_LSTM(adj, adj2, adj3, train_x.shape, device=self.device).to(self.device)
        
        self.criterion = nn.HuberLoss(delta=0.1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Create data loaders
        self.train_dataset = TensorDataset(self.train_x, self.train_y)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        
        self.valid_dataset = TensorDataset(self.valid_x, self.valid_y)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size)

    def calculate_metrics(self, outputs, targets):
        outputs = outputs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        mad = np.mean(np.abs(outputs - targets))
        mape = np.mean(np.abs((targets - outputs) / (targets + 1e-8))) * 100
        
        return mad, mape

    def train(self):
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'train_mad': [], 'train_mape': [], 
                  'val_mad': [], 'val_mape': []}
        
        for epoch in range(self.epochs):
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
            
            train_outputs = torch.cat(train_outputs_all)
            train_targets = torch.cat(train_targets_all)
            train_mad, train_mape = self.calculate_metrics(train_outputs, train_targets)
            
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
            
            val_outputs = torch.cat(val_outputs_all)
            val_targets = torch.cat(val_targets_all)
            val_mad, val_mape = self.calculate_metrics(val_outputs, val_targets)
            
            train_loss /= len(self.train_loader)
            val_loss /= len(self.valid_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mad'].append(train_mad)
            history['train_mape'].append(train_mape)
            history['val_mad'].append(val_mad)
            history['val_mape'].append(val_mape)
            
            print(f'Epoch {epoch+1}/{self.epochs} - '
                  f'Train Loss: {train_loss:.4f} - '
                  f'Val Loss: {val_loss:.4f} - '
                #   f'Train MAD: {train_mad:.4f} - '
                  f'Val MAD: {val_mad:.4f} - '
                #   f'Train MAPE: {train_mape:.2f}% - '
                  f'Val MAPE: {val_mape:.2f}%')
        
        return history

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            data = torch.FloatTensor(data).to(self.device)
            predictions = self.model(data)
        return predictions.cpu().numpy()