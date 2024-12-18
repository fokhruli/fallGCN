import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim1, feature_dim2, fused_dim=64, num_heads=4):
        super(AttentionFusion, self).__init__()
        self.query_proj = nn.Linear(feature_dim1, fused_dim)
        self.key_proj = nn.Linear(feature_dim2, fused_dim)
        self.value_proj = nn.Linear(feature_dim2, fused_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)
        self.output_proj = nn.Linear(fused_dim, fused_dim)  # Ensure this matches fused_dim

    def forward(self, x1, x2):
        """
        Args:
            x1: Tensor of shape [batch, feature_dim1]
            x2: Tensor of shape [batch, feature_dim2]
        Returns:
            fused: Tensor of shape [batch, fused_dim]
        """
        # Project the inputs
        query = self.query_proj(x1).unsqueeze(1)  # [batch, 1, fused_dim]
        key = self.key_proj(x2).unsqueeze(1)      # [batch, 1, fused_dim]
        value = self.value_proj(x2).unsqueeze(1)  # [batch, 1, fused_dim]

        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(query, key, value)  # [batch, 1, fused_dim]

        # Project the attention output
        fused = self.output_proj(attn_output.squeeze(1))  # [batch, fused_dim]
        return fused


class SGCN_LSTM_Fused(nn.Module):
    def __init__(self, adj_matrix, adj_matrix2, adj_matrix3, input_shape, biomech_dim=45, device='cuda'):
        super(SGCN_LSTM_Fused, self).__init__()
        self.device = device
        
        # Ensure adj_matrices are tensors
        if not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.from_numpy(adj_matrix)
        if not isinstance(adj_matrix2, torch.Tensor):
            adj_matrix2 = torch.from_numpy(adj_matrix2)
        if not isinstance(adj_matrix3, torch.Tensor):
            adj_matrix3 = torch.from_numpy(adj_matrix3)
        
        self.adj = adj_matrix.clone().detach().float().to(self.device)
        self.adj2 = adj_matrix2.clone().detach().float().to(self.device)
        self.adj3 = adj_matrix3.clone().detach().float().to(self.device)
        
        self.num_nodes = input_shape[2]
        self.num_features = input_shape[3]
        
        # ST-GCN layers (same as original)
        self.temporal_conv1 = nn.Conv2d(self.num_features, 64, kernel_size=(9, 1), padding=(4, 0))
        self.gcn_conv1_1 = nn.Conv2d(64 + self.num_features, 64, kernel_size=(1, 1))
        self.gcn_conv1_2 = nn.Conv2d(64 + self.num_features, 64, kernel_size=(1, 1))
        self.temp_conv1_1 = nn.Conv2d(128, 16, kernel_size=(9, 1), padding=(4, 0))
        self.temp_conv1_2 = nn.Conv2d(16, 16, kernel_size=(9, 1), padding=(4, 0))
        self.temp_conv1_3 = nn.Conv2d(16, 16, kernel_size=(9, 1), padding=(4, 0))
        
        self.temporal_conv2 = nn.Conv2d(48, 64, kernel_size=(9, 1), padding=(4, 0))
        self.gcn_conv2_1 = nn.Conv2d(64 + 48, 64, kernel_size=(1, 1))
        self.gcn_conv2_2 = nn.Conv2d(64 + 48, 64, kernel_size=(1, 1))
        self.temp_conv2_1 = nn.Conv2d(128, 16, kernel_size=(9, 1), padding=(4, 0))
        self.temp_conv2_2 = nn.Conv2d(16, 16, kernel_size=(9, 1), padding=(4, 0))
        self.temp_conv2_3 = nn.Conv2d(16, 16, kernel_size=(9, 1), padding=(4, 0))
        
        self.temporal_conv3 = nn.Conv2d(48, 64, kernel_size=(9, 1), padding=(4, 0))
        self.gcn_conv3_1 = nn.Conv2d(64 + 48, 64, kernel_size=(1, 1))
        self.gcn_conv3_2 = nn.Conv2d(64 + 48, 64, kernel_size=(1, 1))
        self.temp_conv3_1 = nn.Conv2d(128, 16, kernel_size=(9, 1), padding=(4, 0))
        self.temp_conv3_2 = nn.Conv2d(16, 16, kernel_size=(9, 1), padding=(4, 0))
        self.temp_conv3_3 = nn.Conv2d(16, 16, kernel_size=(9, 1), padding=(4, 0))
        
        # LSTM for ST-GCN features
        self.stgcn_lstm1 = nn.LSTM(input_size=48 * self.num_nodes, hidden_size=40, 
                                 batch_first=True)
        self.stgcn_lstm2 = nn.LSTM(input_size=40, hidden_size=80, batch_first=True)

        # # BiLSTM for biomechanical features
        # self.biomech_lstm = nn.LSTM(input_size=biomech_dim, hidden_size=40,
        #                            batch_first=True, bidirectional=True)
        
        # # Fusion layers
        # self.fusion_layer = nn.Sequential(
        #     nn.Linear(80 + 80, 128),  # 80 from each BiLSTM (40*2 due to bidirectional)
        #     nn.ReLU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.25)
        # )
        
        # # Final classification layer
        # self.final_dense = nn.Linear(64, 2)
        
        # self.dropout = nn.Dropout(0.25)

        # Biomechanical feature reduction network
        self.biomech_reduction = nn.Sequential(
            nn.Linear(biomech_dim, 30),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.Linear(64, 30),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # 1D CNN for biomechanical temporal processing
        self.biomech_cnn = nn.Sequential(
            nn.Conv1d(30, 40, kernel_size=7, padding=3),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.Conv1d(40, 40, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # BiLSTM for biomechanical sequence processing
        self.biomech_lstm = nn.LSTM(
            input_size=45,  # Input from CNN output
            hidden_size=40,  # Will give 80 features after bidirectional concat
            batch_first=True #,
            # bidirectional=True
        )
        self.biomech_lstm1 = nn.LSTM(
            input_size=45,  # Input from CNN output
            hidden_size=80,  # Will give 80 features after bidirectional concat
            batch_first=True #,
            # bidirectional=True
        )
        
       # Define the attention fusion layer
        fused_dim = 80  # Ensure this is consistent
        self.attention_fusion = AttentionFusion(
            feature_dim1=80,      # Dimension of stgcn_features2
            feature_dim2=80,      # Dimension of biomech_features
            fused_dim=fused_dim,
            num_heads=4           # Number of attention heads
        )

        # Update the fusion layer to accept fused_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_dim, 128),  # in_features=128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Final classification layer
        self.final_dense = nn.Linear(160, 2)
        self.dropout = nn.Dropout(0.25)

    def process_biomech_features(self, biomech_data):
        # Input shape: [batch, timesteps, features]
        batch_size, timesteps, _ = biomech_data.shape
        # import pdb
        # pdb.set_trace()
        # # Reshape for feature reduction
        # biomech_flat = biomech_data.reshape(-1, biomech_data.shape[-1])

        # # Reduce dimensionality from 45 to 30
        # reduced_features = self.biomech_reduction(biomech_flat)
        # reduced_features = reduced_features.reshape(batch_size, timesteps, 30)
        
        # # Prepare for 1D CNN [batch, channels, timesteps]
        # cnn_input = reduced_features.transpose(1, 2)
        
        # # Apply 1D CNN
        # cnn_output = self.biomech_cnn(cnn_input)
        
        # # Prepare for LSTM [batch, timesteps, features]
        # lstm_input = cnn_output.transpose(1, 2)
        
        # Process through BiLSTM
        # lstm_output, _ = self.biomech_lstm(biomech_data)
        # lstm_output = self.dropout(lstm_output)
        lstm_output = biomech_data
        lstm_output, _ = self.biomech_lstm1(lstm_output)
        lstm_output = self.dropout(lstm_output)
        
        # Take final timestep features
        final_features = lstm_output[:, -1, :]  # Shape: [batch, 80]
        
        return final_features
    

    def sgcn_block(self, x, temporal_conv, gcn_conv1, gcn_conv2, temp_conv1, temp_conv2, temp_conv3):
        # Same as original implementation
        k1 = F.relu(temporal_conv(x))
        k = torch.cat([x, k1], dim=1)
        
        x1 = F.relu(gcn_conv1(k))
        gcn_x1 = torch.einsum('vw,ntwc->ntvc', self.adj, x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        y1 = F.relu(gcn_conv2(k))
        gcn_y1 = torch.einsum('vw,ntwc->ntvc', self.adj2, y1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        gcn_1 = torch.cat([gcn_x1, gcn_y1], dim=1)
        
        z1 = F.relu(temp_conv1(gcn_1))
        z1 = self.dropout(z1)
        
        z2 = F.relu(temp_conv2(z1))
        z2 = self.dropout(z2)
        
        z3 = F.relu(temp_conv3(z2))
        z3 = self.dropout(z3)
        
        return torch.cat([z1, z2, z3], dim=1)

    def forward(self, pose_data, biomech_data):
        # Process pose data through ST-GCN (keep existing code)
        x = pose_data.permute(0, 3, 1, 2)
        
        # ST-GCN blocks
        x1 = self.sgcn_block(x, self.temporal_conv1, self.gcn_conv1_1, self.gcn_conv1_2,
                            self.temp_conv1_1, self.temp_conv1_2, self.temp_conv1_3)
        
        x2 = self.sgcn_block(x1, self.temporal_conv2, self.gcn_conv2_1, self.gcn_conv2_2,
                            self.temp_conv2_1, self.temp_conv2_2, self.temp_conv2_3)
        x2 = x2 + x1
        
        x3 = self.sgcn_block(x2, self.temporal_conv3, self.gcn_conv3_1, self.gcn_conv3_2,
                            self.temp_conv3_1, self.temp_conv3_2, self.temp_conv3_3)
        x3 = x3 + x2
        x3 =x1
        # Process through STGCN-LSTM (keep existing code)
        x3 = x3.permute(0, 2, 3, 1)
        batch_size, timesteps, nodes, features = x3.shape
        x3 = x3.reshape(batch_size, timesteps, nodes * features)

        
        stgcn_features1, _ = self.stgcn_lstm1(x3)
        stgcn_features1 = self.dropout(stgcn_features1)
        
        stgcn_features2, _ = self.stgcn_lstm2(stgcn_features1)
        stgcn_features2 = self.dropout(stgcn_features2)
        stgcn_features2 = stgcn_features2[:, -1, :]  # [batch, 80]
        
        # Process biomechanical features
        biomech_features = self.process_biomech_features(biomech_data)  # [batch, 80]

        # Fuse features using attention
        # import pdb
        # pdb.set_trace()
        fused_features = self.attention_fusion(stgcn_features2, biomech_features)  # [batch, 128]
        # fused_features = self.fusion_layer(fused_features)  # [batch, 64]
        # Scale biomech_features to give less priority

        # alpha = 0.5  # Scaling factor (adjust as needed)
        # biomech_features_scaled = biomech_features * alpha

        # Concatenate ST-GCN features and scaled biomech_features
        fused_features = torch.cat([stgcn_features2, fused_features], dim=1)  # [batch, 160]

        # Pass through the fusion layer ()
        # fused_features = self.fusion_layer(fused_features)  # [batch, 64]

        # Final prediction

        out = self.final_dense(fused_features)  # [batch, 2]
        
        return out

class FusedModelTrainer:
    def __init__(self, train_pose_x, train_biomech_x, train_y, 
                 valid_pose_x, valid_biomech_x, valid_y,
                 adj, adj2, adj3, lr=0.0001, epochs=200, batch_size=10, 
                 device='cuda', save_start_epoch=50):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Convert labels if needed
        if len(train_y.shape) > 1 and train_y.shape[1] > 1:
            train_y = np.argmax(train_y, axis=1)
        if len(valid_y.shape) > 1 and valid_y.shape[1] > 1:
            valid_y = np.argmax(valid_y, axis=1)
        
        train_y = train_y.flatten()
        valid_y = valid_y.flatten()
        
        # Convert data to tensors
        self.train_pose_x = torch.FloatTensor(train_pose_x).to(self.device)
        self.train_biomech_x = torch.FloatTensor(train_biomech_x).to(self.device)
        self.train_y = torch.LongTensor(train_y).to(self.device)
        
        self.valid_pose_x = torch.FloatTensor(valid_pose_x).to(self.device)
        self.valid_biomech_x = torch.FloatTensor(valid_biomech_x).to(self.device)
        self.valid_y = torch.LongTensor(valid_y).to(self.device)
        
        print(f'Train labels shape: {self.train_y.shape}')
        print(f'Validation labels shape: {self.valid_y.shape}')
        
        # Create model
        self.model = SGCN_LSTM_Fused(adj, adj2, adj3, train_pose_x.shape, 
                                    biomech_dim=train_biomech_x.shape[-1],
                                    device=self.device).to(self.device)
        
        # Calculate class weights
        class_counts = np.bincount(train_y.flatten())
        num_classes = 2
        if len(class_counts) < num_classes:
            class_counts = np.append(class_counts, [0]*(num_classes - len(class_counts)))
        
        class_weights = 1. / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum()
        
        self.class_weights = torch.FloatTensor(class_weights).to(self.device)
        print(f'Class weights: {self.class_weights}')
        
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_start_epoch = save_start_epoch
        
        # Create dataset and loaders
        self.train_dataset = TensorDataset(self.train_pose_x, self.train_biomech_x, self.train_y)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        
        self.valid_dataset = TensorDataset(self.valid_pose_x, self.valid_biomech_x, self.valid_y)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False)


    def calculate_metrics(self, outputs, targets):
        # Same as original implementation
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        
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
        
        scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, 
                                    patience=100, verbose=True)
        
        os.makedirs('models', exist_ok=True)
        best_model_path = os.path.join('models', 'best_model_fused.pth')

        for epoch in range(self.epochs):
            # Training Phase
            self.model.train()
            train_loss = 0
            train_outputs_all = []
            train_targets_all = []
            
            for batch_pose_x, batch_biomech_x, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_pose_x, batch_biomech_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                train_outputs_all.append(outputs)
                train_targets_all.append(batch_y)
            
            train_loss /= len(self.train_loader)
            train_outputs = torch.cat(train_outputs_all)
            train_targets = torch.cat(train_targets_all)
            train_metrics = self.calculate_metrics(train_outputs, train_targets)            
            # Validation Phase
            self.model.eval()
            val_loss = 0
            val_outputs_all = []
            val_targets_all = []
            
            with torch.no_grad():
                for batch_pose_x, batch_biomech_x, batch_y in self.valid_loader:
                    outputs = self.model(batch_pose_x, batch_biomech_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    val_outputs_all.append(outputs)
                    val_targets_all.append(batch_y)
            
            val_loss /= len(self.valid_loader)
            val_outputs = torch.cat(val_outputs_all)
            val_targets = torch.cat(val_targets_all)
            val_precision, val_recall, val_f1, val_auprc = self.calculate_metrics(val_outputs, val_targets)
            
            # Save best model
            if (epoch + 1) >= self.save_start_epoch and val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                torch.save(self.model.state_dict(), best_model_path)
                print(f'New best model saved at epoch {epoch+1} with AUPRC: {val_auprc:.4f}')
            
            # Update scheduler
            scheduler.step(val_auprc)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_precision'].append(train_metrics[0])
            history['train_recall'].append(train_metrics[1])
            history['train_f1'].append(train_metrics[2])
            history['train_auprc'].append(train_metrics[3])
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1'].append(val_f1)
            history['val_auprc'].append(val_auprc)
            
            print(f'Epoch {epoch+1}/{self.epochs} - '
                  f'Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - '
                  f'Train AUPRC: {train_metrics[3]:.4f} - Val AUPRC: {val_auprc:.4f}')
        
        print(f'Training completed. Best validation AUPRC: {best_val_auprc:.4f}')
        return history

    def predict(self, pose_x, biomech_x):
        """
        Make predictions on new data
        Args:
            pose_x: numpy array of pose features
            biomech_x: numpy array of biomechanical features
        Returns:
            numpy array of predicted probabilities
        """
        self.model.eval()
        pose_tensor = torch.FloatTensor(pose_x).to(self.device)
        biomech_tensor = torch.FloatTensor(biomech_x).to(self.device)
        dataset = TensorDataset(pose_tensor, biomech_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        predictions = []
        
        with torch.no_grad():
            for batch_pose_x, batch_biomech_x in loader:
                outputs = self.model(batch_pose_x, batch_biomech_x)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class '1'
                predictions.extend(probabilities.cpu().numpy())
        
        return np.array(predictions)

    def load_best_model(self, model_path):
        """
        Load the best model from saved checkpoint
        Args:
            model_path: path to the saved model checkpoint
        """
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

