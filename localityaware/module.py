import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree
from itertools import cycle

### ==============================================
###  🚀 Dataset Classes
### ==============================================

class OldDataSet(Dataset):
    """Stores the historical dataset with a KDTree"""
    def __init__(self, data, k=5):
        self.data = data
        self.k = min(k, len(data) - 1)
        self.tree = cKDTree(self.data[:, :-1])  # KDTree for fast neighbor search
        self._update_indices()

    def _update_indices(self):
        """ Update the nearest neighbor indices """
        distances, indices = self.tree.query(self.data[:, :-1], k=min(self.k + 1, len(self.data)))
        self.indices = indices[:, 1:]  # Exclude self-neighbor

    def merge(self, new_data):
        """Merge new dataset into old dataset and rebuild KDTree"""
        self.data = np.vstack((self.data, new_data))
        self.k = min(self.k, len(self.data) - 1)
        self.tree = cKDTree(self.data[:, :-1])
        self._update_indices()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, -1]
        neighbor_values = self.data[self.indices[idx]]
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32), \
               torch.tensor(neighbor_values[:, :-1], dtype=torch.float32), \
               torch.tensor(neighbor_values[:, -1], dtype=torch.float32)


class NewDataSet(Dataset):
    """Stores new samples before merging, with KDTree updates"""
    def __init__(self, k=5):
        self.data = np.empty((0, k + 1))  # Start with empty array
        self.k = k
        self.tree = None  # No tree yet

    def add_samples(self, new_samples):
        """Safely add new samples and update KDTree."""
        if len(self.data) == 0:
            self.data = new_samples
        else:
            self.data = np.vstack((self.data, new_samples))

        if len(self.data) >= self.k:
            self.tree = cKDTree(self.data[:, :-1])
            self._update_indices()

    def _update_indices(self):
        """Update KDTree with available points"""
        if len(self.data) > 1:
            adjusted_k = min(self.k, len(self.data) - 1)
            distances, indices = self.tree.query(self.data[:, :-1], k=adjusted_k + 1)
            self.indices = indices[:, 1:]  # Exclude self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, -1]
        neighbor_values = self.data[self.indices[idx]] if self.tree else np.zeros((self.k, self.data.shape[1]))
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32), \
               torch.tensor(neighbor_values[:, :-1], dtype=torch.float32), \
               torch.tensor(neighbor_values[:, -1], dtype=torch.float32)

    def clear(self):
        """Reset dataset after merging"""
        self.data = np.empty((0, self.k + 1))
        self.tree = None


### ==============================================
###  🚀 Training & Loss Functions
### ==============================================

def GradPIELossFunction(data_point, model, dataset, device='cpu'):
    """Loss function for training"""
    x, y, neighbor_x, neighbor_y = data_point
    x, y = x.to(device), y.to(device)
    neighbor_x, neighbor_y = neighbor_x.to(device), neighbor_y.to(device)

    model_preds = model(x)
    neighbor_x_flat = neighbor_x.view(-1, neighbor_x.shape[-1])
    model_preds_neighbors = model(neighbor_x_flat).view(neighbor_x.shape[0], neighbor_x.shape[1])

    diff = (neighbor_y - y.unsqueeze(1)) - (model_preds_neighbors - model_preds.unsqueeze(1))
    loss = (diff ** 2).mean()
    return loss


def train_model(model, old_dataset, new_dataset=None, device='cpu', batch_size_old=256, batch_size_new=2, epochs=10, lr=0.001, new_data_weight=1.0):
    """Training function that balances old and new data"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    old_dataloader = DataLoader(old_dataset, batch_size=batch_size_old, shuffle=True)

    if new_dataset is None or len(new_dataset) == 0:
        print("⚠️ No new data available, training only on old dataset.")
        for epoch in range(epochs):
            for data_point in old_dataloader:
                optimizer.zero_grad()
                loss = GradPIELossFunction(data_point, model, old_dataset, device)
                loss.backward()
                optimizer.step()
        return model

    new_dataloader = DataLoader(new_dataset, batch_size=batch_size_new, shuffle=True)
    new_data_iter = cycle(new_dataloader) if len(new_dataset) > 0 else None

    for epoch in range(epochs):
        total_loss = 0
        for old_data in old_dataloader:
            optimizer.zero_grad()

            if new_data_iter is not None:
                new_data = next(new_data_iter)
                new_loss = GradPIELossFunction(new_data, model, new_dataset, device) * new_data_weight
            else:
                new_loss = 0

            old_loss = GradPIELossFunction(old_data, model, old_dataset, device)
            total_batch_loss = old_loss + (new_loss if new_loss != 0 else 0)
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()

        print(f"Epoch {epoch}: Total Loss={total_loss:.4f}")

    return model

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=[5, 5, 5], output_dim=1):
        """
        Fully connected feedforward neural network (MLP).

        Args:
        - input_dim (int): Number of input features.
        - hidden_layers (list of int): List where each element represents the number of neurons in that layer.
        - output_dim (int): Number of output neurons.
        """
        super(MLP, self).__init__()

        # **1️⃣ Create Layer List**
        layers = []
        prev_dim = input_dim

        # **2️⃣ Hidden Layers**
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())  # Activation function
            prev_dim = hidden_dim

        # **3️⃣ Output Layer**
        layers.append(nn.Linear(prev_dim, output_dim))

        # **4️⃣ Register as Sequential Model**
        self.net = nn.Sequential(*layers)

        # **5️⃣ Initialize Weights**
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)
