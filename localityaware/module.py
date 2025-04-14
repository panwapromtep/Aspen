import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree
from itertools import cycle
import csv
import time
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F

### ==============================================
###  🚀 Dataset Classes
### ==============================================

#tree logic looks good
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

def MSELossFunction(data_point, model, device='cpu'):
    # print("Using MSELossFunction")
    device = torch.device(device)
    x, y = data_point  # Ignore neighbor_x and neighbor_y
    x = x.to(device)
    y = y.to(device)
    
    # print("x.shape:", x.shape)
    # print("y.shape:", y.shape)

    model_preds = model(x)  # shape: (batch_size, output_dim)
    mse = nn.MSELoss()
    loss = mse(model_preds, y)
    
    return loss

def GradPIELossMSEFunction(data_point, model, lambda_mse = 1e-2, device='cpu'):
    device = torch.device('cpu')
    x, y, neighbor_x, neighbor_y = data_point

    x = x.to(device)
    y = y.to(device)
    
    neighbor_x = neighbor_x.to(device)
    neighbor_y = neighbor_y.to(device)

    batch_size, k, feature_dim = neighbor_x.shape
    model_preds = model(x)  # shape: (batch_size, output_dim)

    # Flatten neighbor_x for a single forward pass through the model.
    neighbor_x_flat = neighbor_x.reshape(batch_size * k, feature_dim)

    # Get model predictions for neighbors.
    model_preds_neighbors = model(neighbor_x_flat)  # shape: (batch_size*k, output_dim)
    model_preds_neighbors = model_preds_neighbors.view(batch_size, k)
    
    mse = nn.MSELoss()
    
    mseloss = mse(model_preds, y.unsqueeze(1)) * lambda_mse
    diff = (neighbor_y - y.unsqueeze(1)) - (model_preds_neighbors - model_preds)
    loss = (diff ** 2).mean() + mseloss
    return loss


def train_model(model, 
                old_dataset, 
                new_dataset=None, 
                device='cpu', 
                batch_size_old=256, 
                batch_size_new=2, 
                epochs=10, 
                lr=0.001,
                lambda_mse=1, 
                new_data_weight=1.0,
                print_loss=False
                ):
    device = torch.device(device)
    """Training function that balances old and new data"""
    model = model.to(device)
    print('device', device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    old_dataloader = DataLoader(old_dataset, batch_size=batch_size_old, shuffle=True)

    if new_dataset is None or len(new_dataset) == 0:
        print("⚠️ No new data available, training only on old dataset.")
        for epoch in range(epochs):
            total_loss = 0
            for data_point in old_dataloader:
                optimizer.zero_grad()
                loss = GradPIELossMSEFunction(data_point, model, lambda_mse=lambda_mse, device=device)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if print_loss and epoch % 10 == 0:
                print(f"Epoch {epoch}: Total Loss={total_loss:.4f}")
        return model

    new_dataloader = DataLoader(new_dataset, batch_size=batch_size_new, shuffle=True)
    new_data_iter = cycle(new_dataloader) if len(new_dataset) > 0 else None

    for epoch in range(epochs):
        total_loss = 0
        for old_data in old_dataloader:
            optimizer.zero_grad()

            if new_data_iter is not None:
                new_data = next(new_data_iter)
                new_loss = GradPIELossMSEFunction(new_data, model, lambda_mse=lambda_mse, device=device) * new_data_weight
            else:
                new_loss = 0

            old_loss = GradPIELossMSEFunction(old_data, model, lambda_mse=lambda_mse, device=device)
            total_batch_loss = old_loss + (new_loss if new_loss != 0 else 0)
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()
        if print_loss and epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Loss={total_loss:.4f}")

    return model
### Scaler
class TorchMinMaxScaler:
    """
    Differentiable min-max scaler for both input features (X) and, optionally, target values (y).
    Scales data to a target feature range (default is [-1, 1]) using PyTorch operations.
    """
    def __init__(self, feature_range=(-1, 1), min_vals=None, max_vals=None, scale_y=False, min_y=None, max_y=None, eps=1e-8):
        """
        Args:
            feature_range (tuple): Target range for scaling (default: (-1, 1)).
            min_vals (array-like): Minimum values per feature (for X).
            max_vals (array-like): Maximum values per feature (for X).
            scale_y (bool): If True, also scale the target variable (y-values).
            min_y (array-like): Minimum value(s) for y.
            max_y (array-like): Maximum value(s) for y.
            eps (float): Small constant to avoid division by zero.
        """
        self.feature_range = feature_range
        self.scale_y = scale_y
        self.eps = eps
        
        # Convert min and max values to torch tensors if provided.
        if min_vals is not None and max_vals is not None:
            self.min_x = torch.tensor(min_vals, dtype=torch.float32)
            self.max_x = torch.tensor(max_vals, dtype=torch.float32)
        else:
            self.min_x = None
            self.max_x = None

        if self.scale_y:
            if min_y is not None and max_y is not None:
                self.min_y = torch.tensor(min_y, dtype=torch.float32)
                self.max_y = torch.tensor(max_y, dtype=torch.float32)
            else:
                self.min_y = None
                self.max_y = None

    def transform(self, X, y=None):
        """
        Apply min-max scaling to features X (and optionally target values y) using torch operations.
        
        Args:
            X (Tensor or array-like): Input features.
            y (Tensor or array-like, optional): Target values.
        
        Returns:
            X_scaled (Tensor): Scaled features.
            (optionally) y_scaled (Tensor): Scaled target values.
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        # Scale to feature_range, assuming feature_range is (-1, 1):
        X_scaled = 2 * (X - self.min_x) / (self.max_x - self.min_x + self.eps) - 1
        
        if y is not None and self.scale_y:
            y = torch.as_tensor(y, dtype=torch.float32)
            # Ensure y is at least 2D for proper broadcasting
            if y.ndim == 1:
                y = y.unsqueeze(1)
            y_scaled = 2 * (y - self.min_y) / (self.max_y - self.min_y + self.eps) - 1
            return X_scaled, y_scaled
        
        return X_scaled

    def inverse_transform(self, X, y=None):
        """
        Undo min-max scaling on features X (and optionally target values y) using torch operations.
        
        Args:
            X (Tensor or array-like): Scaled features.
            y (Tensor or array-like, optional): Scaled target values.
        
        Returns:
            X_inv (Tensor): Features in the original scale.
            (optionally) y_inv (Tensor): Target values in the original scale.
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        X_inv = (X + 1) * (self.max_x - self.min_x) / 2 + self.min_x
        
        if y is not None and self.scale_y:
            y = torch.as_tensor(y, dtype=torch.float32)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            y_inv = (y + 1) * (self.max_y - self.min_y) / 2 + self.min_y
            return X_inv, y_inv
        
        return X_inv

def optimize_surrogate_model(model, old_dataset, new_dataset, assSim, 
                             optim_steps=40, N_s=5, 
                             lr={"model learning rate": 1e-2, "input learning rate": 1e-2}, 
                             merge_interval=10,
                             x_init=torch.tensor([0.9, -0.9], dtype=torch.float32, requires_grad=True),
                             epochs={"start": 20, "during": 3},
                             batch_size_old=256,
                             batch_size_new=2,
                             min_vals=None,
                             max_vals=None,
                             upsteps=5,
                             patience = 10,
                             tolerance = 1e-2,
                             scaler=None,  # ✅ Scale y-values only if provided
                             device="cpu",
                             log_file="optimization_log.csv",
                             calls_log_file="assSim_calls_log.csv",
                             lambda_mse = 0,
                             new_data_weight = 20,
                             surface_log_folder="model_surfaces",
                             stddev = 0.1
                             ):  # ✅ Folder for saving NN surfaces

    x_path, y_path = [], []
    assSim_call_count = 0
    
    # ✅ Train initial surrogate model
    model = train_model(model, 
                        old_dataset, 
                        new_dataset, 
                        batch_size_old=batch_size_old, 
                        device=device, 
                        epochs=epochs["start"], 
                        lambda_mse=lambda_mse)
    
    # check loss of model
    
    # ✅ Ensure surface storage directory exists
    os.makedirs(surface_log_folder, exist_ok=True)
    
    save_model_surface(model,
                       0,
                       surface_log_folder=surface_log_folder,
                       min_vals = min_vals,
                        max_vals = max_vals,
                        device = device,
                        scaler = scaler
                       )


    # **Logging Setup**
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "Time (s)", "x_1", "x_2", "y_val (real)"])
    
    with open(calls_log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "assSim Calls", "Objective Value (real)"])

    start_time = time.time()
    
    alpha = lr["input learning rate"]      # learning rate
    beta1 = 0.4
    beta2 = 0.8
    adam_epsilon = 1e-8
    
    # Initialize first and second moment estimates as zeros (same shape as x_init)
    m = torch.zeros_like(x_init)
    v = torch.zeros_like(x_init)
    t = 0

    for step in range(optim_steps):
        #check patience
        if step > 0 and len(y_path) > 1:
            if abs(y_path[-1] - y_path[-2]) < tolerance:  # Small improvement threshold
                patience -= 1
            else:
                patience = 10  # Reset patience when there's meaningful improvement
        
        if patience == 0:
            print("Early stopping at step", step)
            break
        
        iter_time = time.time() - start_time
        new_samples = []
        print(f"Step {step + 1}/{optim_steps} - Time: {iter_time:.2f}s")
        
        x = x_init.detach()  # Evaluate new x_init (real-domain value)
        y = assSim.run_obj(assSim.unflatten_params(x))  # Evaluate in real-world scale
        assSim_call_count += 1
        x_path.append(x.numpy().tolist())
        y_path.append(y)

        # ✅ Log updated values using x and y (real values)
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([step + 1, iter_time, x[0].item(), x[1].item(), y])
            
        with open(calls_log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([step + 1, assSim_call_count, y])
        
        #Do scaling on x_init and y_init
        if scaler is not None:
            x_init_scaled, y_init_scaled = scaler.transform(x.numpy().reshape(1, -1), np.array([[y]]))  
            x_init_scaled, y_init_scaled = x_init_scaled.flatten(), y_init_scaled.flatten()  
        else:
            x_init_scaled, y_init_scaled = x.numpy(), np.array([y])  
        
        #Add x_init and y_init to new samples
        new_samples.append(np.concatenate((x_init_scaled, y_init_scaled)))  # Safe concatenation

        # ✅ Generate N_s samples around x_init
        for _ in range(N_s):
            # Generate noise in scaled space (stddev is defined in the scaled space)
            noise = torch.randn_like(x_init_scaled) * stddev
            sample_scaled = x_init_scaled + noise
            # Inverse-transform back to real domain for simulation evaluation
            sample_real = scaler.inverse_transform(sample_scaled)

            y_sample_real = assSim.run_obj(assSim.unflatten_params(sample_real.squeeze(0)))
            assSim_call_count += 1

            # Scale the sample back to the scaled space
            sample_scaled, y_sample_scaled = scaler.transform(sample_real, np.array([[y_sample_real]]))
            sample_scaled, y_sample_scaled = sample_scaled.flatten(), y_sample_scaled.flatten()
            new_samples.append(np.concatenate((sample_scaled, y_sample_scaled)))  # Safe concatenation

        # ✅ Add new samples to dataset
        new_dataset.add_samples(np.stack(new_samples))
        
        # ✅ Train model (already scaled inputs)
        model = train_model(model, 
                            old_dataset, 
                            new_dataset, 
                            device=device, 
                            batch_size_new=batch_size_new,
                            batch_size_old=batch_size_old,
                            epochs=epochs["during"], 
                            new_data_weight=new_data_weight,
                            lambda_mse=lambda_mse)

        # ✅ **Gradient-Based Optimization Using Surrogate Model**
        for inner_step in range(upsteps):
            t += 1
            # Scale x_init using the differentiable TorchMinMaxScaler (do not convert to numpy)
            x_init_scaled = scaler.transform(x_init.reshape(1, -1))
            
            # Compute the surrogate prediction
            y_pred_scaled = model(x_init_scaled).requires_grad_(True)
            
            # Compute the gradient of the prediction with respect to the original x_init
            grad_pred = torch.autograd.grad(y_pred_scaled, x_init_scaled, create_graph = True, retain_graph=True)[0].squeeze(0)
 
            m = beta1 * m + (1 - beta1) * grad_pred
            v = beta2 * v + (1 - beta2) * (grad_pred ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            x_init_scaled = (x_init_scaled - alpha * m_hat / (torch.sqrt(v_hat) + adam_epsilon)).detach().clone().requires_grad_(True)
            x_init = scaler.inverse_transform(x_init_scaled).flatten()

        
        grad_pred_clean = grad_pred.squeeze(0).detach()  # Now a 1D tensor

        # ✅ Save NN surface every `merge_interval` steps
        if step % merge_interval == 0 and step != 0:
            save_model_surface(model, 
                               step, 
                               surface_log_folder=surface_log_folder, 
                               min_vals=min_vals, 
                               max_vals=max_vals, 
                               device=device, 
                               scaler=scaler,
                               grad = grad_pred_clean.tolist(), 
                               x = x_init.tolist())
            
            old_dataset.merge(new_dataset.data)
            new_dataset.clear()

    print(f"Optimization completed. Logs saved to {log_file} and {calls_log_file}")
    print(new_samples)
    return x_path, y_path

def save_model_surface(model, step, surface_log_folder, min_vals, max_vals, device, scaler, x=None, grad=None):
    """
    Generate and store the neural network's surface predictions for x1, x2 every 10 steps.
    Also saves the current x and its gradient (assumed to be a pair of numbers) if provided.
    """
    # Define a grid over the input space
    x1_vals = np.linspace(min_vals[0], max_vals[0], 100)
    x2_vals = np.linspace(min_vals[1], max_vals[1], 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    grid_points = np.column_stack((X1.ravel(), X2.ravel()))
    # Convert grid to tensor
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)
    print(grid_tensor)
    grid_tensor = scaler.transform(grid_tensor)


    # Predict using the model
    with torch.no_grad():
        predictions = model(grid_tensor)
    
    predictions = scaler.inverse_transform(predictions).cpu().numpy()
    print(predictions)

    # Save surface predictions to CSV
    surface_file = os.path.join(surface_log_folder, f"surface_step_{step}.csv")
    with open(surface_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x1", "x2", "NN Output"])
        for i in range(len(grid_points)):
            writer.writerow([grid_points[i, 0], grid_points[i, 1], predictions[i, 0]])
    print(f"✅ Neural network surface saved at step {step}: {surface_file}")

    # Save x if provided (assuming x is a 1D array with 2 numbers)
    if x is not None:
        x_file = os.path.join(surface_log_folder, f"x_step_{step}.csv")
        with open(x_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["x1", "x2"])
            # Write x as a single row
            writer.writerow(x)
        print(f"✅ Input x saved at step {step}: {x_file}")
    
    # Save gradient if provided (assuming grad is a pair of numbers)
    if grad is not None:
        grad_file = os.path.join(surface_log_folder, f"grad_step_{step}.csv")
        with open(grad_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["grad_x1", "grad_x2"])
            writer.writerow(grad)
        print(f"✅ Gradient saved at step {step}: {grad_file}")

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
        # layers.append(nn.Softplus())

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
