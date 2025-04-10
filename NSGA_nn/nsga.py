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
from torch.utils.data import Dataset, DataLoader

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)
print(sys.path)

from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS


from localityaware.module import *

from pymoo.core.sampling import Sampling


class ResumeFromPopulation(Sampling):
    def __init__(self, X):
        super().__init__()
        self.X = X

    def _do(self, problem, n_samples, **kwargs):
        return self.X

class FlashOpProblemNN(ElementwiseProblem):
    def __init__(self, model):
        super().__init__(n_var=2, n_obj=1, xl=[-1, -1], xu=[1, 1])
        self.model = model

    def _evaluate(self, x, out, *args, **kwargs):
        x_eval = torch.tensor([[x[0], x[1]]], dtype=torch.float32)
        with torch.no_grad():
            out["F"] = self.model(x_eval).numpy()
            
def train_model_nsga1(model, 
                dataset, 
                device='cpu', 
                batch_size=256, 
                epochs=10, 
                lr=0.001,
                lambda_mse=1, 
                print_loss=False
                ):
    device = torch.device(device)
    """Training function that balances old and new data"""
    model = model.to(device)
    print('device', device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0
        for data_point in dataloader:
            optimizer.zero_grad()
            loss = GradPIELossMSEFunction(data_point, model, lambda_mse=lambda_mse, device=device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if print_loss and epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Loss={total_loss:.4f}")
    return model

def train_model_nsga2(model, 
                dataset, 
                device='cpu', 
                batch_size=256, 
                epochs=10, 
                lr=0.001,
                lambda_mse=1, 
                print_loss=False
                ):
    device = torch.device(device)
    """Training function that balances old and new data"""
    model = model.to(device)
    print('device', device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0
        for data_point in dataloader:
            optimizer.zero_grad()
            loss = MSELossFunction(data_point, model, device=device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if print_loss and epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Loss={total_loss:.4f}")
    return model
        

def generate_new_samples_nsga(res, scaler, assSim, new_data_size=10):
    all_x = res.pop.get("X")
    all_f = res.pop.get("F")
    # sort the population based on the objective value
    I = np.argsort(all_f[:, 0])
    #take the x values of the first new_data_size samples
    new_samples = all_x[I[:new_data_size]]
    #unscale the new samples and pass through assSim to get the true objective value
    new_samples_unscaled = scaler.inverse_transform(new_samples)
    new_samples_y = []
    for i in range(new_data_size):
        y_val = assSim.run_obj(assSim.unflatten_params(new_samples_unscaled[i]))
        new_samples_y.append(y_val)
    new_samples_y = np.array(new_samples_y)
    new_samples_scaled, new_samples_y_scaled = scaler.transform(new_samples_unscaled, new_samples_y)
    return np.concatenate([new_samples_scaled, new_samples_y_scaled], axis=1)

def optimize_surr_nsga_1(
    model,
    old_dataset,
    assSim,
    lr=1e-4,
    epoch=30,
    min_vals=None,
    max_vals=None,
    scaler=None,
    device="cpu",
    iter=10,
    pop_size=10,
    n_gen=3,
    new_data_size=10,
    print_loss=False,
    print_it_data=False,
    lambda_mse=0
):
    x_path, y_path = [], []
    assSim_call_count = 0
    new_dataset = NewDataSet(k=14)

    populations = []
    current_pop = None

    for it in range(iter):
        print(f"Iteration {it}: Training surrogate model...")
        model = train_model_nsga1(
            model, old_dataset, device=device,
            epochs=epoch, lr=lr, lambda_mse=lambda_mse,
            print_loss=print_loss
        )

        # Initialize GA with previous population if available
        if current_pop is not None:
            print(f"Using previous population of size {len(current_pop)}")
            #store the population in the populations list

            algorithm = GA(
                pop_size=pop_size,
                eliminate_duplicates=True,
                sampling=ResumeFromPopulation(current_pop)
            )
        else:
            algorithm = GA(
                pop_size=pop_size,
                sampling=LHS(),
                eliminate_duplicates=True
            )

        problem = FlashOpProblemNN(model)

        res = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            verbose=True,
            save_history=True
        )

        # Save population to reuse
        current_pop = res.pop.get("X")
        initial_gen = res.history[0].pop.get("X")
        populations.append(initial_gen)

        # Evaluate optimal solution using true simulation
        optim_input_scaled = res.X
        optim_input_tensor = torch.tensor(optim_input_scaled, dtype=torch.float32)
        optim_input = scaler.inverse_transform(optim_input_tensor).numpy()

        y_val = assSim.run_obj(assSim.unflatten_params(optim_input))
        assSim_call_count += 1

        # Scale new input-output pair and add to dataset
        optim_input_scaled, y_scaled = scaler.transform(optim_input, np.array([[y_val]]))
        new_samples = [np.concatenate([optim_input_scaled.flatten(), y_scaled.flatten()])]

        # Generate additional candidate samples from GA population
        additional_samples = generate_new_samples_nsga(res, scaler, assSim, new_data_size=new_data_size)
        new_samples.extend(additional_samples)

        # Update dataset
        new_dataset.add_samples(np.stack(new_samples))
        old_dataset.merge(new_dataset.data)
        new_dataset.clear()

        # Track paths
        x_path.append(optim_input)
        y_path.append(y_val)

        if print_it_data:
            print(f"Iteration {it}: Optimal input {optim_input}, output {y_val}, dataset size {old_dataset.data.shape}")

    return {
        'model': model,
        'x_path': x_path,
        'y_path': y_path,
        'dataset': old_dataset,
        'assSim_call_count': assSim_call_count,
        'populations': populations
    }

def optimize_surr_nsga_2(
    model,
    old_dataset,
    assSim,
    lr=1e-4,
    epoch=30,
    min_vals=None,
    max_vals=None,
    scaler=None,
    device="cpu",
    iter=10,
    pop_size=10,
    n_gen=3,
    new_data_size=10,
    print_loss=False,
    print_it_data=False,
    lambda_mse=0
):
    x_path, y_path = [], []
    assSim_call_count = 0
    new_dataset = NewDataSet(k=14)

    populations = []
    current_pop = None

    for it in range(iter):
        print(f"Iteration {it}: Training surrogate model...")
        model = train_model_nsga2(
            model, old_dataset, device=device,
            epochs=epoch, lr=lr, lambda_mse=lambda_mse,
            print_loss=print_loss
        )

        # Initialize GA with previous population if available
        if current_pop is not None:
            print(f"Using previous population of size {len(current_pop)}")
            #store the population in the populations list

            algorithm = GA(
                pop_size=pop_size,
                eliminate_duplicates=True,
                sampling=ResumeFromPopulation(current_pop)
            )
        else:
            algorithm = GA(
                pop_size=pop_size,
                sampling=LHS(),
                eliminate_duplicates=True
            )

        problem = FlashOpProblemNN(model)

        res = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            verbose=True,
            save_history=True
        )

        # Save population to reuse
        current_pop = res.pop.get("X")
        initial_gen = res.history[0].pop.get("X")
        populations.append(initial_gen)

        # Evaluate optimal solution using true simulation
        optim_input_scaled = res.X
        optim_input_tensor = torch.tensor(optim_input_scaled, dtype=torch.float32)
        optim_input = scaler.inverse_transform(optim_input_tensor).numpy()

        y_val = assSim.run_obj(assSim.unflatten_params(optim_input))
        assSim_call_count += 1

        # Scale new input-output pair and add to dataset
        optim_input_scaled, y_scaled = scaler.transform(optim_input, np.array([[y_val]]))
        new_samples = [np.concatenate([optim_input_scaled.flatten(), y_scaled.flatten()])]

        # Generate additional candidate samples from GA population
        additional_samples = generate_new_samples_nsga(res, scaler, assSim, new_data_size=new_data_size)
        new_samples.extend(additional_samples)

        # Update dataset
        new_dataset.add_samples(np.stack(new_samples))
        old_dataset.merge(new_dataset.data)
        new_dataset.clear()

        # Track paths
        x_path.append(optim_input)
        y_path.append(y_val)

        if print_it_data:
            print(f"Iteration {it}: Optimal input {optim_input}, output {y_val}, dataset size {old_dataset.data.shape}")

    return {
        'model': model,
        'x_path': x_path,
        'y_path': y_path,
        'dataset': old_dataset,
        'assSim_call_count': assSim_call_count,
        'populations': populations
    }
