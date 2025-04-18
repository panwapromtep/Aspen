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
import time
import torch
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)
print(sys.path)

from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS


from localityaware.module import *
from localityaware.module import MSELossFunction

from pymoo.core.sampling import Sampling


class DynamicDataset(Dataset):
    def __init__(self, data, num_inputs=2):
        """
        Args:
            data (np.array): Data with shape (N, total_features)
            num_inputs (int): Number of columns in data corresponding to input features.
                              The remaining columns will be treated as outputs.
        """
        self.data = data
        self.num_inputs = num_inputs
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # The first num_inputs columns are inputs; the remaining columns are outputs.
        x = self.data[idx, :self.num_inputs]
        y = self.data[idx, self.num_inputs:]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    def add_samples(self, new_samples):
        """Safely add new samples and update KDTree."""
        if len(self.data) == 0:
            self.data = new_samples
        else:
            self.data = np.vstack((self.data, new_samples))

class ResumeFromPopulation(Sampling):
    def __init__(self, X, **kwargs):
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
        

def train_model_nsga(model, 
                dataset, 
                device='cpu', 
                batch_size=256, 
                epochs=10, 
                lr=0.001,
                print_loss=False
                ):
    device = torch.device(device)
    """Training function that balances old and new data"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0
        for data_point in dataloader:
            #print dimensions of data_point
            # print("data_point.shape:", data_point.shape)
            optimizer.zero_grad()
            # print("data_point[0].shape:", data_point[0].shape)
            # print("data_point[1].shape:", data_point[1].shape)
            loss = MSELossFunction(data_point, model, device=device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if print_loss and epoch % 50 == 0:
            print(f"Epoch {epoch}: Total Loss={total_loss:.4f}")
    return model
        
def generate_new_samples_nsga(res, scaler, assSim, new_data_size=10):
    """
    Generate additional candidate samples from the GA population using random sampling.
    
    Parameters:
        res : The optimization result from pymoo.
        scaler : The scaling object with inverse_transform and transform methods.
        assSim : The simulation object that implements run_obj and unflatten_params.
        new_data_size : The fixed number of additional samples to select.
    
    Returns:
        A NumPy array where each row is a concatenated (scaled) candidate input (8-D)
        and the corresponding (scaled) objective values (2-D), i.e. an array of shape (new_data_size, 10).
    """
    all_x = res.pop.get("X")  # shape: (pop_size, n_var)
    
    # Randomly select new_data_size candidates from the current population.
    indices = np.random.choice(len(all_x), size=new_data_size, replace=False)
    new_samples = all_x[indices]  # shape: (new_data_size, n_var)
    
    # Unscale these candidate inputs.
    new_samples_unscaled = scaler.inverse_transform(new_samples)
    
    # Evaluate each candidate using the true simulation.
    new_samples_y = []
    for sample in new_samples_unscaled:
        y_val = assSim.run_obj(assSim.unflatten_params(sample))
        new_samples_y.append(y_val)
    new_samples_y = np.array(new_samples_y)  # shape: (new_data_size, n_obj)
    
    # Re-scale both inputs and outputs.
    new_samples_scaled, new_samples_y_scaled = scaler.transform(new_samples_unscaled, new_samples_y)
    
    # Concatenate the scaled inputs (n_var columns) and scaled outputs (n_obj columns)
    return np.concatenate([new_samples_scaled, new_samples_y_scaled], axis=1)

def optimize_surr_nsga(
    model,
    dataset,
    assSim,
    problem,
    lrs={'first':1e-4, 'others':1e-5},
    epochs={'first':1000, 'others':100},
    batch_size=256,
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
    max_aspen_calls_per_iter=None,
):
    """
    Surrogate‐assisted NSGA‑II optimization loop with controlled Aspen calls
    and fallback sampling if no Pareto front is found.
    """

    iteration_log = []
    x_path, y_path = [], []
    assSim_call_count = 0

    populations = []
    current_pop = None

    for it in range(iter):
        start_time = time.time()

        # pick LR / epochs
        if it == 0:
            lr, epoch = lrs['first'], epochs['first']
        else:
            print("dataset.data.shape:", dataset.data.shape)
            lr, epoch = lrs['others'], epochs['others']

        # train surrogate
        print(f"Iteration {it}: Training surrogate model...")
        model = train_model_nsga(
            model, dataset, device=device,
            epochs=epoch, lr=lr,
            print_loss=print_loss
        )

        # set up NSGA-II with resume or LHS
        if current_pop is not None:
            #print(f"Using previous population of size {len(current_pop)}")
            #sampling = ResumeFromPopulation(current_pop)
            algorithm = NSGA2(pop_size=pop_size, sampling=LHS(), eliminate_duplicates=True)
        else:
           algorithm = NSGA2(pop_size=pop_size, sampling=LHS(), eliminate_duplicates=True)

        res = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            verbose=True,
            save_history=True
        )

        # store for next iteration
        current_pop = res.pop.get("X")
        populations.append(res.history[0].pop.get("X"))

        # get scaled inputs (may be None if no front)
        optim_input_scaled = res.X

        # decide whether to use front or fallback population
        if optim_input_scaled is None or len(optim_input_scaled) == 0:
            print("No Pareto front found; sampling the final GA population instead")
            fallback_scaled = current_pop
            optim_input_scaled = np.array(fallback_scaled)

        # inverse‐scale to real inputs
        optim_input_tensor = torch.tensor(optim_input_scaled, dtype=torch.float32)
        optim_input = scaler.inverse_transform(optim_input_tensor).numpy()

        # evaluate up to new_data_size candidates
        y_vals = []
        calls_this_iter = 0
        for idx, candidate in enumerate(optim_input):
            if max_aspen_calls_per_iter is not None and calls_this_iter >= max_aspen_calls_per_iter:
                print(f"Reached cap of {max_aspen_calls_per_iter} Aspen calls this iteration.")
                break
            if idx >= new_data_size:
                break
            y = assSim.run_obj(assSim.unflatten_params(candidate))
            y_vals.append(y)
            assSim_call_count += 1
            calls_this_iter   += 1

        print("assSim_call_count:", assSim_call_count)

        # prepare arrays for scaling
        y_vals_array = np.array(y_vals)
        inputs_evaluated = optim_input[:len(y_vals)]

        # scale evaluated inputs & outputs
        eval_scaled_in, eval_scaled_out = scaler.transform(inputs_evaluated, y_vals_array)
        evaluated_samples = np.hstack([eval_scaled_in, eval_scaled_out])

        # generate additional samples if needed
        if len(y_vals) < new_data_size:
            needed = new_data_size - len(y_vals)
            print(f"Generating {needed} additional samples to reach {new_data_size}")
            additional_samples = generate_new_samples_nsga(
                res, scaler, assSim, new_data_size=needed
            )
            assSim_call_count += needed
        else:
            # empty array with correct column count
            n_cols = evaluated_samples.shape[1]
            additional_samples = np.empty((0, n_cols))

        # add everything to dataset
        new_samples = np.vstack([evaluated_samples, additional_samples])
        dataset.add_samples(new_samples)

        # logging & bookkeeping
        elapsed = time.time() - start_time
        x_path.append(inputs_evaluated)
        y_path.append(y_vals)
        iteration_log.append({
            "iteration": it,
            "time_sec": elapsed,
            "assSim_calls": assSim_call_count,
            "x": inputs_evaluated,
            "y": y_vals
        })

        if print_it_data:
            print(f"Iteration {it}: Added {new_data_size} points, dataset size {dataset.data.shape}")

    return {
        'model': model,
        'x_path': x_path,
        'y_path': y_path,
        'dataset': dataset,
        'assSim_call_count': assSim_call_count,
        'populations': populations,
        'iteration_log': iteration_log
    }
