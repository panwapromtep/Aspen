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
            print("data_point:", data_point)
            optimizer.zero_grad()
            loss = MSELossFunction(data_point, model, device=device)
            print(loss)
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



import numpy as np
import torch

def generate_new_samples_ga(res, model, scaler, assSim, new_data_size=10, device='cpu'):
    """
    Select top candidates from the GA population based on the surrogate's 
    prediction of the *first* objective, evaluate them with the true simulator,
    and return scaled samples [X_scaled | Y_scaled].

    Args:
        res            : pymoo optimization result (res.pop.get("X"))
        model          : surrogate model (batch input -> batch output)
        scaler         : TorchMinMaxScaler (scale_y=True)
        assSim         : true simulation object with run_obj & unflatten_params
        new_data_size  : number of new samples to generate
        device         : 'cpu' or 'cuda'
    Returns:
        np.ndarray shape (new_data_size, n_var + n_obj)
    """

    # 1) Get GA population in surrogate-scaled space
    X_scaled = np.array(res.pop.get("X"))               # (pop_size, n_var)
    pop_size, n_var = X_scaled.shape

    # 2) Torch tensor for model
    X_scaled_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)

    # 3) Predict scaled objectives
    model.eval()
    with torch.no_grad():
        Y_scaled_pred_t = model(X_scaled_t)             # (pop_size, n_obj)

    # 4) Inverse-scale outputs to real objective values
    #    Only need dummy for X, since inverse_transform requires X input
    X_dummy_t = X_scaled_t.cpu()
    _, Y_pred_real_t = scaler.inverse_transform(X_dummy_t, Y_scaled_pred_t.cpu())
    Y_pred_real = Y_pred_real_t.numpy()                # (pop_size, n_obj)

    # 5) Rank by first objective
    top_idx = np.argsort(Y_pred_real[:, 0])[:new_data_size]
    X_top_scaled = X_scaled[top_idx]                   # (new_data_size, n_var)
    X_top_scaled_t = torch.tensor(X_top_scaled, dtype=torch.float32)

    # 6) Inverse-scale top inputs to real space for true sim
    X_top_real_t = scaler.inverse_transform(X_top_scaled_t.cpu())  # returns X_real_t
    X_top_real = X_top_real_t.numpy()                            # (new_dataSize, n_var)

    # 7) Evaluate true simulator
    Y_true = []
    for x in X_top_real:
        y = assSim.run_obj(assSim.unflatten_params(x))
        arr = np.atleast_1d(y)
        Y_true.append(arr)
    Y_true = np.vstack(Y_true)                      # (new_data_size, n_obj)

    # 8) Re-scale real inputs and true outputs
    X_real_t = torch.tensor(X_top_real, dtype=torch.float32)
    Y_real_t = torch.tensor(Y_true,    dtype=torch.float32)
    X_sel_scaled_t, Y_sel_scaled_t = scaler.transform(X_real_t, Y_real_t)

    # 9) Return concatenated [X_scaled | Y_scaled]
    X_sel = X_sel_scaled_t.numpy()
    Y_sel = Y_sel_scaled_t.numpy()
    return np.hstack([X_sel, Y_sel])


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
    all_res = []
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
        if current_pop is not None and len(current_pop) > 0:
            sampling = ResumeFromPopulation(current_pop)
        else:
            # you can keep your original pop_size here
            sampling = LHS()

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            eliminate_duplicates=True
        )

        res = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            verbose=True,
            save_history=True
        )
        
        all_res.append(res)

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
            "x": optim_input,
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
        'iteration_log': iteration_log,
        'all_result': all_res
    }




def optimize_surr_ga(
    model,
    dataset,
    assSim,
    problem,
    algo_factory,
    lrs={'first':1e-4, 'others':1e-5},
    epochs={'first':1000, 'others':100},
    batch_size=256,
    scaler=None,
    device="cpu",
    iter=10,
    pop_size=10,
    n_gen=3,
    new_data_size=10,
    print_loss=False,
    print_it_data=False
):    
    """
    Surrogate‐assisted GA optimization loop using a user‐provided algorithm factory.
    """

    iteration_log = []
    x_path, y_path = [], []
    assSim_call_count = 0
    res_data_list = []

    current_pop = None

    for it in range(iter):
        start_time = time.time()
        
        # select learning rate and epochs
        if it == 0:
            lr = lrs['first']
            epoch = epochs['first']
        else:
            lr = lrs['others']
            epoch = epochs['others']

        # train surrogate
        print(f"Iteration {it}: Training surrogate model...")
        model = train_model_nsga(
            model, dataset, device=device,
            epochs=epoch, lr=lr,
            print_loss=print_loss
        )

        # get algorithm instance from factory
        algorithm = algo_factory(it, current_pop)

        # run GA
        res = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            verbose=True,
            save_history=True
        )
        # Extract only the pieces you need from res
        res_data = {
            'final_X':    res.X.tolist(),
            'final_F':    res.F.tolist(),
            'final_CV':   res.CV.tolist() if hasattr(res, 'CV') else None,
            'history': [
                {
                    'X': gen.pop.get("X").tolist(),
                    'F': gen.pop.get("F").tolist(),
                    'G': gen.pop.get("G").tolist()
                }
                for gen in res.history
            ]
        }
        res_data_list.append(res_data)
        # update population
        current_pop = res.pop.get("X")

        # evaluate best solution
        optim_input_scaled = res.X
        optim_input_tensor = torch.tensor(optim_input_scaled, dtype=torch.float32)
        optim_input = scaler.inverse_transform(optim_input_tensor).numpy()

        y_val = assSim.run_obj(assSim.unflatten_params(optim_input))
        assSim_call_count += 1
        elapsed = time.time() - start_time

        iteration_log.append({
            "iteration": it,
            "time_sec": elapsed,
            "assSim_calls": assSim_call_count,
            "x": optim_input,
            "y": y_val
        })

        # scale and add new samples
        optim_input_scaled, y_scaled = scaler.transform(optim_input, np.array([[y_val]]))
        new_samples = [np.concatenate([optim_input_scaled.flatten(), y_scaled.flatten()])]

        additional = generate_new_samples_ga(
            res, model, scaler, assSim, new_data_size=new_data_size
        )
        new_samples.extend(additional)
        assSim_call_count += new_data_size

        dataset.add_samples(np.stack(new_samples))
        x_path.append(optim_input)
        y_path.append(y_val)

        if print_it_data:
            print(f"Iteration {it}: Optimal input {optim_input}, output {y_val}, dataset size {dataset.data.shape}")

    return {
        'model': model,
        'x_path': x_path,
        'y_path': y_path,
        'dataset': dataset,
        'assSim_call_count': assSim_call_count,
        'iteration_log': iteration_log,
        'res_data_list':     res_data_list
    }
