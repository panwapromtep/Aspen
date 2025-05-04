# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:39:46 2025

"""
import sys
import os
import numpy as np
import time

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

from AspenSim import AspenSim
# from CodeLibrary import Simulation

from pymoo.core.problem import Problem
from pymoo.problems.multi import BNH
problem = BNH()

import torch

class BNHWrapper:
    def __init__(self):
        pass

    def runSim(self, x):
        """
        x: flat array-like of shape (2,) — [x1, x2]
        returns: dict with keys 'F' (objectives), 'G' (constraints)
        """
        x = np.array(x)
        x1, x2 = x[0], x[1]

        # Objective functions
        f1 = 4 * x1**2 + 4 * x2**2
        f2 = (x1 - 5)**2 + (x2 - 5)**2

        # Constraints
        g1 = (x1 - 5)**2 + x2**2 - 25  # should be <= 0
        g2 = 7.7 - ((x1 - 8)**2 + (x2 + 3)**2)  # should be <= 0

        return {
            "F": [f1, f2],
            "G": [g1, g2]
        }

    def run_obj(self, x):
        return self.runSim(x)["F"]

    def unflatten_params(self, flat_array):
        return flat_array

from pymoo.core.problem import Problem

import torch
import numpy as np
from pymoo.core.problem import Problem

class SurrogateBNHProblem(Problem):
    def __init__(self, model, scaler, device='cpu'):
        # decision vars live in [-1,1]²
        xl = np.array([-1., -1.])
        xu = np.array([ 1.,  1.])
        super().__init__(
            n_var=2, n_obj=2, n_ieq_constr=2,
            xl=xl, xu=xu,
            elementwise=False,
            vectorized=True
        )
        self.model  = model
        self.scaler = scaler
        self.device = device

    def _evaluate(self, X, out, *args, **kwargs):
        # X: numpy (n,2) in [-1,1] range
        # 1) to torch
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        # 2) get original-space inputs for constraints
        X_orig_t = self.scaler.inverse_transform(X_t)      # returns torch.Tensor (n,2)
        x1 = X_orig_t[:,0].cpu().numpy()
        x2 = X_orig_t[:,1].cpu().numpy()

        # 3) scale again for the surrogate model
        X_scaled_t = self.scaler.transform(X_orig_t)        # torch.Tensor (n,2)

        # 4) model forward → get scaled predictions
        self.model.eval()
        with torch.no_grad():
            Y_scaled_t = self.model(X_scaled_t)

        # 5) un‑scale predictions to objective space
        _, Y_unscaled_t = self.scaler.inverse_transform(X_scaled_t, Y_scaled_t)
        F = Y_unscaled_t.cpu().numpy()                     # shape (n,2)

        # 6) compute analytical constraints in original space
        g1 = (x1 - 5)**2  + x2**2     - 25                  # <= 0
        g2 = 7.7 - ((x1 - 8)**2 + (x2 + 3)**2)             # <= 0
        G  = np.column_stack([g1, g2])

        # 7) hand back to pymoo
        out["F"] = F
        out["G"] = G
