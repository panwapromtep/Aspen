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
import torch

class VinylDistillationProblem(Problem):
    def __init__(self, model):
        # n_var = 8 input variables, n_obj = 2 objectives.
        # Set vectorized=True so that _evaluate receives a matrix of solutions.
        super().__init__(n_var=8, n_obj=2, xl=[-1]*8, xu=[1]*8, vectorized=True)
        self.model = model

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Vectorized evaluation of candidate solutions.

        Parameters:
            X: A 2D NumPy array of shape (n, 8), where each row is a candidate solution.
            out: A dictionary where results (objectives) should be stored under key "F".
        """
        # Convert X into a torch tensor. X is assumed to be a numpy array with shape (n, 8)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            # Evaluate the model on the entire batch.
            # The model should return a tensor of shape (n, 2)
            F = self.model(X_tensor).numpy()
            
        out["F"] = F


class VCDistillationDummy:
    """
    A dummy version of a distillation column simulation.
    This class implements the same interface as the real Aspen version but 
    returns fabricated results instead of calling Aspen.
    """

    def __init__(self, AspenFile, wdpath, visibility=False):
        self.AspenFile = AspenFile
        self.wdpath = wdpath
        self.visibility = visibility
        print("Dummy mode: Initialized distillation column simulation. No Aspen calls will be made.")

    def reset(self):
        print("Dummy mode: reset() called (nothing to reset).")

    @staticmethod
    def flatten_params(x_dict):
        """
        Flatten dictionary format into a NumPy array.
        Here we assume that the dictionary uses the 'RadFrac' block.
        """
        flat_list = []
        for block_type in ["RadFrac"]:
            for block, params in x_dict[block_type].items():
                flat_list.extend(params)
        return np.array(flat_list)

    @staticmethod
    def unflatten_params(flat_array):
        """
        Convert a flat NumPy array back to dictionary format.
        For the dummy we assume that the input consists of two parameters—
        one for each of two blocks: 'RADFRAC1' and 'RADFRAC2'.
        """
        # flat_array = flat_array[0]
        print("flat_array:", flat_array)
        if len(flat_array) != 8:
            raise ValueError("Expected flat array of length 8, got {}.".format(len(flat_array)))
        return {
            "RadFrac": {
                "RADFRAC1": flat_array[:4].tolist(),
                "RADFRAC2": flat_array[4:8].tolist()
            }
        }

    def open_simulation(self):
        print("Dummy mode: open_simulation() called. (No Aspen simulation initialized.)")

    def close_simulation(self):
        print("Dummy mode: close_simulation() called. (Nothing to close.)")

    def runSim(self, x):
        """
        Instead of running an Aspen simulation, this dummy function computes 
        fabricated results based on the input parameters.
        
        We assume x has the following structure:
            x = {"RadFrac": {"RADFRAC1": [p1], "RADFRAC2": [p2]}}
        where p1 and p2 are some parameters that influence the column design.
        """
        try:
            rad_params = x["RadFrac"]
            p1 = rad_params.get("RADFRAC1", [10])[0]
            p2 = rad_params.get("RADFRAC2", [12])[0]
        except Exception as e:
            print("Dummy mode: Could not extract parameters, using defaults. Error:", e)
            p1, p2 = 10, 12

        # Compute dummy results (these formulas are arbitrary)
        results = {
            "COL_1_DIAM": 1.0 + 0.1 * p1,
            "COL_2_DIAM": 1.2 + 0.1 * p2,
            "COL_1_HEIGHT": 10.0 + p1,
            "COL_2_HEIGHT": 11.0 + p2,
            "COL_1_HEAT_UTIL": 100.0 + p1,
            "COL_2_HEAT_UTIL": 105.0 + p2,
            "COL_1_COOL_UTIL": 90.0 + p1,
            "COL_2_COOL_UTIL": 95.0 + p2,
            "COL_1_REBOILER_DUTY": 5000 + 50 * p1,
            "COL_2_REBOILER_DUTY": 5200 + 50 * p2,
        }
        return results

    def calc_column_cap_cost(self, height, diameter):
        # Use the same formula as the real version (dummy formula)
        return 17640 * (diameter ** 1.066) * (height ** 0.802)

    def calc_tac(self, results):
        # Calculate a dummy total annual cost (TAC) based on fabricated results.
        col1_capital = self.calc_column_cap_cost(results["COL_1_HEIGHT"], results["COL_1_DIAM"])
        col2_capital = self.calc_column_cap_cost(results["COL_2_HEIGHT"], results["COL_2_DIAM"])
        operating_cost_1 = (results["COL_1_HEAT_UTIL"] + results["COL_1_COOL_UTIL"]) * 8000
        operating_cost_2 = (results["COL_2_HEAT_UTIL"] + results["COL_2_COOL_UTIL"]) * 8000
        tac = (col1_capital + col2_capital) / 3 + operating_cost_1 + operating_cost_2
        return tac

    def calc_co2_emission(self, results):
        # Compute dummy CO2 emissions from reboiler duties
        col1_emission = ((results["COL_1_REBOILER_DUTY"] / 0.8) / 22000) * 0.0068 * 3.67 * 8000 * 3600
        col2_emission = ((results["COL_2_REBOILER_DUTY"] / 0.8) / 22000) * 0.0068 * 3.67 * 8000 * 3600
        total_emission = col1_emission + col2_emission
        return total_emission

    def costFunc(self, results):
        tac = self.calc_tac(results)
        co2_emission = self.calc_co2_emission(results)
        return tac, co2_emission

    def run_obj(self, x):
        res = self.runSim(x)
        return self.costFunc(res)
 
   