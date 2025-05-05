# -*- coding: utf-8 -*-
"""
Created on Mon Mar 3 13:36:14 2025

"""
from VCDistillation import VCDistillation
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
import time  # Import the time module
import pandas as pd  # Import pandas for saving results to Excel
import pickle  # Import pickle for saving data
from datetime import datetime  # Import datetime for timestamping files

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

    #? The order of the paremeters in the dicitonary is as follows:
    #? [number of stages, [feed stage, feed stream name], reflux ratio, distillate to feed ratio]

class AspenProblem(ElementwiseProblem):
    def __init__(self, assSim):
        super().__init__(n_var=8, n_obj=2, n_ieq_constr=2, xl=[30, 2, 0.1, 0.4, 34, 2, 0.1, 0.7], xu=[45, 35, 2, 0.5, 45, 37, 2.0, 0.95])
        self.assSim = assSim

    def _evaluate(self, x, out, *args, **kwargs):  
        # punish solutions where feed stage is greater than number of stages
        if round(x[1]) > round(x[0]) or round(x[5]) > round(x[4]):
            out["F"] = np.array([1E3, 1E3])
            out["G"] = np.array([1E3, 1E3])
            return
        
        x_eval = self.assSim.unflatten_params(x)
        results = self.assSim.runSim(x_eval)
        
        # Objectives        
        cost = self.assSim.costFunc(results)
        
        # Constraints
        acetylene_purity = results["ACETYLENE_PURITY"]
        vinyl_chloride_purity = results["VC_PURITY"]
        g1 = acetylene_purity - 0.00005  # Ensure acetylene purity < 0.005
        g2 = 0.9999 - vinyl_chloride_purity  # Ensure vinyl chloride purity > 0.99
    
        
        out["F"] = np.array([cost[2], cost[3]])  # Add penalty to TAC
        out["G"] = np.array([g1, g2])
    
def main():
    print("Starting NSGAIII optimization with VCDistillation.")
    assSim = VCDistillation(AspenFile="Vinyl Chloride Distillation.bkp", 
                             wdpath="../Vinyl_Distillation", 
                             visibility=False)
    
    problem = AspenProblem(assSim)
    #ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=10)  # Use pymoo's ref_dirs
    algorithm = NSGA3(pop_size=15)
    
    start_time = time.time()
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 20),
                   verbose=True,
                   save_history=True)
    
    
    end_time = time.time()
    
    print("Optimization completed.")
    print("Execution time: %.2f seconds" % (end_time - start_time))
    
    # Generate a timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract the best TAC, CO2 emissions, and parameters from the last generation
    last_gen = res.history[-1]
    last_gen_F = last_gen.pop.get("F")
    last_gen_X = last_gen.pop.get("X")
    best_idx = np.argmin(last_gen_F[:, 0])  # Index of the best TAC
    best_tac = last_gen_F[best_idx, 0]  # Best TAC
    best_co2 = last_gen_F[best_idx, 1]  # Corresponding CO2 emissions
    best_params = last_gen_X[best_idx]  # Corresponding parameters

    print(f"Last Generation: Best TAC = {best_tac:.4f}, Best CO2 Emissions = {best_co2:.4f}")
    print("Parameters for Best TAC and CO2 Emissions:")
    print(f"  Number of Stages (Column 1): {best_params[0]:.2f}")
    print(f"  Feed Stage (Column 1): {best_params[1]:.2f}")
    print(f"  Reflux Ratio (Column 1): {best_params[2]:.2f}")
    print(f"  Distillate to Feed Ratio (Column 1): {best_params[3]:.2f}")
    print(f"  Number of Stages (Column 2): {best_params[4]:.2f}")
    print(f"  Feed Stage (Column 2): {best_params[5]:.2f}")
    print(f"  Reflux Ratio (Column 2): {best_params[6]:.2f}")
    print(f"  Distillate to Feed Ratio (Column 2): {best_params[7]:.2f}")

    # Save the results for each individual in each generation to a pickle file
    all_generations_data = []
    for gen_idx, gen in enumerate(res.history):
        F = gen.pop.get("F")  # Get objective values for the generation
        X = gen.pop.get("X")  # Get design variables for the generation
        for i in range(len(F)):
            all_generations_data.append({
                "Generation": gen_idx + 1,
                "Design Variables": X[i],
                "TAC (Objective 1)": F[i, 0],
                "CO2 Emissions (Objective 2)": F[i, 1]
            })
    
    pickle_filename = f"data_100gen"
    with open(pickle_filename, "wb") as f:
        pickle.dump(all_generations_data, f)
    print(f"All generations data saved to '{pickle_filename}'.")


if __name__ == "__main__":
    main()
