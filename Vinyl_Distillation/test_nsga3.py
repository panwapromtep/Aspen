# -*- coding: utf-8 -*-
"""
Created on Mon Mar 3 13:36:14 2025

"""
import matplotlib
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

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

    #? The order of the paremeters in the dicitonary is as follows:
    #? [number of stages, [feed stage, feed stream name], reflux ratio, distillate to feed ratio]

class AspenProblem(ElementwiseProblem):
    def __init__(self, assSim):
        super().__init__(n_var=8, n_obj=2, n_ieq_constr=2, xl=[30, 2, 0.1, 0.46, 35, 2, 0.1, 0.89], xu=[36, 29, 1.5, 0.48, 42, 34, 1.5, 0.91])
        self.assSim = assSim

    def _evaluate(self, x, out, *args, **kwargs):  
        # punish solutions where feed stage is greater than number of stages
        if x[1] > x[0] or x[5] > x[4]:
            out["F"] = np.array([1E3, 1E3])
            out["G"] = np.array([1E3, 1E3])
            return
        
        x_eval = self.assSim.unflatten_params(x)
        results = self.assSim.runSim(x_eval)
        
        # Objectives        
        cost = self.assSim.costFunc(results)
        print("Cost:", cost)
        out["F"] = np.array([cost[2], cost[3]])  # TAC and CO2 emission as objectives

        # Constraints
        acetylene_purity = results["ACETYLENE_PURITY"]
        vinyl_chloride_purity = results["VC_PURITY"]
        out["G"] = np.array([
            acetylene_purity - 0.005,  # Ensure acetylene purity < 0.005
            0.99 - vinyl_chloride_purity  # Ensure vinyl chloride purity > 0.99
        ])
    
def main():
    print("Starting NSGAIII optimization with VCDistillation.")
    assSim = VCDistillation(AspenFile="Vinyl Chloride Distillation.bkp", 
                             wdpath="../Vinyl_Distillation", 
                             visibility=False)
    
    problem = AspenProblem(assSim)
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)  # Use pymoo's ref_dirs
    algorithm = NSGA3(pop_size=10, ref_dirs=ref_dirs)
    
    start_time = time.time()
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 10),
                   verbose=True,
                   save_history=True)
    
    
    end_time = time.time()
    print("Execution time: %.2f seconds" % (end_time - start_time))
    

    # Extract the best objective function value for each generation
    best_obj_values = []
    for gen in res.history:
        F = gen.pop.get("F")  # Get objective values for the generation
        best_obj_values.append(np.min(F[:, 0]))  # Track the best value for the first objective (TAC)

    # Plot the objective value vectors [TAC, CO2 Emissions] for each generation
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(res.history)))  # Generate a color for each generation

    for gen_idx, gen in enumerate(res.history):
        F = gen.pop.get("F")  # Get objective values for the generation
        plt.scatter(F[:, 0], F[:, 1], color=colors[gen_idx], label=f"Generation {gen_idx + 1}")

    plt.xlabel("Total Annual Cost (TAC)")
    plt.ylabel("CO2 Emissions")
    plt.title("Objective Values Across Generations")
    plt.legend()
    plt.grid(True)
    plt.savefig("objective_values_across_generations.png")  # Save the plot as a PNG file
    print("Plot saved as 'objective_values_across_generations.png'.")
    plt.show()

if __name__ == "__main__":
    main()
