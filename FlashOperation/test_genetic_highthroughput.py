# -*- coding: utf-8 -*-
"""
Created on Mon Mar 3 13:36:14 2025

@author: wsangpa1
"""

from Refrig2DrumHeatExConstr import Refrig2DrumConstraintHeatExConstr
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
import pandas as pd

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

class AspenProblem(ElementwiseProblem):
    def __init__(self, assSim):
        
        super().__init__(n_var=2, n_obj=1, xl = [1, 1], xu = [25, 25])
        self.assSim = assSim

    def _evaluate(self, x, out, *args, **kwargs):  
        x_eval = {
            "Flash2": {'FLASH1': [x[0]], 'FLASH2': [x[1]]}
            }
        
        out["F"] = np.array([self.assSim.run_obj(x_eval)])
        
        

def run_experiment(gen_size, assSim):
    problem = AspenProblem(assSim)
    #algorithm = GA(pop_size=pop_size, eliminate_duplicates=True)
    algorithm = GA(pop_size=5, eliminate_duplicates=True)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', gen_size),
                   verbose=True,
                   save_history=True)
    
    total_exec_time = res.exec_time  # Get the total execution time
    
    # Extract average and minimum objective values  for each generation
    avg_obj_values = []
    min_obj_values = []
    min_design_space_values = []
    for gen in res.history:
        F = gen.pop.get("F")
        X = gen.pop.get("X")
        avg_obj_values.append(np.mean(F))
        min_obj_values.append(np.min(F))
        min_design_space_values.append(X[np.argmin(F)])
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Generation': range(1, len(avg_obj_values) + 1),
        'Average Objective Value': avg_obj_values,
        'Minimum Objective Value': min_obj_values,
        'Minimum Design Space Value': min_design_space_values,
        'Total Execution Time (s)': [total_exec_time] * len(avg_obj_values)  # Add total execution time to the DataFrame
    })
    
    # Save the DataFrame to an Excel file
    #df.to_excel(f'results_pop_size_{pop_size}_10gen.xlsx', index=False, sheet_name='Results')
    df.to_excel(f'results_pop_size_5_{gen_size}gen.xlsx', index=False, sheet_name='Results')

    
    
def main():
    assSim = Refrig2DrumConstraintHeatExConstr(AspenFile="FlashOperation.bkp", 
                                               wdpath="../FlashOperation", 
                                               visibility=False,
                                               Penalty=1e2)
    '''
    pop_sizes = [3, 5, 10, 15]
    for pop_size in pop_sizes:
        run_experiment(pop_size, assSim)
    '''
        
    gen_sizes = [3, 5, 10, 15]
    for gen_size in gen_sizes:
        run_experiment(gen_size, assSim)
    
    
    
    assSim.close_simulation()

if __name__ == "__main__":
    main()