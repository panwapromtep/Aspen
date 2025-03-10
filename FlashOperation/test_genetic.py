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
        
        super().__init__(n_var=2, n_obj=1, xl = [1, 1], xu = [20, 20])
        self.assSim = assSim

    def _evaluate(self, x, out, *args, **kwargs):  
        x_eval = {
            "Flash2": {'FLASH1': [x[0]], 'FLASH2': [x[1]]}
            }
        
        out["F"] = np.array([self.assSim.run_obj(x_eval)])
        
        

def main():
    print("here")
    print(os.getcwd())
    assSim = Refrig2DrumConstraintHeatExConstr(AspenFile = "FlashOperation.bkp", 
                                   wdpath = "../FlashOperation", 
                                   visibility=False,
                                   Penalty=1e5
                                   )
    
    problem = AspenProblem(assSim)
    algorithm = GA(pop_size=10, eliminate_duplicates=True)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 10),
                   verbose=True,
                   save_history=True)
    
    total_exec_time = res.exec_time  # Get the total execution time
    
    print("Best solution found: %s" % res.X)
    x_eval = {
        "Flash2": {'FLASH1': [res.X[0]], 'FLASH2': [res.X[1]]}
        }
    assSim.run_obj(x_eval)
    print("Function value: %s" % res.F)
    print("Constraint violation: %s" % res.CV)
    print("Execution time: %s seconds" % total_exec_time)
    
    #plot the convergence
    
    # Extract average and minimum objective values for each generation
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
    df.to_excel('results.xlsx', index=False, sheet_name='Results')
    
    # Get the final minimum objective value and corresponding design space values
    final_min_obj_value = res.F[0]
    final_design_space_values = res.X
    
    # Plot average and minimum objective value vs. generation
    plt.plot(range(1, len(avg_obj_values) + 1), avg_obj_values, marker='o', label='Average Objective Value')
    plt.plot(range(1, len(min_obj_values) + 1), min_obj_values, marker='x', label='Minimum Objective Value')
    plt.xlabel('Generation')
    plt.ylabel('Objective Value')
    plt.title('Objective Value vs. Generation')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Add text to the graph
    textstr = f'Final Min Obj Value: {final_min_obj_value:.4f}\n Design Space Values: {final_design_space_values}'
    plt.gcf().text(0.15, 0.75, textstr, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
    
if __name__ == "__main__":
    main()