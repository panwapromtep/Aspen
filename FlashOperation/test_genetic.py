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

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

class AspenProblem(ElementwiseProblem):
    def __init__(self, assSim):
        
        super().__init__(n_var=2, n_obj=1, xl = [-100, -100], xu = [100, 100])
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
                                   Penalty=1e2
                                   )
    
    problem = AspenProblem(assSim)
    algorithm = GA(pop_size=10, eliminate_duplicates=True)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 10),
                   verbose=True,
                   save_history = True)
    

    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)
    print("Constraint violation: %s" % res.CV)
    print("Execution time: %s seconds" % res.exec_time)
    
    #plot the convergence
    
    # Extract average objective values for each generation
    avg_obj_values = []
    for gen in res.history:
        F = gen.pop.get("F")
        avg_obj_values.append(np.mean(F))
    
    # Get the final minimum objective value and corresponding design space values
    final_min_obj_value = res.F[0]
    final_design_space_values = res.X
    
    # Plot average objective value vs. generation
    plt.plot(avg_obj_values, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Average Objective Value')
    plt.title('Average Objective Value vs. Generation')
    plt.grid(True)
    
    # Add text to the graph
    textstr = f'Final Min Obj Value: {final_min_obj_value:.4f}\n Design Space Values: {final_design_space_values}'
    plt.gcf().text(0.15, 0.75, textstr, fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
    
if __name__ == "__main__":
    main()