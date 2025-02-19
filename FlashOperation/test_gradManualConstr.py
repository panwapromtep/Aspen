# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:36:14 2025

@author: ppromte1
"""

from Refrig2DrumHeatExConstr import Refrig2DrumConstraintHeatExConstr
from Refrig2DrumHeatExConstrDummy import Refrig2DrumConstraintHeatExConstDummy
from Refrig2Drum2Comp import Refrig2Drum2Comp
import os, sys
import matplotlib.pyplot as plt
import numpy as np
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

from grad_ import gradMoment

def main():
    print("here")
    print(os.getcwd())
    assSim = Refrig2DrumConstraintHeatExConstr(AspenFile = "FlashOperation.bkp", 
                                   wdpath = "../FlashOperation", 
                                   visibility=False,
                                   Penalty=1e2
                                   )
    
    x_dict = {
        "Flash2": {"FLASH1": [7], "FLASH2": [1.2]}
    }

    x_init = assSim.flatten_params(x_dict)
    
    print("x_init obj:", assSim.run_obj(assSim.unflatten_params(x_init)))

    optimizer = gradMoment(assSim,
                           minmax=[np.array([0, 0]), 
                                   np.array([34, 34])]
                           )
    best_params, best_obj, obj_path = optimizer.optimize(x_init, max_iter=200,
                                                         alpha = 1e-2,
                                                         beta = 0.01,
                                                         epsilon=1e-5,
                                                         patience = 30,
                                                         obj_norm=1e4
                                                         )

    print("Optimized Parameters:", best_params)
    print("Best Objective Value:", best_obj)
    print("obj path\n", obj_path)
    
    # Plot the objective path
    plt.plot(obj_path, marker='o', linestyle='-')
    plt.title('Objective Function Path')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True)
    plt.show()
 

if __name__ == "__main__":
    main()

