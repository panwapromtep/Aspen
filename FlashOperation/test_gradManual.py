# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:36:14 2025

@author: ppromte1
"""

from Refrig2Drum2Comp import Refrig2Drum2Comp
import os, sys
import matplotlib.pyplot as plt
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

from grad_ import gradMoment

def main():
    assSim = Refrig2Drum2Comp(AspenFile = "FlashOperation.bkp", wdpath = "../FlashOperation", visibility=False)
    
    x_dict = {
        "Flash2": {"FLASH1": [7], "FLASH2": [1.2]},
        "Heater": {"COOLER1": [25, 34.478], "COOLER2": [25, 34.478]}, #temp, pressure
        "Compr": {"COMP1": [40], "COMP2": [40]}
    }

    x_init = assSim.flatten_params(x_dict)

    optimizer = gradMoment(assSim)
    best_params, best_obj, obj_path = optimizer.optimize(x_init, max_iter=50,
                                                         alpha = 1e-4,
                                                         beta = 0.9,
                                                         patience = 5)

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

