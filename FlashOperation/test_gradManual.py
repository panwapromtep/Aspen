# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:36:14 2025

@author: ppromte1
"""

from Refrig2Drum2Comp import Refrig2Drum2Comp
import os, sys
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

from grad_ import gradMoment

def main():
    assSim = Refrig2Drum2Comp(AspenFile = "FlashOperation.bkp", wdpath = "../FlashOperation")

    x_dict = {
        "Flash2": {"FLASH1": [48.9, 20.7], "FLASH2": [10, 12.4]},
        "Heater": {"COOLER1": [4.4, 22.1], "COOLER2": [15.6, 36.5]},
        "Compr": {"COMP1": [22.3], "COMP2": [37.2]}
    }

    x_init = assSim.flatten_params(x_dict)

    optimizer = gradMoment(assSim)
    print("here")
    
    best_params, best_obj = optimizer.optimize(x_init, max_iter=1)

    print("Optimized Parameters:", best_params)
    print("Best Objective Value:", best_obj)


if __name__ == "__main__":
    main()
