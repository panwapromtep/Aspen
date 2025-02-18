# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:36:14 2025

@author: ppromte1
"""

from Refrig2Drum2Comp import Refrig2Drum2Comp
import os, sys
import matplotlib.pyplot as plt
import numpy as np
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)
import time

from grad_ import gradMoment



def testRecalculating():
    # Initialize the Aspen simulation
    assSim = Refrig2Drum2Comp(AspenFile="FlashOperation.bkp", 
                              wdpath="../FlashOperation",
                              visibility=True)

    # Define 3 distinct parameter sets
    x_dicts = [
        {
            "Flash2": {"FLASH1": [7], "FLASH2": [1.2]},
            "Heater": {"COOLER1": [25, 34.478], "COOLER2": [25, 34.478]},
            "Compr": {"COMP1": [40], "COMP2": [40]}
        },
        {
            "Flash2": {"FLASH1": [5], "FLASH2": [1.5]},
            "Heater": {"COOLER1": [22, 30.000], "COOLER2": [23, 31.000]},
            "Compr": {"COMP1": [35], "COMP2": [38]}
        },
        {
            "Flash2": {"FLASH1": [6.5], "FLASH2": [1.3]},
            "Heater": {"COOLER1": [26, 35.000], "COOLER2": [26, 35.000]},
            "Compr": {"COMP1": [42], "COMP2": [43]}
        }
    ]

    # Number of cycles
    cycles = 3

    # Dictionary to store results
    results = {i: [] for i in range(len(x_dicts))}

    # Run calculations in cycles
    for cycle in range(cycles):
        print(f"\n🔄 Starting cycle {cycle+1}/{cycles}...")
        for i, x_dict in enumerate(x_dicts):
            x = assSim.flatten_params(x_dict)

            # Calculate the objective function
            obj = assSim.run_obj(assSim.unflatten_params(x))
            print(f"🔍 Cycle {cycle+1}, Set {i+1}: Objective = {obj}")

            # Store the result
            results[i].append(obj)
            
            assSim.reset()
            # Wait to ensure stability
            time.sleep(10)

    # Close Aspen simulation
    #assSim.close_simulation()

    # Compare the results across cycles
    for i, obj_list in results.items():
        print(f"\n🔢 Results for parameter set {i+1}: {obj_list}")

        # Check if all calculated values are close to the first one
        first_value = obj_list[0]
        for j, value in enumerate(obj_list):
            if not np.isclose(first_value, value, rtol=1e-5, atol=1e-5):
                raise ValueError(
                    f"🚨 Objective function mismatch for set {i+1}!\n"
                    f"Expected ~{first_value}, but got {value} on cycle {j+1}\n"
                    f"All results: {obj_list}"
                )

    print("\n🎉 All tests passed! Objective function calculations are consistent across cycles.")
    

def main():
    assSim = Refrig2Drum2Comp(AspenFile = "FlashOperation.bkp", 
                              wdpath = "../FlashOperation",
                              visibility=False
                              )
    
    x_dict = {
        "Flash2": {"FLASH1": [7], "FLASH2": [1.2]},
        "Heater": {"COOLER1": [25, 34.478], "COOLER2": [25, 34.478]}, #temp, pressure
        "Compr": {"COMP1": [40], "COMP2": [40]}
    }

    x = assSim.flatten_params(x_dict)
    
    print("now run grad approx but dont update x")
    obj_path = []
    optimizer = gradMoment(assSim)
    
    for i in range(1):
        obj = assSim.run_obj(assSim.unflatten_params(x))
        obj_path.append(obj)
        print(f"Iteration {i}: Objective = {obj}, x = {x}")
        grad = optimizer.grad_approx(x)
        
    plt.plot(obj_path, marker='o', linestyle='-')
    plt.title('Objective Function Path')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.show()
    
    assSim.close_simulation()
    
if __name__ == "__main__":
    #testRecalculating()
    main()

