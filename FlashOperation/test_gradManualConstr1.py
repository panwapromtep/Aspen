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
import psutil
import time

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

from grad_1 import gradMoment

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
    
    # Start measuring CPU and memory usage
    start_time = time.time()
    cpu_usage = []
    memory_usage = []
    timestamps = []
    
    def sample_usage():
        cpu_usage.append(psutil.cpu_percent(interval=None))
        memory_usage.append(psutil.virtual_memory().used / (1024 * 1024))  # Convert to MB
        timestamps.append(time.time() - start_time)
    
    # Sample initial usage
    sample_usage()
    
    best_params, best_obj, obj_path = optimizer.optimize(x_init, max_iter=200,
                                                         alpha = 1e-2,
                                                         beta = 0.01,
                                                         epsilon=1e-5,
                                                         patience = 30,
                                                         obj_norm=1e4,
                                                         callback=sample_usage
                                                         )
    
    # Sample final usage
    sample_usage()
    
    end_time = time.time()
    
    print("Optimized Parameters:", best_params)
    print("Best Objective Value:", best_obj)
    print("obj path\n", obj_path)
    
    # Print CPU and memory usage
    print("Total execution time: %.2f seconds" % (end_time - start_time))
    
    # Plot CPU and memory usage over time
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('CPU Usage (%)', color='tab:blue')
    ax1.plot(timestamps, cpu_usage, color='tab:blue', label='CPU Usage')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Memory Usage (MB)', color='tab:green')
    ax2.plot(timestamps, memory_usage, color='tab:green', label='Memory Usage')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    plt.title('Gradient Descent CPU and Memory Usage Over Time')
    plt.show()
    
    # Plot the objective path
    plt.plot(obj_path, marker='o', linestyle='-')
    plt.title('Objective Function Path')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True)
    plt.show()
 

if __name__ == "__main__":
    main()