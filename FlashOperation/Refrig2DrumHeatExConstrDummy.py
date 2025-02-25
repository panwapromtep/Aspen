# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:39:46 2025

@author: ppromte1
"""

import sys
import os
import numpy as np

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)


from AspenSim import AspenSim
# from CodeLibrary import Simulation
from FlashOperation.Refrig2Drum2Comp import Refrig2Drum2Comp

class Refrig2DrumConstraintHeatExConstDummy(Refrig2Drum2Comp):
    def __init__(self, AspenFile=None, wdpath=None, visibility=False, Penalty=1e4):
        """
        Dummy version that replaces Aspen simulation with a simple function x1^2 + x2^2
        and applies a temperature constraint based on x1 + x2.
        """
        super().__init__(AspenFile, wdpath, visibility)
        self.Penalty = Penalty  # Still included for compatibility

    @staticmethod
    def flatten_params(x_dict):
        """
        Flatten dictionary format into a NumPy array.
        """
        flat_list = []
        for block_type in ["Flash2"]:
            for block, params in x_dict[block_type].items():
                flat_list.extend(params)
        return np.array(flat_list)

    @staticmethod
    def unflatten_params(flat_array):
        """
        Convert flat NumPy array back to dictionary format.
        """
        x_dict = {
            "Flash2": {"FLASH1": [flat_array[0]], "FLASH2": [flat_array[1]]},
        }
        return x_dict

    def runSim(self, x):
        """
        Instead of running Aspen, return a dummy function evaluation: x1^2 + x2^2
        and set temperature constraint based on x1 + x2.
        """
        x1, x2 = x["Flash2"]["FLASH1"][0], x["Flash2"]["FLASH2"][0]

        # Compute the dummy objective function
        cost_value = (x1)**2 + (x2)**2

        # Set TEMPOUT based on condition
        if x1 + x2 > 40:
            temp_out = 200
        else:
            temp_out = -20

        results = {
            "DUMMY_COST": cost_value,
            "TEMPOUT": temp_out
        }
        return results

    def costFunc(self, results):
        """
        Compute the cost function based on the dummy output, with temperature penalty.
        """
        cost = results["DUMMY_COST"]  # Main cost function

        # Apply penalty if TEMPOUT > -28.9
        # if results["TEMPOUT"] > -28.9:
        #     cost += self.Penalty * (results["TEMPOUT"] + 28.9) ** 2

        return cost
    
    def open_simulation(self):
        """Override this method so it does nothing, preventing Aspen from being called."""
        print("⚠️ Dummy Mode: Skipping Aspen simulation initialization.")
