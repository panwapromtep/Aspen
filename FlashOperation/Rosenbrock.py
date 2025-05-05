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

class Rosenbrock(AspenSim):
    def __init__(self, AspenFile=None, wdpath=None, visibility=False):
        """
        Dummy that replaces an Aspen run with the Rosenbrock function.
        Signature matches your real AspenSim class.
        """
        print("⚠️  Dummy Mode: Rosenbrock function in place of AspenSim.")

    @staticmethod
    def flatten_params(x_dict):
        """
        Flatten the same dict structure your real sim expects:
        { "Flash2": { "FLASH1": [x1], "FLASH2": [x2] } }
        """
        return np.array([
            x_dict["Flash2"]["FLASH1"][0],
            x_dict["Flash2"]["FLASH2"][0]
        ], dtype=float)

    @staticmethod
    def unflatten_params(flat_array):
        """
        Wrap a flat [x1, x2] back into the dict format.
        """
        return {
            "Flash2": {
                "FLASH1": [float(flat_array[0])],
                "FLASH2": [float(flat_array[1])]
            }
        }

    def open_simulation(self):
        """No‐op (skip Aspen init)."""
        pass

    def close_simulation(self):
        """No‐op (skip Aspen cleanup)."""
        pass

    def runSim(self, x):
        """
        Instead of Aspen, compute Rosenbrock:
          f = (1 - x1)^2 + 100*(x2 - x1^2)^2
        and return it in a results dict under 'DUMMY_COST'.
        """
        x1 = x["Flash2"]["FLASH1"][0]
        x2 = x["Flash2"]["FLASH2"][0]

        cost_value = (1 - x1)**2 + 100 * (x2 - x1**2)**2

        return { "DUMMY_COST": cost_value }

    def costFunc(self, results):
        """
        Pull out the single‐objective cost.
        """
        return results["DUMMY_COST"]

    def run_obj(self, x):
        """
        High‐level entrypoint, matching your surrogate loop’s expectations.
        """
        return self.costFunc(self.runSim(x))
