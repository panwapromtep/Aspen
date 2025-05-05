import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

from AspenSim import AspenSim

class VCDistillationToy(AspenSim):
    def __init__(self, AspenFile=None, wdpath=None, visibility=False):
        """
        Dummy that replaces an Aspen run with a multi-objective toy function.
        Signature matches your real AspenSim class.
        """
        print("⚠️  Dummy Mode: Multi-objective toy function in place of AspenSim.")

    @staticmethod
    def flatten_params(x_dict):
        """
        Flatten the same dict structure your real sim expects:
        { "RadFrac": { "RADFRAC1": [x1, x2], "RADFRAC2": [x3, x4] } }
        """
        return np.array([
            x_dict["RadFrac"]["RADFRAC1"][0],
            x_dict["RadFrac"]["RADFRAC1"][1],
            x_dict["RadFrac"]["RADFRAC2"][0],
            x_dict["RadFrac"]["RADFRAC2"][1]
        ], dtype=float)

    @staticmethod
    def unflatten_params(flat_array):
        """
        Wrap a flat [x1, x2, x3, x4] back into the dict format.
        """
        return {
            "RadFrac": {
                "RADFRAC1": [float(flat_array[0]), float(flat_array[1])],
                "RADFRAC2": [float(flat_array[2]), float(flat_array[3])]
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
        Compute the Binh and Korn multi-objective problem:
          f1 = 4*x1^2 + 4*x2^2
          f2 = (x1 - 5)^2 + (x2 - 5)^2
        and return them in a results dict.
        """
        x1, x2, x3, x4 = x["RadFrac"]["RADFRAC1"] + x["RadFrac"]["RADFRAC2"]

        # Compute objectives
        f1 = 4 * x1**2 + 4 * x2**2
        f2 = (x1 - 5)**2 + (x2 - 5)**2

        # Compute constraints
        c1 = (x1 - 5)**2 + x2**2 - 25  # Should be <= 0
        c2 = 7.7 - ((x1 - 8)**2 + (x2 + 3)**2)  # Should be <= 0

        return { "OBJ_1": f1, "OBJ_2": f2, "CONSTR_1": c1, "CONSTR_2": c2 }

    def costFunc(self, results):
        """
        Return the two objectives as a tuple.
        """
        return results["OBJ_1"], results["OBJ_2"]

    def constraintFunc(self, results):
        """
        Return the constraints as a tuple.
        """
        return results["CONSTR_1"], results["CONSTR_2"]

    def run_obj(self, x):
        """
        High‐level entrypoint, matching your surrogate loop’s expectations.
        """
        results = self.runSim(x)
        return self.costFunc(results), self.constraintFunc(results)

def main():
    # Define the bounds for x1 and x2
    x1_bounds = np.linspace(0, 5, 150)  # Reduce from 500 to 100
    x2_bounds = np.linspace(0, 3, 150)  # Reduce from 500 to 100
    
    # Create a grid of x1 and x2 values
    x1, x2 = np.meshgrid(x1_bounds, x2_bounds)
    x1_flat = x1.flatten()
    x2_flat = x2.flatten()
    
    print("Grid created successfully.")
    
    # Compute the objective functions
    f1 = 4 * x1_flat**2 + 4 * x2_flat**2
    f2 = (x1_flat - 5)**2 + (x2_flat - 5)**2
    
    # Compute the constraints
    c1 = (x1_flat - 5)**2 + x2_flat**2 - 25  # Should be <= 0
    c2 = 7.7 - ((x1_flat - 8)**2 + (x2_flat + 3)**2)  # Should be <= 0
    
    # Filter feasible solutions
    feasible = (c1 <= 0) & (c2 <= 0)
    f1_feasible = f1[feasible]
    f2_feasible = f2[feasible]
    
    print("Feasible solutions filtered.")
    
    # Combine feasible objective values for non-dominated sorting
    F = np.vstack((f1_feasible, f2_feasible)).T
    
    # Identify the Pareto front
    non_dominated_sorting = NonDominatedSorting()
    pareto_indices = non_dominated_sorting.do(F, only_non_dominated_front=True)
    pareto_f1 = F[pareto_indices, 0]
    pareto_f2 = F[pareto_indices, 1]
    
    # Plot the feasible region and Pareto front
    plt.figure(figsize=(12, 6))
    
    # Left plot: Feasible region in decision space
    plt.scatter(x1_flat[feasible], x2_flat[feasible], s=1, color="blue", label="Feasible Region")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Feasible Region with Pareto Set")
    plt.legend()
    plt.grid(True)
    
    # Right plot: Objective space with Pareto front
    plt.figure(figsize=(8, 5))
    plt.scatter(f1_feasible, f2_feasible, s=15, color="blue", label="Feasible Objectives")
    plt.scatter(pareto_f1, pareto_f2, s=10, color="red", label="Pareto Front")
    plt.xlabel("Objective Function 1")
    plt.ylabel("Objective Function 2")
    plt.title("Objective Space with Pareto Front")
    plt.legend()
    plt.grid(True)
    
    # Show the plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
