# Gradient Descent Module
import pickle
import numpy as np
from AspenSim import AspenSim
from typing import Callable
import unittest
import time
from FlashOperation.Refrig2Drum2Comp import Refrig2Drum2Comp

class gradMinMax():
    def __init__(self, 
                 sim: AspenSim,
                 x_range: list[tuple]):
        self.sim = sim
        sekf.

    def optimize(self, x_init: np.ndarray, alpha=1e-4, beta=0.9, epsilon=1e-4, max_iter=1000, patience=10):
        """
        Perform gradient descent optimization with momentum.

        Args:
            x_init (np.ndarray): Initial guess for the parameters.
            alpha (float, optional): Learning rate. Defaults to 1e-4.
            beta (float, optional): Momentum factor. Defaults to 0.9.
            epsilon (float, optional): Convergence threshold for the gradient norm. Defaults to 1e-4.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
            patience (int, optional): Number of iterations to wait for improvement before early stopping. Defaults to 10.

        Returns:
            tuple: Best parameters found, best objective value, and the path of objective values during optimization.
        """
        x = x_init.copy()
        v = np.zeros_like(x)
        t = 0
        best_x = x.copy()
        best_obj = float('inf')
        patience_counter = 0
        obj_path = []
    
        while t < max_iter:
            t += 1
            #print("hi")
            self.sim.reset()
            #print("or hi")
            grad = self.grad_approx(x)
            # grad = 0
            v = beta * v + (1 - beta) * grad
            x = x - alpha * v 
            print("new x", x)
            self.sim.reset()
            obj = self.sim.run_obj(self.sim.unflatten_params(x))
            obj_path.append(obj)

            if obj < best_obj:
                best_obj = obj
                best_x = x.copy()
                patience_counter = 0
            else:
                patience_counter += 1
    
            if patience_counter >= patience:
                print(f"Early stopping at iteration {t}")
                break
    
            if np.linalg.norm(grad) < epsilon:
                print(f"Convergence achieved at iteration {t}")
                break
    
        # Close Aspen after the optimization is complete
        self.sim.close_simulation()
    
        return self.sim.unflatten_params(best_x), best_obj, obj_path


    
    
    def grad_approx(self, x, h=1e-2):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
    
            # Initial objective calculation (already present in your code)
            #objx = self.sim.run_obj(self.sim.unflatten_params(x))
            #print(f"Objective at x ({x}): {objx}")
    
            #print(f"\n==== Gradient Calculation for index {i} ====")
            #print(f"x_plus: {x_plus}")
            #print(f"x_minus: {x_minus}")
    
            try:
                # Calculate obj_plus
                x_plus_flat = self.sim.unflatten_params(x_plus)
                #print("x_plus_unflat", x_plus_flat)
                obj_plus = self.sim.run_obj(x_plus_flat)
                #print(f"obj_plus: {obj_plus}")
    
                # Wait 8 seconds
                self.sim.reset()
                #print("Wait before obj_minus...")
                #time.sleep(8)
    
                # Calculate obj_minus
                x_minus_flat = self.sim.unflatten_params(x_minus)
                obj_minus = self.sim.run_obj(x_minus_flat)
                #print(f"obj_minus: {obj_minus}")
    
    
                # Calculate the gradient
                grad[i] = (obj_plus - obj_minus) / (2 * h)
                self.sim.reset()
                # Recompute the objective at x to check for drift
                #objx_after = self.sim.run_obj(self.sim.unflatten_params(x))
                #print(f"Recomputed objective at x: {objx_after}")
                """
                # Check if objective at x changed unexpectedly
                if not np.isclose(objx, objx_after, rtol=1e-5, atol=1e-5):
                    raise ValueError(
                        f"Objective function instability detected:\n"
                        f"Initial objective: {objx}\n"
                        f"Recomputed objective: {objx_after}\n"
                        f"Parameters: {x}"
                    )
                """
            except Exception as e:
                print(f"Failed to compute gradient for index {i}: {e}")
                grad[i] = 0.0  # Set gradient to zero if the calculation fails
    
        #print("Final Computed Gradient:", grad)
        return grad
    
    
    
    
