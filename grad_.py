# Gradient Descent Module
import pickle
import numpy as np
from AspenSim import AspenSim
from typing import Callable
import unittest
from FlashOperation.Refrig2Drum2Comp import Refrig2Drum2Comp

class gradMoment():
    def __init__(self, sim: AspenSim):
        self.sim = sim

    def optimize(self, x_init: np.ndarray, alpha=0.01, beta=0.9, epsilon=1e-4, max_iter=1000, patience=10):
        x = x_init.copy()
        v = np.zeros_like(x)
        t = 0
        best_x = x.copy()
        best_obj = float('inf')
        patience_counter = 0
    
        while t < max_iter:
            t += 1
            grad = self.grad_approx(x)
            v = beta * v + (1 - beta) * grad
            x = x - alpha * v
            print("new x", x)
            obj = self.sim.run_obj(self.sim.unflatten_params(x))

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
    
        return self.sim.unflatten_params(best_x), best_obj


    def grad_approx(self, x, h=1e-5):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            print("x_plus", x_plus)
            print("x_minus", x_minus)
            try:
                obj_plus = self.sim.run_obj(self.sim.unflatten_params(x_plus))
                print('obj_plus:', obj_plus)
                obj_minus = self.sim.run_obj(self.sim.unflatten_params(x_minus))
                print('pbj_minus:', obj_minus)
                grad[i] = (obj_plus - obj_minus) / (2 * h)
            except Exception as e:
                print(f"Failed to compute gradient for index {i}: {e}")
                break
    
            # Small delay to ensure Aspen stability
            #import time
            #time.sleep(1)  # 1-second delay between runs
        print("grad:", grad)
        return grad


