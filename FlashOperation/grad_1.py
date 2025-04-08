# Gradient Descent Module
import pickle
import numpy as np
from AspenSim import AspenSim
from typing import Callable
import unittest
import time
from FlashOperation.Refrig2Drum2Comp import Refrig2Drum2Comp

class gradMoment():
    def __init__(self, sim: AspenSim, minmax = [np.array([0, 0]), np.array([100, 100])]):
        self.sim = sim
        self.minmax = minmax
        

    def optimize(self, x_init: np.ndarray, 
                 alpha=1e-4, beta=0.9, 
                 epsilon=1e-4, 
                 max_iter=1000, 
                 patience=10,
                 obj_norm = 1e6,
                 callback: Callable = None
                 ):
                 
        """
        Perform gradient descent optimization with momentum.

        Args:
            x_init (np.ndarray): Initial guess for the parameters.
            alpha (float, optional): Learning rate. Defaults to 1e-4.
            beta (float, optional): Momentum factor. Defaults to 0.9.
            epsilon (float, optional): Convergence threshold for the gradient norm. Defaults to 1e-4.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
            patience (int, optional): Number of iterations to wait for improvement before early stopping. Defaults to 10.
            callback (Callable, optional): Function to call after each iteration. Defaults to None.

        Returns:
            tuple: Best parameters found, best objective value, and the path of objective values during optimization.
        """
        
        min_val, max_val, best_x, best_obj, obj_path = self.descentLoop(x_init, alpha, beta, beta, epsilon, epsilon, max_iter, patience, obj_norm, callback)
    
        # Rescale best_x back to original values
        best_x_unscaled = best_x * (max_val - min_val) + min_val
        return self.sim.unflatten_params(best_x_unscaled), best_obj, obj_path

    def descentLoop(self, x_init, alpha, beta1, beta2, adam_epsilon, grad_epsilon, max_iter, patience, obj_norm, callback):
        min_val, max_val = self.minmax
        x_scaled = (x_init - min_val) / (max_val - min_val)  # Scale to [0,1]
        x = x_scaled.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        t = 0
        best_x = x.copy()
        best_obj = float('inf')
        patience_counter = 0
        obj_path = []
        
        while t < max_iter:
            t += 1
            grad = self.grad_approx(x, obj_norm=obj_norm)
            
            # Adam update rules
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            x = x - alpha * m_hat / (np.sqrt(v_hat) + adam_epsilon)
            
            # Evaluate objective on unscaled parameters
            obj = self.sim.run_obj(self.sim.unflatten_params(x * (max_val - min_val) + min_val))
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

            if np.linalg.norm(grad) < grad_epsilon:
                print(f"Convergence achieved at iteration {t}")
                break

            if callback:
                callback()

        return min_val, max_val, best_x, best_obj, obj_path


    def grad_approx(self, x, h=1e-2, obj_norm = 1e6):
        min_val, max_val = self.minmax
        h_scaled = np.array(h / (max_val - min_val), dtype=float)  # Ensure it's an array of floats
        # print("h", h)
        # print("h_scaled", h_scaled)
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
    
            try:
                
                # Convert scaled parameters back to real scale before simulation
                x_plus_unscaled = x_plus * (max_val - min_val) + min_val
                x_minus_unscaled = x_minus * (max_val - min_val) + min_val

                obj_plus = self.sim.run_obj(self.sim.unflatten_params(x_plus_unscaled))
                obj_minus = self.sim.run_obj(self.sim.unflatten_params(x_minus_unscaled))
                
                # print("x_plus_unscaled", x_plus_unscaled)
                # print("obj_plus", obj_plus)
                # print("x_minus_unscaled", x_minus_unscaled)
                # print("obj_minus", obj_minus)

                grad[i] = (obj_plus - obj_minus) / (2 * float(h_scaled[i]) * obj_norm)

                
                
            except Exception as e:
                print(f"Failed to compute gradient for index {i}: {e}")
                grad[i] = 0.0  
            print("grad", grad)
        return grad