# Gradient Descent Module
import pickle
import numpy as np
from AspenSim import AspenSim

class gradMoment():
    def __init__(self, 
        obj_f,
        aspensim: AspenSim,
        check_file = None
    ):
        self.obj_f = obj_f
        self.aspensim = aspensim
        self.check_file = check_file
    
    def load_chkpt(filename):
        pass
    
    def save_chkpt(filename):
        # save to pickle file
        pass

    def optimize(self, x_init: np.ndarray, 
                 alpha = 0.01, 
                 beta = 0.9, 
                 epsilon = 1e-4, 
                 max_iter = 1000,
                 patience = 10):
        
        # initialize the parameters
        x = x_init # initial guess
        v = np.zeros_like(x) # velocity
        t = 0 # iteration counter
        best_x = x
        best_obj = float('inf')
        patience_counter = 0
        
        while t < max_iter:
            t += 1
            grad = self.grad_approx(x) # compute gradient
            v = beta * v + (1 - beta) * grad # update velocity
            x = x - alpha * v # update parameters
            
            obj = self.obj_f(x) # evaluate objective function
            
            if obj < best_obj:
                best_obj = obj
                best_x = x
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at iteration {t}")
                break
            
            if np.linalg.norm(grad) < epsilon:
                print(f"Convergence achieved at iteration {t}")
                break
        
        return best_x, best_obj
        
        

    def grad_approx():
        pass
