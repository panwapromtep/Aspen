# Gradient Descent Module
import pickle
import numpy as np
from AspenSim import AspenSim
from typing import Callable
import unittest
from scipy.optimize import minimize
from scipy.optimize import approx_fprime

class grad_des():
    def __init__(self, obj_f, aspensim: AspenSim, check_file=None):
        self.obj_f = obj_f
        self.aspensim = aspensim
        self.check_file = check_file

    def grad_approx(self, x, epsilon=None):
        if epsilon is None:
            epsilon = np.sqrt(np.finfo(float).eps)
        return approx_fprime(x, self.obj_f, epsilon)

    def optimize(self, x_init, method='CG', options=None):
        result = minimize(self.obj_f, 
                          x_init, 
                          method=method, 
                          jac=self.grad_approx, 
                          options=options
                          )
        return result.x, result.fun
    
    # def grad_approx(self, x, h=1e-5):
    #     grad = np.zeros_like(x)
    #     for i in range(len(x)):
    #         x_plus = x.copy()
    #         x_minus = x.copy()
    #         x_plus[i] += h
    #         x_minus[i] -= h
    #         grad[i] = (self.obj_f(x_plus) - self.obj_f(x_minus)) / (2 * h)
    #     return grad
    
class TestGradMoment(unittest.TestCase):
    def setUp(self):
        self.optimizer = gradMoment(None, aspensim=None)

    def test_quadratic_function(self):
        def obj_f(x): return np.sum(x ** 2)
        self.optimizer.obj_f = obj_f
        x = np.array([1.0, -2.0, 3.0])
        grad = self.optimizer.grad_approx(x)
        print(f"Approx gradient for quadratic at {x}: {grad}")
        print(f"True gradient for quadratic at {x}: {2 * x}")
        np.testing.assert_almost_equal(grad, 2 * x, decimal=5)
        print()

    def test_optimization_quadratic(self):
        def obj_f(x): return np.sum(x ** 2)
        self.optimizer.obj_f = obj_f
        x_init = np.array([1.0, -2.0, 3.0])
        best_x, best_obj = self.optimizer.optimize(x_init)
        print(f"Optimized values for quadratic: x = {best_x}, obj = {best_obj}")
        print(f"True optimal values for quadratic: x = {np.zeros_like(x_init)}, obj = 0.0")
        np.testing.assert_almost_equal(best_x, np.zeros_like(x_init), decimal=2)
        self.assertAlmostEqual(best_obj, 0.0, places=2)
        print()

    def test_3d_rosembrock(self):
        def obj_f(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0]**2) ** 2 + (1 - x[2]) ** 2
        def grad(x):
            return np.array([
                -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
                200 * (x[1] - x[0]**2),
                -2 * (1 - x[2])
            ])
        self.optimizer.obj_f = obj_f
        x = np.array([1.0, 2.0, -1.0])
        grad_val = self.optimizer.grad_approx(x)
        print(f"Approx gradient for Rosenbrock at {x}: {grad_val}")
        print(f"True gradient for Rosenbrock at {x}: {grad(x)}")
        np.testing.assert_almost_equal(grad_val, grad(x), decimal=3)
        print()

    def test_sine_cosine_function(self):
        def obj_f(x): return np.sin(x[0]) + np.cos(x[1]) + x[2]**3
        def grad(x): return np.array([np.cos(x[0]), -np.sin(x[1]), 3 * x[2]**2])
        self.optimizer.obj_f = obj_f
        x = np.array([np.pi/4, np.pi/6, 1.0])
        grad_val = self.optimizer.grad_approx(x)
        print(f"Approx gradient for sine-cosine at {x}: {grad_val}")
        print(f"True gradient for sine-cosine at {x}: {grad(x)}")
        np.testing.assert_almost_equal(grad_val, grad(x), decimal=3)
        print()

if __name__ == "__main__":
    unittest.main()