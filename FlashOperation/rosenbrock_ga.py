# -*- coding: utf-8 -*-
"""
Created on Mon Mar 3 13:36:14 2025

"""
from Rosenbrock import Rosenbrock
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
import pickle  # Import pickle for saving data
import time

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

class AspenProblem(ElementwiseProblem):
    def __init__(self, assSim):
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=0, xl=[-15, -15], xu=[15, 15])  # No constraints for Rosenbrock
        self.assSim = assSim

    def _evaluate(self, x, out, *args, **kwargs):  
        x_eval = {
            "Flash2": {'FLASH1': [x[0]], 'FLASH2': [x[1]]}
        }
        
        results = self.assSim.runSim(x_eval)
        
        # Objective function
        out["F"] = np.array([self.assSim.costFunc(results)])
        

def main():
    print("here")
    print(os.getcwd())
    assSim = Rosenbrock()  # Use the Rosenbrock class
    
    problem = AspenProblem(assSim)
    algorithm = GA(pop_size=50, eliminate_duplicates=True)
    
    start_time = time.time()  # Record the start time
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 10),
                   verbose=True,
                   save_history=True)
    
    end_time = time.time()
    total_exec_time = res.exec_time  # Get the total execution time
    
    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)
    print("Execution time: %s seconds" % total_exec_time)
    
    # Print total execution time
    print("Total execution time: %.2f seconds" % (end_time - start_time))
    
    # Extract average and minimum objective values for each generation
    avg_obj_values = []
    min_obj_values = []
    population_data = []  # Store population data for each generation
    for gen_idx, gen in enumerate(res.history):
        F = gen.pop.get("F")
        X = gen.pop.get("X")  # Get design variables for the generation

        avg_obj_values.append(np.mean(F))
        min_obj_values.append(np.min(F))
        population_data.append({"Generation": gen_idx + 1, "Population": X.tolist()})

    # Calculate Aspen calls for each generation
    aspen_calls = []
    for i in range(len(res.history)):
        aspen_calls.append(algorithm.pop_size * (i + 1))  # Assuming each generation has pop_size evaluations

    # Save avg_obj_values, min_obj_values, and aspen_calls vs. generations to a single pickle file
    obj_values_pickle_filename = "rosenbrock_obj_values_vs_generations.pkl"
    with open(obj_values_pickle_filename, "wb") as f:
        pickle.dump({
            "Generations": list(range(1, len(avg_obj_values) + 1)),
            "Avg_Obj_Values": avg_obj_values,
            "Min_Obj_Values": min_obj_values,
            "Aspen_Calls": aspen_calls
        }, f)
    print(f"Saved objective values (average, minimum) and Aspen calls vs. generations to '{obj_values_pickle_filename}'.")

    # Save population distribution over iterations to a pickle file
    population_pickle_filename = "rosenbrock_population_distribution.pkl"
    with open(population_pickle_filename, "wb") as f:
        pickle.dump(population_data, f)
    print(f"Saved population distribution over iterations to '{population_pickle_filename}'.")

    # Plot number of Aspen calls vs objective function value
    plt.figure(figsize=(10, 6))
    plt.plot(aspen_calls, avg_obj_values, marker='o', label='Average Objective Value', color='blue')
    plt.plot(aspen_calls, min_obj_values, marker='s', label='Minimum Objective Value', color='green')
    plt.xlabel('Number of Aspen Calls')
    plt.ylabel('Objective Value')
    plt.title('Objective Value vs. Number of Aspen Calls')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot average and minimum objective values vs. generation
    plt.figure(figsize=(10, 6))
    generations = range(1, len(avg_obj_values) + 1)
    plt.plot(generations, avg_obj_values, marker='o', label='Average Objective Value', color='blue')
    plt.plot(generations, min_obj_values, marker='s', label='Minimum Objective Value', color='green')
    plt.xlabel('Generation')
    plt.ylabel('Objective Value')
    plt.title('Objective Value vs. Generation')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the population distribution over iterations
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(res.history)))  # Generate a color for each generation

    for gen_idx, gen in enumerate(res.history):
        X = gen.pop.get("X")  # Get design variables for the generation
        plt.scatter(X[:, 0], X[:, 1], color=colors[gen_idx], label=f"Iter {gen_idx}")

    # Plot the optimal point as a red star
    plt.scatter(res.X[0], res.X[1], color='red', marker='*', s=200, label='Optimal Point')

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Population Distribution Over Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()
