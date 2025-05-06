from VCDistillationToy import VCDistillationToy
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.util.ref_dirs import get_reference_directions
import matplotlib.pyplot as plt
import pickle

class VCDistillationToyProblem(ElementwiseProblem):
    def __init__(self, assSim):
        super().__init__(n_var=4, n_obj=2, n_ieq_constr=2, xl=[-5, -5, -5, -5], xu=[5, 5, 5, 5])
        self.assSim = assSim

    def _evaluate(self, x, out, *args, **kwargs):
        x_eval = self.assSim.unflatten_params(x)
        results = self.assSim.runSim(x_eval)
        f1, f2 = self.assSim.costFunc(results)
        c1, c2 = self.assSim.constraintFunc(results)
        out["F"] = np.array([f1, f2])
        out["G"] = np.array([c1, c2])  # Add constraints to the output

def main():
    print("Starting NSGA-III optimization with VCDistillationToy.")
    assSim = VCDistillationToy()

    problem = VCDistillationToyProblem(assSim)
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=35)  # Use pymoo's ref_dirs
    print(len(ref_dirs))
    pop_size = len(ref_dirs)
    n_gen = 50

    algorithm = NSGA3(
        pop_size=pop_size,               # match ref_dirs for full coverage
        ref_dirs=ref_dirs,
        sampling=LHS(),                       # better initial diversity
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.3, eta=5),         # stronger mutation = more exploration
        eliminate_duplicates=True
        )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   verbose=True,
                   save_history=True)

    # Save results to a pickle file
    with open("vcdistillation_toy_results.pkl", "wb") as f:
        pickle.dump(res, f)
    print("Results saved to 'vcdistillation_toy_results.pkl'.")

    # Plot the Pareto front
    F = res.F
    plt.figure(figsize=(10, 6))
    plt.scatter(F[:, 0], F[:, 1], c="blue", label="Pareto Front")
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Pareto Front for VCDistillationToy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
