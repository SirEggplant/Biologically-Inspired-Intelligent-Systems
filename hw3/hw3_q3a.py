import numpy as np
import sys

"""
################################################################################
CSCI 633: Biologically-Inspired Intelligent Systems
Version taught by Alexander G. Ororbia II

@author Alexander G. Ororbia II, Evan Lin

Homework #3, Question 3a
################################################################################
"""
# Note: This code assumes row-oriented data/input vectors

seed = 1990 #123 #1234 #69
np.random.seed(seed)

D = 2 # dimensionality of solution space for this problem/simulation

def f(x):  
    """
    Rosenbrock function to be MINIMIZED.
    f(x) = (x1 - 1)^2 + 100*(x2 - x1^2)^2
    Global minimum at (1,1) with value 0.
    Domain: -2 ≤ xi ≤ 2
    """
    x1, x2 = x[0], x[1]
    return (x1 - 1)**2 + 100 * (x2 - x1**2)**2

def solve(f, pop_size=50, F=0.8, Cr=0.9, max_gen=2000, bounds=(-2.0, 2.0)):
    """
    Differential Evolution (DE/rand/1/bin) for minimization.

    Args:
        f: objective function
        pop_size: number of individuals
        F: scaling factor (mutation)
        Cr: crossover probability
        max_gen: maximum number of generations
        bounds: tuple (lower, upper) for all dimensions

    Returns:
        best_solution (numpy array): best found solution
    """
    lower, upper = bounds
    # initialize population
    pop = np.random.uniform(lower, upper, (pop_size, D))
    fitness = np.array([f(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_val = fitness[best_idx]

    for _ in range(max_gen):
        for i in range(pop_size):
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

            # mutation
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, lower, upper)

            # binomial crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(D)
            for j in range(D):
                if np.random.rand() < Cr or j == j_rand:
                    trial[j] = mutant[j]

            # selection
            trial_f = f(trial)
            if trial_f <= fitness[i]:
                pop[i] = trial
                fitness[i] = trial_f
                if trial_f < best_val:
                    best = trial.copy()
                    best_val = trial_f
    return best

################################################################################
# Set up problem/algorithm hyper-parameters/settings
################################################################################

# problem settings/constraints
lower_bound = -2.0
upper_bound = 2.0
bounds = (lower_bound, upper_bound)

# Parameters to test
pop_sizes = [30, 50, 100]
F_values = [0.5, 0.8, 1.0]
Cr_values = [0.3, 0.5, 0.9]
max_gen = 2000
num_trials = 30

# test/check for global optima against analytical solution
x_star = np.ones((1,D))
f_star = f(x_star.flatten()) # should be f(x) = <0,..,0>
print("True global minimum: f(x*) = {:.10f} at x* = (1,1)".format(f_star))
f_target = np.zeros((1,)) # global optimum is at <0,...,0> yielding f* = f(x) = 0
np.testing.assert_almost_equal(f_star, f_target, decimal=8)
print("Passed Rosenbrock check.\n")

################################################################################
# Begin simulation and run your multi-trial statists
# Make sure you compute your mean & standard deviation of your solution's f*
################################################################################

print("Running DE on Rosenbrock...")
print("=" * 80)

for pop_size in pop_sizes:
    for F in F_values:
        for Cr in Cr_values:
            trial_fitness = []
            for trial in range(num_trials):
                np.random.seed(seed + trial)
                best_sol = solve(f, pop_size=pop_size, F=F, Cr=Cr,
                                 max_gen=max_gen, bounds=bounds)
                trial_fitness.append(f(best_sol))

            mean_f = np.mean(trial_fitness)
            std_f = np.std(trial_fitness, ddof=1)
            print(f"NP={pop_size:3d}, F={F:4.2f}, Cr={Cr:4.2f} -> "
                  f"mean={mean_f:.8f} ± {std_f:.8f}")

print("=" * 80)
print("Experiments complete.")