import numpy as np
import sys

"""
################################################################################
CSCI 633: Biologically-Inspired Intelligent Systems
Version taught by Alexander G. Ororbia II

@author Alexander G. Ororbia II, Evan Lin

Homework #3, Question 3b
################################################################################
"""
# Note: This code assumes row-oriented data/input vectors

seed = 1990 #123 #1234 #69
np.random.seed(seed)

D = 2 # dimensionality of solution space for this problem/simulation

def f(x):
    """
    Easom function to be MINIMIZED.
    f(x) = -cos(x1)*cos(x2)*exp(-(x1-pi)^2 - (x2-pi)^2)
    Global minimum at (pi, pi) with value -1.
    Domain: -100 ≤ xi ≤ 100
    """
    x1, x2 = x[0], x[1]
    term = -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi)**2 - (x2 - np.pi)**2)
    return term

def solve(f, pop_size=50, F=0.8, Cr=0.9, max_gen=2000, bounds=(-100.0, 100.0)):
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
lower_bound = -100.0
upper_bound = 100.0
bounds = (lower_bound, upper_bound)

# Parameters to test
pop_sizes = [30, 50, 100]
F_values = [0.5, 0.8, 1.0]
Cr_values = [0.3, 0.5, 0.9]
max_gen = 2000
num_trials = 30

# test/check for global optima against analytical solution
x_star = np.ones((1,D)) * np.pi
f_star = f(x_star.flatten()) # should be f(x) = <0,..,0>
print("True global minimum: f(x*) = {:.10f} at x* = ({}, {})".format(f_star, np.pi, np.pi))
f_target = -1.0
tol = 1e-8
delta = np.abs(f_star - f_target) # how far is your f* from the true f*
np.testing.assert_array_less(delta, tol)
print("Passed Easom check.\n")

################################################################################
# Begin simulation and run your multi-trial statists
# Make sure you compute your mean & standard deviation of your solution's f*
################################################################################

print("Running DE on Easom...")
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
