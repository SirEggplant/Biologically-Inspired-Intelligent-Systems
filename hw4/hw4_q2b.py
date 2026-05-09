import numpy as np
import sys

"""
################################################################################
CSCI 633: Biologically-Inspired Intelligent Systems
Version taught by Alexander G. Ororbia II

@author Alexander G. Ororbia II, Evan Lin

Homework #4, Question 2b
################################################################################
"""
# Note: This code assumes row-oriented data/input vectors

seed = 1990 #123 #1234 #69
np.random.seed(seed)

D = 2 # dimensionality of solution space for this problem/simulation
TWOPI = 2 * np.pi

def f(x):
    """
    Eggcrate function to be MINIMIZED.
    f(x,y) = x^2 + y^2 + 25*(sin^2 x + sin^2 y)
    Global minimum: 0 at (0,0).
    Domain: [-2π, 2π]^2.
    """
    x1, x2 = x[0], x[1]
    return x1*x1 + x2*x2 + 25.0*(np.sin(x1)**2 + np.sin(x2)**2)

def solve(f, pop_size=25, beta0=1.0, gamma=1.0, alpha=0.5,
         max_iter=200, lower=-TWOPI, upper=TWOPI):
    """
    Firefly Algorithm for minimisation.
    We convert to maximisation by using brightness = -f(x).
    Args:
        f: objective function (minimise)
        pop_size, beta0, gamma, alpha, max_iter, lower, upper
    Returns:
        best_solution
    """
    # initialize
    pop = np.random.uniform(lower, upper, (pop_size, D))
    brightness = np.array([-f(ind) for ind in pop])
    best_idx = np.argmax(brightness)
    best = pop[best_idx].copy()
    best_val = f(best)

    for _ in range(max_iter):
        for i in range(pop_size):
            for j in range(pop_size):
                if brightness[j] > brightness[i]:
                    r = np.linalg.norm(pop[i] - pop[j])
                    beta = beta0 * np.exp(-gamma * r**2)
                    rand_step = (np.random.rand(D) - 0.5) * 2 * alpha
                    new_pos = pop[i] + beta * (pop[j] - pop[i]) + rand_step
                    new_pos = np.clip(new_pos, lower, upper)
                    new_val = f(new_pos)
                    new_bright = -new_val
                    if new_bright > brightness[i]:
                        pop[i] = new_pos
                        brightness[i] = new_bright
                        if new_val < best_val:
                            best = new_pos.copy()
                            best_val = new_val
    return best

################################################################################
# Set up problem/algorithm hyper-parameters/settings
################################################################################

# problem settings/constraints
lower_bound = -2.0 * np.pi
upper_bound = 2.0 * np.pi

# WRITEME: OTHER SETUP/CONFIGURATIONS GO HERE, VALUES YOU WANT TO TEST...
pop_sizes = [5, 10, 25, 50, 100]
param_sets = [
    (1.0, 1.0, 0.5, "A"),
    (0.8, 0.5, 0.2, "B")
]
num_trials = 30
max_iter = 200

# test/check for global optima against analytical solution
x_star = np.array([0.0, 0.0])
f_star = f(x_star)
print(f"True global minimum: f(0,0) = {f_star:.10f}")
assert abs(f_star - 0.0) < 1e-6, "Eggcrate function check failed"
print("Eggcrate function check passed.\n")

################################################################################
# Begin simulation and run your multi-trial statists
# Make sure you compute your mean & standard deviation of your solution's f*
################################################################################

print("FA on eggcrate function (minimisation)")
print("=" * 80)

for pop_size in pop_sizes:
    for beta0, gamma, alpha, label in param_sets:
        trial_fitness = []
        for trial in range(num_trials):
            np.random.seed(seed + trial + pop_size * len(param_sets))
            best_sol = solve(f, pop_size=pop_size, beta0=beta0, gamma=gamma,
                             alpha=alpha, max_iter=max_iter,
                             lower=lower_bound, upper=upper_bound)
            trial_fitness.append(f(best_sol))

        mean_val = np.mean(trial_fitness)
        std_val = np.std(trial_fitness, ddof=1)
        print(f"n={pop_size:3d}, {label}: {mean_val:.8f} ± {std_val:.8f}")
print("=" * 80)