import numpy as np
import sys

"""
################################################################################
CSCI 633: Biologically-Inspired Intelligent Systems
Version taught by Alexander G. Ororbia II

@author Alexander G. Ororbia II, Evan Lin

Homework #4, Question 2a
################################################################################
"""
# Note: This code assumes row-oriented data/input vectors

seed = 1990 #123 #1234 #69
np.random.seed(seed)

D = 2 # dimensionality of solution space for this problem/simulation

def f(x):
    """
    Four‑peak function to be MAXIMIZED.
    f(x,y) = e^{-(x-4)^2-(y-4)^2} + e^{-(x+4)^2-(y-4)^2}
             + 2*(e^{-x^2-y^2} + e^{-x^2-(y+4)^2})
    Global maxima: 2 at (0,0) and (0,-4); local maxima: 1 at (-4,4) and (4,4).
    Domain: [-5,5]^2.
    """
    x1, x2 = x[0], x[1]
    term1 = np.exp(-(x1-4)**2 - (x2-4)**2)
    term2 = np.exp(-(x1+4)**2 - (x2-4)**2)
    term3 = np.exp(-x1**2 - x2**2)
    term4 = np.exp(-x1**2 - (x2+4)**2)
    return term1 + term2 + 2*(term3 + term4)

def solve(f, pop_size=25, beta0=1.0, gamma=1.0, alpha=0.5,
         max_iter=200, lower=-5.0, upper=5.0):
    """
    Firefly Algorithm for maximisation.
    Args:
        f: objective function (maximise)
        pop_size: number of fireflies
        beta0, gamma, alpha: FA hyper‑parameters
        max_iter: maximum generations
        lower, upper: search bounds
    Returns:
        best_solution (numpy array)
    """
    # initialize population
    pop = np.random.uniform(lower, upper, (pop_size, D))
    brightness = np.array([f(ind) for ind in pop])
    best_idx = np.argmax(brightness)
    best = pop[best_idx].copy()
    best_val = brightness[best_idx]

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
                    if new_val > brightness[i]:
                        pop[i] = new_pos
                        brightness[i] = new_val
                        if new_val > best_val:
                            best = new_pos.copy()
                            best_val = new_val
    return best

################################################################################
# Set up problem/algorithm hyper-parameters/settings
################################################################################

# problem settings/constraints
lower_bound = -5.0
upper_bound = 5.0

# WRITEME: OTHER SETUP/CONFIGURATIONS GO HERE, VALUES YOU WANT TO TEST...
pop_sizes = [5, 10, 25, 50, 100]
param_sets = [
    (1.0, 1.0, 0.5, "A"),   # (beta0, gamma, alpha, label)
    (0.8, 0.5, 0.2, "B")
]

num_trials = 30
max_iter = 200

################################################################################
# check for critical modes against analytical solution
# since the 4-peak function has 4 key modes, your function must pass all 4 checks
# or it is invalid/incorrect
x_star_lm1 = np.asarray([-4.,4.]) # local mode 1
x_star_lm2 = np.asarray([4.,4.]) # local mode 2
x_star_gm1 = np.asarray([0.,0.]) # global mode 1
x_star_gm2 = np.asarray([0.,-4.]) # global mode 2
# check local modes
f_star_local = 1.0
f_star_global = 2.0
tol = 1e-6

f_val = f(x_star_lm1)
print(f"f* = {f_star_local}  f(x) = {f_val}")
assert abs(f_val - f_star_local) < tol, "Local mode 1 failed"
f_val = f(x_star_lm2)
print(f"f* = {f_star_local}  f(x) = {f_val}")
assert abs(f_val - f_star_local) < tol, "Local mode 2 failed"
f_val = f(x_star_gm1)
print(f"f* = {f_star_global}  f(x) = {f_val}")
assert abs(f_val - f_star_global) < tol, "Global mode 1 failed"
f_val = f(x_star_gm2)
print(f"f* = {f_star_global}  f(x) = {f_val}")
assert abs(f_val - f_star_global) < tol, "Global mode 2 failed"
print("All function checks passed.\n")
################################################################################

################################################################################
# Begin simulation and run your multi-trial statists
# Make sure you compute your mean & standard deviation of your solution's f*
################################################################################

print("FA on four‑peak function (maximisation)")
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
        # Print without scientific notation (non‑rounded)
        print(f"n={pop_size:3d}, {label}: {mean_val:.8f} ± {std_val:.8f}")
print("=" * 80)