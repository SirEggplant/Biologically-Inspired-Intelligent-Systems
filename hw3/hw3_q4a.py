import numpy as np
import sys

"""
################################################################################
CSCI 633: Biologically-Inspired Intelligent Systems
Version taught by Alexander G. Ororbia II

@author Alexander G. Ororbia II, Evan Lin

Homework #3, Question 4a
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

def solve(f, swarm_size=30, w=0.7, c1=1.5, c2=1.5, max_iter=2000,
          bounds=(-2.0, 2.0)):
    """
    Particle Swarm Optimization (PSO) for minimization.

    Args:
        f: objective function
        swarm_size: number of particles
        w: inertia weight
        c1: cognitive coefficient
        c2: social coefficient
        max_iter: maximum number of iterations
        bounds: tuple (lower, upper) for all dimensions

    Returns:
        best_solution (numpy array): best found solution
    """
    lower, upper = bounds
    # Initialize particles
    positions = np.random.uniform(lower, upper, (swarm_size, D))
    velocities = np.random.uniform(-1, 1, (swarm_size, D))
    personal_best_pos = positions.copy()
    personal_best_val = np.array([f(p) for p in positions])
    global_best_idx = np.argmin(personal_best_val)
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_val = personal_best_val[global_best_idx]

    for _ in range(max_iter):
        for i in range(swarm_size):
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)
            # Update velocity
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best_pos[i] - positions[i]) +
                             c2 * r2 * (global_best_pos - positions[i]))
            # Update position
            positions[i] = positions[i] + velocities[i]
            # Clip to bounds
            positions[i] = np.clip(positions[i], lower, upper)

            # Evaluate
            val = f(positions[i])
            if val < personal_best_val[i]:
                personal_best_pos[i] = positions[i].copy()
                personal_best_val[i] = val
                if val < global_best_val:
                    global_best_val = val
                    global_best_pos = positions[i].copy()
    return global_best_pos

################################################################################
# Set up problem/algorithm hyper-parameters/settings
################################################################################

# problem settings/constraints
lower_bound = -2.0
upper_bound = 2.0
bounds = (lower_bound, upper_bound)

# Parameters to test
swarm_sizes = [30, 50, 100]
w_values = [0.4, 0.7, 0.9]
c1 = 1.5
c2 = 1.5
max_iter = 2000
num_trials = 30

# test/check for global optima against analytical solution
x_star = np.ones((1, D))
f_star = f(x_star.flatten()) # should be f(x) = <0,..,0>
print("True global minimum: f(x*) = {:.10f} at x* = (1,1)".format(f_star))
f_target = np.zeros((1,)) # global optimum is at <0,...,0> yielding f* = f(x) = 0
np.testing.assert_almost_equal(f_star, f_target, decimal=8)
print("Passed Rosenbrock check.\n")

################################################################################
# Begin simulation and run your multi-trial statists
# Make sure you compute your mean & standard deviation of your solution's f*
################################################################################

print("Running PSO on Rosenbrock...")
print("=" * 80)

for swarm_size in swarm_sizes:
    for w in w_values:
        trial_fitness = []
        for trial in range(num_trials):
            np.random.seed(seed + trial)
            best_sol = solve(f, swarm_size=swarm_size, w=w, c1=c1, c2=c2,
                             max_iter=max_iter, bounds=bounds)
            trial_fitness.append(f(best_sol))

        mean_f = np.mean(trial_fitness)
        std_f = np.std(trial_fitness, ddof=1)
        print(f"Swarm={swarm_size:3d}, w={w:4.2f} -> "
              f"mean={mean_f:.8f} ± {std_f:.8f}")

print("=" * 80)
print("Experiments complete.")
