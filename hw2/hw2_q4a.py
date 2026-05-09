import numpy as np
import sys

"""
################################################################################
CSCI 633: Biologically-Inspired Intelligent Systems
Version taught by Alexander G. Ororbia II

@author Alexander G. Ororbia II, Evan Lin

Homework #2, Question 4a
################################################################################
"""
# Note: This code assumes row-oriented data/input vectors

seed = 1990 #123 #1234 #69
np.random.seed(seed)

D = 2 # dimensionality of solution space for this problem/simulation

def f(x):  # returns a scalar
    """
    Rosenbrock's function to be minimized.
    f(x) = (x1 - 1)^2 + 100 * (x2 - x1^2)^2
    
    Args:
        x: A 1xD array representing a candidate solution, where D=2 for this problem.
    
    Returns:
        float: The value of Rosenbrock's function at x.
    """
    x1, x2 = x[0], x[1]
    t1 = (x1 - 1)**2
    t2 = 100 * (x2 - x1**2)**2
    result = t1 + t2
    return result

def solve(f, initial_temp, cooling_rate, step_size, max_iters, lower_bound, upper_bound):
    """
    Simulated Annealing algorithm for minimization.
    
    Args:
        f: Objective function to minimize
        initial_temp: Starting temperature
        cooling_rate: Cooling factor
        step_size: Alpha for random walk 
        max_iters: Maximum number of iterations
        lower_bound: Lower bound of search space
        upper_bound: Upper bound of search space
    
    Returns:
        numpy.ndarray: Best solution found (x_hat)
    """
    D = 2
    # Initialize random starting point
    current = np.random.uniform(lower_bound, upper_bound, D)
    current_f = f(current)
    
    # Track best solution
    best = current.copy()
    best_f = current_f
    
    # Current temperature
    T = initial_temp
    
    for _ in range(max_iters):
        # Generate candidate
        candidate = current + step_size * np.random.randn(D)
        candidate = np.clip(candidate, lower_bound, upper_bound)
        # Evaluate candidate
        candidate_f = f(candidate)
        # Calculate change
        delta_f = candidate_f - current_f

        # Accept or not
        if delta_f < 0:  # Better solution = accept
            current = candidate
            current_f = candidate_f
        else:  # Worse solution = accept with probability
            p = np.exp(-delta_f / T)
            if np.random.rand() < p:
                current = candidate
                current_f = candidate_f
        
        # Update solution
        if current_f < best_f:
            best = current.copy()
            best_f = current_f

        # Cool temperature
        T = cooling_rate * T
    
    return best

################################################################################
# Set up problem/algorithm hyper-parameters/settings
################################################################################

# problem settings/constraints
lower_bound = -2.0
upper_bound = 2.0

# Simulated Annealing Hyperparameters
initial_temps = [100, 50, 10, 5, 1]
step_sizes = [0.1, 0.5, 1.0]
cooling_rate = 0.95
max_iters = 5000
num_trials = 30

# test/check for global optima against analytical solution
x_star = np.ones((1,D))
f_max = f(x_star.flatten()) # should be f(x) = <0,..,0>
print("True Gobal Minimum:  f(x*) = {}  at x* = {}".format(f_max, x_star))
f_star = np.zeros((1,1)) # global maximum is at <0,...,0> yielding f* = f(x) = 0
np.testing.assert_array_equal(f_max, f_star.flatten())
print("Passed!\n")

################################################################################
# Begin simulation and run your multi-trial statists
# Make sure you compute your mean & standard deviation of your solution's f*
################################################################################

print("Running Rosenbrock's function...")
print("-----------------------------------")

# Store results
for step_size in step_sizes:
    print(f"\nStep Size α = {step_size}")
    print("-----------------------------------")
    print(f"{'Initial Temp':<12} {'f(x) Found (mean ± std)':<40}")
    print("-----------------------------------")
    
    for T0 in initial_temps:
        trial_fitness = []
        
        for trial in range(num_trials):
            # Use different seed for each trial
            np.random.seed(seed + trial)
            
            best_sol = solve(f, 
                initial_temp=T0,
                cooling_rate=cooling_rate,
                step_size=step_size,
                max_iters=max_iters,
                lower_bound=lower_bound,
                upper_bound=upper_bound
            )
            
            trial_fitness.append(f(best_sol))
        
        # Calculate statistics
        mean_f = np.mean(trial_fitness)
        std_f = np.std(trial_fitness, ddof=1)  # standard deviation
        
        print(f"T0 = {T0:<6} {mean_f:.8f} ± {std_f:.8f}")
    
    print("-----------------------------------")
