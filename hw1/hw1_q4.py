import numpy as np

"""
################################################################################
CSCI 633: Biologically-Inspired Intelligent Systems

@author Evan_Lin

Homework #1, Question 4
################################################################################
"""
## Note: This code assumes row-oriented data/input vectors
##       in other words, an x would be of shape 1xD not Dx1 like in class

seed = 1990 # <-- note that you can change this seed to other values
np.random.seed(seed)

D = 2 # dimensionality of solution space for this problem/simulation


def f(x):
    """
    Negative Alpine function to be MAXIMIZED.
    
    Args:
        x: A 1xD array representing a candidate solution, where D=2 for this problem.
    
    Returns:
        float: The value of the negative Alpine function at x.
            f(x) = -sum( |x_i * sin(x_i) + 0.1 * x_i| ) for i=1 to D.
            The global maximum is 0 at x = [0, 0].
    """
    term = np.abs(x * np.sin(x) + 0.1 * x)
    result = -np.sum(term)
    return result

def solve(f, num_restarts, max_steps_per_restart=1000, step_size=0.5, bounds=(-10.0, 10.0)):
    """
    Stochastic Hill Climbing with Random Restarts.
    
    Args:
        f (callable): The objective function to maximize.
        num_restarts (int): Number of random restarts (N).
        max_steps_per_restart (int): Maximum iterations per local search.
        step_size (float): Standard deviation for Gaussian perturbation.
        bounds (tuple): (lower_bound, upper_bound) for the search space.
    
    Returns:
        tuple: (best_solution, best_fitness, avg_steps_per_restart)
               best_solution (numpy.ndarray): Best solution found across all restarts.
               best_fitness (float): Fitness of the best solution.
               avg_steps_per_restart (float): Average steps taken before convergence.
    """
    D = 2
    lower, upper = bounds
    best_solution_overall = None
    best_fitness_overall = -np.inf
    total_steps_all = 0
    
    for _ in range(num_restarts):
        current = np.random.uniform(lower, upper, D)
        current_fitness = f(current)
        
        steps_this_restart = 0
        for step in range(max_steps_per_restart):
            candidate = current + np.random.randn(D) * step_size
            
            candidate = np.clip(candidate, lower, upper)
            
            candidate_fitness = f(candidate)
            
            if candidate_fitness > current_fitness:
                current = candidate
                current_fitness = candidate_fitness
            
            steps_this_restart += 1
        
        total_steps_all += steps_this_restart
        
        if current_fitness > best_fitness_overall:
            best_fitness_overall = current_fitness
            best_solution_overall = current.copy()

    avg_steps = total_steps_all / num_restarts
    return best_solution_overall, best_fitness_overall, avg_steps

################################################################################
# Run the simulation/experiment code -- you may adapt/modify as you like
################################################################################

# problem settings/constraints (you can rename these if needed)
lower_bound = -10.0
upper_bound = 10.0
bounds = (lower_bound, upper_bound)

# WRITEME: OTHER SETUP/CONFIGURATIONS GO HERE, VALUES YOU WANT TO TEST...
# Parameters
max_steps_per_restart = 1000
step_size = 0.5
# Restart values
restart_values = [1, 2, 3, 5, 10, 30, 50, 100, 150, 200]
num_trials = 30 

################################################################################
# TODO: Your f(x) MUST check for global optima against analytical solution
# DO NOT MODIFY THIS BLOCK OF CODE
##################### START CODE BLOCK #####################
x_star = np.zeros((1,D)) # x* <-- the true "unknown" global optima
f_max = f(x_star.flatten())
f_star = np.zeros((1,1)) # global maximum is at <0,...,0> yielding f* = f(x) = 0
print("True Gobal Maximum:  f(x*) = ",f_max)
# your code should NOT trigger this if it is correct
np.testing.assert_array_equal(f_max, f_star)
print("Passed")

##################### END OF CODE BLOCK #####################
################################################################################


################################################################################
# Begin simulation and run your multi-trial statists
# Make sure you compute your mean & standard deviation of your solution's f*
################################################################################

print("Running experiments...")
print("-" * 80)
print(f"{'# Restarts':<12} {'f(x)max Found (mean ± std)':<40} {'# Iterations (mean)':<20}")
print("-" * 80)

results = []

for n_restarts in restart_values:
    trial_fitness = []
    trial_iterations = []
    
    for trial in range(num_trials):
        np.random.seed(seed + trial)
        
        best_sol, best_fit, avg_steps = solve(
            f, 
            num_restarts=n_restarts,
            max_steps_per_restart=max_steps_per_restart,
            step_size=step_size,
            bounds=bounds
        )
        
        trial_fitness.append(best_fit)
        trial_iterations.append(avg_steps)
    
    mean_fitness = np.mean(trial_fitness)
    std_fitness = np.std(trial_fitness, ddof=1)
    mean_iterations = np.mean(trial_iterations)
    
    results.append({
        'restarts': n_restarts,
        'mean_fit': mean_fitness,
        'std_fit': std_fitness,
        'mean_iter': mean_iterations
    })
    
    print(f"N = {n_restarts:<6} {mean_fitness:.8f} ± {std_fitness:.8f}        {mean_iterations:.2f}")

print("-" * 80)

# WRITEME: Experimental code goes here...
