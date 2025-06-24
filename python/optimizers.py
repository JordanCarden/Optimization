# In python/optimizers.py

import cma
import numpy as np

def run_cma_es(objective_tracker, bounds, random_seed):
    """
    Optimize using CMA-ES by operating on an ObjectiveTracker instance.
    """
    # Initial guess is the center of the bounds
    x0 = np.array([(lb + ub) / 2 for lb, ub in bounds])
    
    # Initial step-size
    sigma0 = 0.25 * np.mean([ub - lb for lb, ub in bounds])
    
    # Recommended population size
    popsize = 4 + int(3 * np.log(len(bounds)))
    
    options = {
        'bounds': [[b[0] for b in bounds], [b[1] for b in bounds]],
        'maxfevals': 5000,
        'popsize': popsize,
        'seed': random_seed,
    }
    
    solution, es = cma.fmin2(objective_tracker.evaluate, x0, sigma0, options=options)
    
    return solution, es.best.f