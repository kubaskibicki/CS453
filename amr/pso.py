import numpy as np
from .interfaces import SearchBounds, RunConfig, PSOResult

def optimize(fitness_fn, bounds, config):
    """The particle swarm optimization (PSO) algorithm"""

    rd = np.random.default_rng(config.seed)

    lower, upper = bounds.as_arrays()
    dim = len(lower)
    n = config.n_particles

    # Get the total width of search space for each dimension
    span = np.where(upper > lower, upper - lower, 1.0)

    # Spawn the particles, uniformly distributed over search space
    pos = rd.uniform(lower, upper, size=(n, dim))
    vel = rd.uniform(-span, span, size=(n, dim)) * 0.1


    # Initial assessment of the swarm
    fit = np.array([fitness_fn(p) for p in pos])

    personal_best = pos.copy()
    personal_best_fit = fit.copy()
    
    glb = int(np.argmin(personal_best_fit))
    glb_best = personal_best[glb].copy()
    gbest_fit = float(personal_best_fit[glb])

    history = [gbest_fit]
    no_improve = 0 # Patience counter
    converged = False
    iterations = 0

    # Main PSO Loop
    for it in range(config.max_iterations):
        iterations = it + 1

        r1 = rd.random((n, dim))
        r2 = rd.random((n, dim))

        # Update the velocity and position of every particle simultaneously
        vel = (config.inertia * vel
               + config.cognitive * r1 * (personal_best - pos)
               + config.social * r2 * (glb_best - pos))
        pos = np.clip(pos + vel, lower, upper)  # keep particles in bounds

        # Evaluation of the swarm
        fit = np.array([fitness_fn(p) for p in pos])
        improved = fit < personal_best_fit

        personal_best[improved] = pos[improved]
        personal_best_fit[improved] = fit[improved]

        # Update global best and check for convergence
        glb = int(np.argmin(personal_best_fit))

        if personal_best_fit[glb] < gbest_fit - config.convergence_eps:
            glb_best = personal_best[glb].copy()
            gbest_fit = float(personal_best_fit[glb])

            no_improve = 0 # Progress was made
        else:
            no_improve += 1 # No meaningful progress

        history.append(gbest_fit)

        if no_improve >= config.convergence_window:
            converged = True
            break

    return PSOResult(
        best_params=glb_best.tolist(),
        best_fitness=gbest_fit,
        iterations=iterations,
        converged=converged,
        history=history,
    )
