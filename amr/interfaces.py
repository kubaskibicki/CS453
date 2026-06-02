from dataclasses import dataclass
import numpy as np

# shared data structures - all modules import from here


@dataclass
class FunctionProfile:
    """A class that informs us about the LLM's analysis of function's
    math properties.
    
    Attributes:
        name (str): Function name
        symmetry (str): Property of math function;
        can be either even, odd or none
        periodic (bool): If it is set, then the function is periodic
        monotonic (str): Property defining if and what kind of monotonicity
        the function has
        domain (tuple): Defines what is the range of the function domain
        output_range (tuple): Range of the function output
    """
    name: str
    symmetry: str = "none"
    periodic: bool = False
    period: float = None
    monotonic: str = "none"
    domain: tuple = (-100.0, 100.0)
    output_range: tuple = (-100.0, 100.0)


@dataclass
class ParamBound:
    """Class for boundaries of parameters.
    
    Attributes:
        lower (float): lower boundary of a parameter
        upper (float): upper boundary of a parameter
        fixed (float): fixed value of a parameter, if set, parameter
        value will not be changed by PSO
    """

    lower: float
    upper: float
    fixed: float = None

    def effective_bounds(self):
        if self.fixed is not None:
            return (self.fixed, self.fixed)  # collapse to a single point
        return (self.lower, self.upper)


@dataclass
class SearchBounds:
    """Class defining the search boundaries
    
    Attributes:
        c1 (ParamBound): boundaries for param c1
        c2 (ParamBound): boundaries for param c2
        a (ParamBound): boundaries for param a
        b (ParamBound): boundaries for param b
        c (ParamBound): boundaries for param d
    """
    c1: ParamBound
    c2: ParamBound
    a: ParamBound
    b: ParamBound
    d: ParamBound

    def as_arrays(self):
        # The high & low values need to be numpy arrays (for PSO)
        lower_bounds, upper_bounds = [], []
        for pb in (self.c1, self.c2, self.a, self.b, self.d):
            lo, hi = pb.effective_bounds()
            lower_bounds.append(lo)
            upper_bounds.append(hi)
        return np.array(lower_bounds, dtype=float), np.array(upper_bounds, dtype=float)


@dataclass
class MRCandidate:
    """Class for a possible Metamorphic Relation
    
    Attributes:
        coefficients (dict): coefficients for the function parameters
        residual (float): residual error of the fitted relation
        valid (bool): Whether the candidate suits the criteria
    """
    coefficients: dict   # {"c1":..,"c2":..,"a":..,"b":..,"d":..}
    residual: float
    valid: bool = False


@dataclass
class PSOResult:
    """Class for noting down Particle Search Optimization results
    
    Attributes:
        best_params (list): list of the best parameters
        best_fitness (float): best fitness value
        iterations (int): num of iterations
        converged (bool): whether the results converged at a point
        history (list): history of subsequent iteration results
    """
    best_params: list
    best_fitness: float
    iterations: int
    converged: bool
    history: list


@dataclass
class RunConfig:
    """Class defining the configuration for running the program
    
    Attributes:
        model (str): used AI model for the application
        n_inputs (int): number of inputs
        tolerance (float): how much error will be tolerated in the results
        n_particles (int): number of particles in PSO
        max_iterations (int): number of maximum iterations of the algorithm
        inertia (float): controls how much particles
            preserve their current velocity
        cognitive(float): coefficient controlling attraction
            toward a particle's personal best position
        social (float): coefficient controlling attraction toward
            the swarm's global best position
        seed (int): seed used for generation of particles so results can be
        reproduced
        use_llm (bool): Whether LLM is used during execution
        out_dir (str): Output directory
        convergence_window (int): Number of recent iterations considered
            when checking convergence
        convergence_eps (float): Threshold for detecting convergence based
            on changes in the optimization objective
    """
    model: str = "qwen2.5:7b-instruct"
    n_inputs: int = 200
    tolerance: float = 1e-3
    n_particles: int = 50
    max_iterations: int = 350
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5
    seed: int = 42
    use_llm: bool = True
    out_dir: str = "results"
    convergence_window: int = 30
    convergence_eps: float = 1e-8
