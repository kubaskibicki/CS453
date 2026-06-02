from .interfaces import FunctionProfile, SearchBounds, ParamBound

def default_bounds():
    """
    Default bounds for the parameters of the template function
    Recall: c1 * P(x1) = c2 * P(a * x1 + b) + d = 0
    """

    c1 = ParamBound(-2.0, 2.0)
    
    c2 = ParamBound(-2.0, 2.0)
    a = ParamBound(-2.0, 2.0)
    b = ParamBound(-10.0, 10.0)

    d = ParamBound(-2.0, 2.0)

    return SearchBounds(c1=c1, c2=c2, a=a, b=b, d=d)

def profile_to_bounds(profile):
    """
    Depending on the profile of the function, resulting from the analysis,
    we tighten the bounds to optimize the pso.
    """

    b = default_bounds()

    # odd functions: P(x) + P(-x) = 0 is a real non-trivial relation
    # even functions: P(x) - P(-x) = 0 is trivially true for ALL mutants, useless
    # so we only fix a=-1 for odd, not even
    if profile.symmetry == "odd":
        b.a = ParamBound(-1.0, -1.0, fixed=-1.0)
        b.b = ParamBound(0.0, 0.0, fixed=0.0)

    # If a function is periodic we lock the argument multiplier 'a'
    # and tighten the bound on 'b', since
    # periodic functions have shift relations within one period
    if profile.periodic and profile.period:
        p = float(profile.period)
        b.a = ParamBound(1.0, 1.0, fixed=1.0)
        b.b = ParamBound(-p, p)

    return b
