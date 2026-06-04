from amr.interfaces import FunctionProfile
from amr.bounds import default_bounds, profile_to_bounds


def test_default_bounds_are_wide_and_free():
    b = default_bounds()
    lows, highs = b.as_arrays()

    assert lows[2] < highs[2]
    assert b.a.fixed is None


def test_even_function_does_not_fix_reflection():
    prof = FunctionProfile(name="cos", symmetry="even")
    b = profile_to_bounds(prof)

    assert b.a.fixed is None  # a should be free


def test_periodic_function_bounds_shift_to_period():
    import numpy as np
    prof = FunctionProfile(name="sin", symmetry="odd", periodic=True,
                           period=2 * np.pi)
    b = profile_to_bounds(prof)

    assert b.a.effective_bounds() == (1.0, 1.0)  # scale fixed to 1
    lo, hi = b.b.effective_bounds()
    assert lo == -2 * np.pi and hi == 2 * np.pi
