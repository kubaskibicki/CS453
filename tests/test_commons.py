import numpy as np

from amr import commons_math as cm
from amr import commons_mutants as cmut
from amr.functions import FUNCTION_NAMES, get_domain

_TRUTH = {"abs": np.abs, "asinh": np.arcsinh, "atan": np.arctan, "cos": np.cos,
          "log1p": np.log1p, "log10": np.log10, "sin": np.sin, "tan": np.tan}


def test_port_matches_numpy():
    rng = np.random.default_rng(0)
    for name in FUNCTION_NAMES:
        lo, hi = get_domain(name)
        x = rng.uniform(lo, hi, 500)
        err = np.max(np.abs(cm.eval_array(name, x) - _TRUTH[name](x)))
        assert err < 1e-9, (name, err)


def test_mutants_differ_from_base():
    x = np.linspace(-1.0, 1.0, 40)
    base = cm.eval_array("sin", x)
    mutants = cmut.generate_mutants("sin")
    assert mutants
    differ = sum(1 for _, f in mutants
                 if not np.allclose(np.nan_to_num(f(x)), np.nan_to_num(base), atol=1e-9))
    assert differ > 0


def test_loop_guard_prevents_hang():
    # mutated break conditions must terminate, not spin forever
    x = np.array([0.5, 3.0, -2.0])
    for _, f in cmut.generate_mutants("cos"):
        f(x)  # would never return without the guard


def test_build_benchmark_is_625():
    by = cmut.build_benchmark(FUNCTION_NAMES, total=625)
    assert sum(len(v) for v in by.values()) == 625
