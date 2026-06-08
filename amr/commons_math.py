"""Python port of the 8 Apache Commons Math 2.2 functions (org.apache.commons.math.util.FastMath).

This keeps FastMath's real algorithm and coefficients: Cody-Waite range
reduction, the 1/8-radian sine/cosine/tangent tables, the polySine/polyCosine
Remez polynomials, the atan reference-angle polynomial, and the asinh/log1p
series. The one deliberate change is dropping FastMath's hi/lo split-double
arithmetic (it exists only for last-ULP accuracy); we use plain Python floats.

These are the subject programs for the mutation benchmark: source-level mutants
of these functions stand in for Commons Math 2.2's 625 Java-source mutants, which
aren't distributed. Each function is scalar; eval_array vectorizes over numpy.
"""
import math

import numpy as np

# entry i holds f(i/8)
EIGHTHS = [i * 0.125 for i in range(14)]
SIN_TABLE = [math.sin(a) for a in EIGHTHS]
COS_TABLE = [math.cos(a) for a in EIGHTHS]
TAN_TABLE = [math.tan(a) for a in EIGHTHS]

HALF_PI = 1.5707963267948966
LN2 = 0.6931471805599453


def _poly_sine(x):
    # sin(x) - x near 0
    x2 = x * x
    p = 2.7553817452272217e-6
    p = p * x2 + -1.9841269659586505e-4
    p = p * x2 + 0.008333333333329196
    p = p * x2 + -0.16666666666666666
    return p * x2 * x


def _poly_cosine(x):
    # cos(x) - 1 near 0
    x2 = x * x
    p = 2.479773539153719e-5
    p = p * x2 + -0.0013888888689039883
    p = p * x2 + 0.041666666666621166
    p = p * x2 + -0.49999999999999994
    return p * x2


def _sin_q(xa):
    # sine on first quadrant
    idx = int(xa * 8.0 + 0.5)
    eps = xa - EIGHTHS[idx]
    sin_eps = eps + _poly_sine(eps)
    cos_eps = 1.0 + _poly_cosine(eps)
    return SIN_TABLE[idx] * cos_eps + COS_TABLE[idx] * sin_eps


def _cos_q(xa):
    return _sin_q(HALF_PI - xa)


def _tan_q(xa, cotan):
    idx = int(xa * 8.0 + 0.5)
    eps = xa - EIGHTHS[idx]
    sin_eps = eps + _poly_sine(eps)
    cos_eps = 1.0 + _poly_cosine(eps)
    sina = SIN_TABLE[idx] * cos_eps + COS_TABLE[idx] * sin_eps
    cosa = COS_TABLE[idx] * cos_eps - SIN_TABLE[idx] * sin_eps
    if cotan:
        sina, cosa = cosa, sina
    return sina / cosa


def _cody_waite(xa):
    # reduce mod pi/2
    k = int(xa * 0.6366197723675814)
    while True:
        rem = (xa - k * 1.570796251296997
               - k * 7.549789948768648e-8 - k * 6.123233995736766e-17)
        if rem > 0.0:
            break
        k -= 1
    return k & 3, rem


def sin(x):
    if x != x or math.isinf(x):
        return float("nan")
    negative = x < 0
    xa = -x if negative else x
    if xa == 0.0:
        return 0.0
    if xa > HALF_PI:
        quadrant, xa = _cody_waite(xa)
    else:
        quadrant = 0
    if negative:
        quadrant ^= 2
    quadrant &= 3
    if quadrant == 0:
        return _sin_q(xa)
    if quadrant == 1:
        return _cos_q(xa)
    if quadrant == 2:
        return -_sin_q(xa)
    return -_cos_q(xa)


def cos(x):
    if x != x or math.isinf(x):
        return float("nan")
    xa = -x if x < 0 else x
    if xa > HALF_PI:
        quadrant, xa = _cody_waite(xa)
    else:
        quadrant = 0
    quadrant &= 3
    if quadrant == 0:
        return _cos_q(xa)
    if quadrant == 1:
        return -_sin_q(xa)
    if quadrant == 2:
        return -_cos_q(xa)
    return _sin_q(xa)


def tan(x):
    if x != x or math.isinf(x):
        return float("nan")
    negative = x < 0
    xa = -x if negative else x
    if xa == 0.0:
        return 0.0
    if xa > HALF_PI:
        quadrant, xa = _cody_waite(xa)
    else:
        quadrant = 0
    if xa > 1.5:
        xa = HALF_PI - xa
        quadrant ^= 1
        negative = not negative
    result = _tan_q(xa, False) if (quadrant & 1) == 0 else -_tan_q(xa, True)
    return -result if negative else result


def atan(x):
    if x != x:
        return float("nan")
    negate = x < 0
    xa = -x if negate else x
    if xa == 0.0:
        return x
    if xa > 1.633123935319537e16:
        return -HALF_PI if negate else HALF_PI
    if xa < 1.0:
        idx = int((-1.7168146928204136 * xa * xa + 8.0) * xa + 0.5)
    else:
        temp = 1.0 / xa
        idx = int(-((-1.7168146928204136 * temp * temp + 8.0) * temp) + 13.07)
    eps = (xa - TAN_TABLE[idx]) / (1.0 + xa * TAN_TABLE[idx])
    eps2 = eps * eps
    yb = 0.07490822288864472
    yb = yb * eps2 + -0.09088450866185192
    yb = yb * eps2 + 0.11111095942313305
    yb = yb * eps2 + -0.1428571423679182
    yb = yb * eps2 + 0.19999999999923582
    yb = yb * eps2 + -0.33333333333333287
    yb = yb * eps2 * eps
    result = EIGHTHS[idx] + eps + yb
    return -result if negate else result


def _ln(x):
    # ln via mantissa + series
    if x == 0.0:
        return float("-inf")
    if x < 0.0 or x != x:
        return float("nan")
    if math.isinf(x):
        return float("inf")
    m, e = math.frexp(x)        # x = m * 2**e
    if m < 0.7071067811865476:  # center mantissa near 1
        m *= 2.0
        e -= 1
    t = (m - 1.0) / (m + 1.0)
    t2 = t * t
    s = 1.0 / 11.0
    s = s * t2 + 1.0 / 9.0
    s = s * t2 + 1.0 / 7.0
    s = s * t2 + 1.0 / 5.0
    s = s * t2 + 1.0 / 3.0
    s = s * t2 + 1.0
    return e * LN2 + 2.0 * t * s


def log10(x):
    return _ln(x) * 0.4342944819032518  # 1 / ln(10)


def log1p(x):
    if x == -1.0:
        return float("-inf")
    if x > 1e-6 or x < -1e-6:
        return _ln(1.0 + x)
    # small x: taylor series
    y = x * 0.333333333333333 - 0.5
    y = y * x + 1.0
    return y * x


def asinh(a):
    negative = a < 0
    if negative:
        a = -a
    if a > 0.167:
        abs_asinh = _ln(math.sqrt(a * a + 1.0) + a)
    else:
        a2 = a * a
        if a > 0.097:
            abs_asinh = a * (1 - a2 * (1 / 3.0 - a2 * (1 / 5.0 - a2 * (1 / 7.0 - a2 * (1 / 9.0 - a2 * (1.0 / 11.0 - a2 * (1.0 / 13.0 - a2 * (1.0 / 15.0 - a2 * (1.0 / 17.0) * 15.0 / 16.0) * 13.0 / 14.0) * 11.0 / 12.0) * 9.0 / 10.0) * 7.0 / 8.0) * 5.0 / 6.0) * 3.0 / 4.0) / 2.0)
        elif a > 0.036:
            abs_asinh = a * (1 - a2 * (1 / 3.0 - a2 * (1 / 5.0 - a2 * (1 / 7.0 - a2 * (1 / 9.0 - a2 * (1.0 / 11.0 - a2 * (1.0 / 13.0) * 11.0 / 12.0) * 9.0 / 10.0) * 7.0 / 8.0) * 5.0 / 6.0) * 3.0 / 4.0) / 2.0)
        elif a > 0.0036:
            abs_asinh = a * (1 - a2 * (1 / 3.0 - a2 * (1 / 5.0 - a2 * (1 / 7.0 - a2 * (1 / 9.0) * 7.0 / 8.0) * 5.0 / 6.0) * 3.0 / 4.0) / 2.0)
        else:
            abs_asinh = a * (1 - a2 * (1 / 3.0 - a2 * (1 / 5.0) * 3.0 / 4.0) / 2.0)
    return -abs_asinh if negative else abs_asinh


def abs(x):  # noqa: A001 - mirrors FastMath.abs
    return -x if x < 0.0 else (0.0 if x == 0.0 else x)


SCALARS = {
    "abs": abs, "asinh": asinh, "atan": atan, "cos": cos,
    "log1p": log1p, "log10": log10, "sin": sin, "tan": tan,
}


def eval_array(name, x):
    """Apply the scalar Commons Math port over a numpy array."""
    fn = SCALARS[name]
    return np.array([fn(float(v)) for v in np.asarray(x, dtype=float).ravel()]
                    ).reshape(np.asarray(x, dtype=float).shape)
