"""Source-level mutants of the Commons Math 2.2 port (amr/commons_math.py).

Mutation operators mirror what Java mutation tools apply to source, which is how
MRI/AutoMR produced their 625 Java mutants: arithmetic operator replacement
(AOR), relational operator replacement (ROR), and constant replacement. Each
mutant changes one operator or constant in a function's call graph, is compiled,
and is exposed as a callable. These stand in for the undistributed 625 Java
mutants; they aren't identical, but they're built the same way from the same
algorithms.
"""
import ast
import math

import numpy as np

from . import commons_math as cm

# mutate function plus its helpers
_UNITS = {
    "abs": ["abs"],
    "asinh": ["asinh", "_ln"],
    "log1p": ["log1p", "_ln"],
    "log10": ["log10", "_ln"],
    "atan": ["atan"],
    "sin": ["sin", "_sin_q", "_cos_q", "_poly_sine", "_poly_cosine", "_cody_waite"],
    "cos": ["cos", "_sin_q", "_cos_q", "_poly_sine", "_poly_cosine", "_cody_waite"],
    "tan": ["tan", "_tan_q", "_poly_sine", "_poly_cosine", "_cody_waite"],
}

_AOR = {ast.Add: [ast.Sub, ast.Mult], ast.Sub: [ast.Add, ast.Mult],
        ast.Mult: [ast.Div, ast.Add], ast.Div: [ast.Mult, ast.Sub],
        ast.Pow: [ast.Mult], ast.Mod: [ast.Mult]}
_ROR = {ast.Lt: [ast.LtE, ast.GtE], ast.Gt: [ast.GtE, ast.LtE],
        ast.LtE: [ast.Lt, ast.Gt], ast.GtE: [ast.Gt, ast.Lt],
        ast.Eq: [ast.NotEq], ast.NotEq: [ast.Eq]}

_GLOBALS = {"math": math, "np": np, "EIGHTHS": cm.EIGHTHS, "SIN_TABLE": cm.SIN_TABLE,
            "COS_TABLE": cm.COS_TABLE, "TAN_TABLE": cm.TAN_TABLE,
            "HALF_PI": cm.HALF_PI, "LN2": cm.LN2}

import inspect


def _unit_source(name):
    return "\n".join(inspect.getsource(getattr(cm, fn)) for fn in _UNITS[name])


def _const_options(v):
    if isinstance(v, bool):
        return []
    if v == 0:
        return [1.0]
    return [-v, v * 1.1]


def _mutable_nodes(tree):
    """Walk tree once, tag mutable nodes with _mid, return [(mid, n_options)]."""
    out = []
    mid = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and type(node.op) in _AOR:
            n = len(_AOR[type(node.op)])
        elif (isinstance(node, ast.Compare) and len(node.ops) == 1
              and type(node.ops[0]) in _ROR):
            n = len(_ROR[type(node.ops[0])])
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            n = len(_const_options(node.value))
        else:
            continue
        if n:
            node._mid = mid
            out.append((mid, n))
            mid += 1
    return out


_LOOP_LIMIT = 1000


def _guard_loops(tree):
    """Rewrite `while COND: BODY` as a bounded loop so a mutated break condition
    can't hang. Exhausting the cap raises, which marks the mutant killed."""
    class G(ast.NodeTransformer):
        def visit_While(self, node):
            self.generic_visit(node)
            guard = ast.If(
                test=ast.UnaryOp(op=ast.Not(), operand=node.test),
                body=[ast.Break()], orelse=[])
            return ast.For(
                target=ast.Name(id="__g", ctx=ast.Store()),
                iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                              args=[ast.Constant(_LOOP_LIMIT)], keywords=[]),
                body=[guard] + node.body,
                orelse=[ast.Raise(exc=ast.Call(
                    func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                    args=[ast.Constant("loop-timeout")], keywords=[]), cause=None)])
    G().visit(tree)


def _apply(src, target_mid, option):
    tree = ast.parse(src)
    _mutable_nodes(tree)

    class T(ast.NodeTransformer):
        def visit_BinOp(self, node):
            self.generic_visit(node)
            if getattr(node, "_mid", None) == target_mid:
                node.op = _AOR[type(node.op)][option]()
            return node

        def visit_Compare(self, node):
            self.generic_visit(node)
            if getattr(node, "_mid", None) == target_mid:
                node.ops = [_ROR[type(node.ops[0])][option]()]
            return node

        def visit_Constant(self, node):
            if getattr(node, "_mid", None) == target_mid:
                node.value = _const_options(node.value)[option]
            return node

    T().visit(tree)
    _guard_loops(tree)
    ast.fix_missing_locations(tree)
    return tree


def _array_fn(scalar):
    def f(x):
        arr = np.asarray(x, dtype=float)
        out = np.empty(arr.size, dtype=float)
        flat = arr.ravel()
        for i in range(arr.size):
            try:
                out[i] = scalar(float(flat[i]))
            except Exception:
                # crash counts as killed
                out[i] = float("nan")
        return out.reshape(arr.shape)
    return f


def generate_mutants(name):
    """Compile every one-operator mutant of a function's unit.

    Returns a list of (description, array_callable). Order is deterministic.
    """
    src = _unit_source(name)
    candidates = [(mid, opt) for mid, n in _mutable_nodes(ast.parse(src))
                  for opt in range(n)]
    mutants = []
    for mid, opt in candidates:
        tree = _apply(src, mid, opt)
        ns = dict(_GLOBALS)
        exec(compile(tree, f"<mutant {name} {mid}.{opt}>", "exec"), ns)
        mutants.append((f"{name}:{mid}.{opt}", _array_fn(ns[name])))
    return mutants


def build_benchmark(functions, total=625, seed=0):
    """Select exactly `total` mutants across `functions`, grouped by function.

    The full candidate pool (>625) is permuted with a fixed seed and the first
    `total` are kept, so the set is reproducible. Returns {function: [(desc, fn)]}.
    """
    pool = []
    for name in functions:
        for desc, fn in generate_mutants(name):
            pool.append((name, desc, fn))
    rng = np.random.default_rng(seed)
    keep = sorted(rng.permutation(len(pool))[:total])
    by_func = {}
    for i in keep:
        name, desc, fn = pool[i]
        by_func.setdefault(name, []).append((desc, fn))
    return by_func
