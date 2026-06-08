"""Evaluate our inferred MRs against the Commons Math 2.2 mutation benchmark.

Loads the MRs from results/<func>.json, builds 625 source-level mutants of the
Commons Math 2.2 port (amr/commons_math.py), and counts how many each function's
MRs kill. A mutant is killed when its MR residual exceeds the MR's own noise floor
(its residual on the correct function) by a factor of KILL_K. That relative
threshold means an MR never fires on the correct function, so false detections are
zero by construction, while tight MRs still catch small faults.

Writes results/commons_benchmark.json. This is our "Ours" column, built the same
way as AutoMR/MRI's 625 mutants (source-level operator mutations), though not the
identical Java set.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from amr.functions import FUNCTION_NAMES, get_domain, needs_clamp
from amr import commons_math as cm
from amr import commons_mutants as cmut
from amr.data import sample_inputs

N_INPUTS = 200
EVAL_SEED = 100
KILL_K = 10.0      # a fault must exceed the MR noise floor by this factor
ABS_FLOOR = 1e-9   # guards MRs that are exact on the base


def _params(mr):
    return (mr["c1"], mr["c2"], mr["a"], mr["b"], mr["d"])


def _second_input(name, a, b, x):
    # clamp like the pipeline
    x2 = a * x + b
    if needs_clamp(name):
        lo, hi = get_domain(name)
        x2 = np.clip(x2, lo, hi)
    return x2


def _mask_and_base(name, params, x):
    c1, c2, a, b, d = params
    bx = cm.eval_array(name, x)
    bx2 = cm.eval_array(name, _second_input(name, a, b, x))
    mask = np.isfinite(bx) & np.isfinite(bx2)
    return bx, bx2, mask


def _base_residual(name, params, x):
    c1, c2, a, b, d = params
    bx, bx2, mask = _mask_and_base(name, params, x)
    if not mask.any():
        return float("inf")
    res = (c1 * bx + c2 * bx2 + d)[mask]
    return float(np.max(np.abs(res)))


def _killed(name, params, mutant_fn, x, threshold):
    c1, c2, a, b, d = params
    _, _, mask = _mask_and_base(name, params, x)
    if not mask.any():
        return False
    x2 = _second_input(name, a, b, x)
    res = (c1 * mutant_fn(x) + c2 * mutant_fn(x2) + d)[mask]
    if not np.all(np.isfinite(res)):
        return True  # diverged where base is finite
    return bool(np.max(np.abs(res)) > threshold)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results", help="dir with per-function MR JSON")
    ap.add_argument("--out", default="results/commons_benchmark.json")
    ap.add_argument("--total", type=int, default=625)
    ap.add_argument("--kill-k", type=float, default=KILL_K)
    args = ap.parse_args()

    benchmark = cmut.build_benchmark(FUNCTION_NAMES, total=args.total)
    per_func, total_killed, total_mutants, total_false = {}, 0, 0, 0

    for name in FUNCTION_NAMES:
        mrs_path = os.path.join(args.results, f"{name}.json")
        mrs = json.load(open(mrs_path))["mrs"] if os.path.exists(mrs_path) else []
        x = sample_inputs(name, N_INPUTS, EVAL_SEED)

        # each MR's kill threshold = K x its residual on the correct function
        oracles = [(p, max(ABS_FLOOR, args.kill_k * _base_residual(name, p, x)))
                   for p in (_params(m) for m in mrs)]
        false_det = sum(1 for p, thr in oracles if _base_residual(name, p, x) > thr)

        mutants = benchmark.get(name, [])
        killed = sum(1 for _, m in mutants
                     if any(_killed(name, p, m, x, thr) for p, thr in oracles))

        n = len(mutants)
        per_func[name] = {
            "killed": killed, "n_mutants": n,
            "kill_rate_pct": round(killed / n * 100, 1) if n else 0.0,
            "mr_count": len(mrs), "false_detections": false_det,
        }
        total_killed += killed
        total_mutants += n
        total_false += false_det
        print(f"{name:6} killed {killed:3}/{n:<3} "
              f"({per_func[name]['kill_rate_pct']:5.1f}%)  false_det={false_det}")

    summary = {
        "subject": "Apache Commons Math 2.2 port (amr/commons_math.py)",
        "mutant_kind": "source-level AOR/ROR/constant mutants",
        "n_inputs": N_INPUTS, "kill_k": args.kill_k,
        "aggregate": {
            "killed": total_killed, "n_mutants": total_mutants,
            "kill_rate_pct": round(total_killed / total_mutants * 100, 1) if total_mutants else 0.0,
            "false_detections": total_false,
        },
        "per_program": per_func,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    agg = summary["aggregate"]
    print(f"\nTOTAL killed {agg['killed']}/{agg['n_mutants']} "
          f"({agg['kill_rate_pct']}%)  false_detections={agg['false_detections']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
