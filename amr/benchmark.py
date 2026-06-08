"""AutoMR/MRI benchmark reference and comparison-table builder.

Numbers come from the AutoMR paper (Tables V/VI, Fig. 5), not a re-run. The
original 625 Java mutants aren't distributed, so we can't reproduce AutoMR's
kill rate locally. We cite their published results and put our kill rate next to
them, marked as a different mutant set.
"""
import csv
import json
import os

DEFAULT_REFERENCE = os.path.join(
    os.path.dirname(__file__), "..", "data", "benchmark", "automr_reference.json"
)


def load_reference(path=DEFAULT_REFERENCE):
    with open(path) as fh:
        return json.load(fh)


def load_our_summary(path):
    """Read our results/summary.csv into {function: {kill_rate_pct, mr_count, ...}}."""
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[row["function"]] = {
                "kill_rate_pct": float(row["kill_rate_pct"]),
                "mr_count": int(row["mr_count"]),
                "runtime_seconds": float(row["runtime_seconds"]),
            }
    return out


def build_comparison(reference, our_summary):
    """Pair AutoMR's published per-function kill rate with ours.

    The paper gives MRI only in aggregate, so per-function rows are AutoMR vs ours.
    """
    rows = []
    for name, ref in reference["per_program"].items():
        ours = our_summary.get(name)
        rows.append({
            "function": name,
            "automr_seeded_faults": ref["seeded_faults"],
            "automr_killed": ref["automr_killed"],
            "automr_kill_rate_pct": ref["automr_kill_rate_pct"],
            "ours_kill_rate_pct": ours["kill_rate_pct"] if ours else None,
            "ours_mr_count": ours["mr_count"] if ours else None,
        })
    return rows
