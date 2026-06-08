"""Print the MRI / AutoMR / Ours comparison table.

MRI and AutoMR columns are the published paper numbers
(data/benchmark/automr_reference.json). The "Ours" column comes from
results/commons_benchmark.json: our MRs run against 625 source-level mutants of
the Commons Math 2.2 port, built the same way as AutoMR's mutants but not the
identical Java set.
"""
import argparse, csv, json, os, statistics, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amr.benchmark import load_reference, build_comparison


def _ours_summary(path):
    data = json.load(open(path))
    summary = {n: {"kill_rate_pct": p["kill_rate_pct"], "mr_count": p["mr_count"],
                   "runtime_seconds": 0.0}
               for n, p in data["per_program"].items()}
    return summary, data["aggregate"]


def _ours_cost(path):
    # runtime per MR and avg PSO iters from the inference run
    rows = list(csv.DictReader(open(path)))
    rt = sum(float(r["runtime_seconds"]) for r in rows)
    mr = sum(int(r["mr_count"]) for r in rows)
    iters = statistics.mean(float(r["avg_pso_iterations"]) for r in rows)
    return rt / mr if mr else 0.0, iters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", default="results/commons_benchmark.json")
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--out", default=None, help="optional path to write the markdown table")
    args = ap.parse_args()

    ref = load_reference()
    ours, ours_agg = _ours_summary(args.ours) if os.path.exists(args.ours) else ({}, None)
    rows = build_comparison(ref, ours)

    agg = ref["aggregate"]
    if ours_agg:
        ours_killed = f"{ours_agg['killed']}/{ours_agg['n_mutants']}"
        ours_rate = f"{ours_agg['kill_rate_pct']}%"
        ours_false = ours_agg["false_detections"]
    else:
        ours_killed, ours_rate, ours_false = "—", "—", "—"

    if os.path.exists(args.summary):
        rt_per_mr, pso_iters = _ours_cost(args.summary)
        ours_rt = f"{rt_per_mr:.1f}"
        ours_pso = f"8x{pso_iters:.0f} (50 part.)"
    else:
        ours_rt, ours_pso = "—", "—"

    lines = []
    lines.append("## AutoMR / MRI / Ours benchmark comparison")
    lines.append("")
    lines.append(f"Subject programs: {ref['source']['subject_programs']}")
    lines.append(f"Source: {ref['source']['paper']} ({ref['source']['repo']})")
    lines.append("")
    lines.append("### Aggregate")
    lines.append("")
    lines.append("| Metric | MRI (baseline) | AutoMR (upper bound) | Ours (LLM-guided PSO) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Mutants killed | {agg['mri']['killed']}/{agg['total_mutants_headline']} "
                 f"| {agg['automr']['killed']}/{agg['total_mutants_headline']} | {ours_killed} |")
    lines.append(f"| Kill rate | {agg['mri']['kill_rate_pct_vs_625']}% "
                 f"| {agg['automr']['kill_rate_pct_vs_625']}% | {ours_rate} |")
    lines.append(f"| False detections | {agg['false_detections']} "
                 f"| {agg['false_detections']} | {ours_false} |")
    lines.append(f"| Runtime per MR (s) | {agg['mri']['avg_seconds_per_mr_and_fault']} "
                 f"| {agg['automr']['avg_seconds_per_mr_and_fault']} | {ours_rt} |")
    lines.append(f"| PSO runs x iters | {agg['pso_runs']}x{agg['pso_iterations']} "
                 f"| {agg['pso_runs']}x{agg['pso_iterations']} | {ours_pso} |")
    lines.append("")
    lines.append("### Per-function")
    lines.append("")
    lines.append("| Function | AutoMR killed/faults | AutoMR kill % | Ours kill %* | Ours #MR |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        ours_kr = "—" if r["ours_kill_rate_pct"] is None else f"{r['ours_kill_rate_pct']}"
        ours_mr = "—" if r["ours_mr_count"] is None else f"{r['ours_mr_count']}"
        lines.append(f"| {r['function']} | {r['automr_killed']}/{r['automr_seeded_faults']} "
                     f"| {r['automr_kill_rate_pct']} | {ours_kr} | {ours_mr} |")
    lines.append("")
    lines.append("\\* Ours kills source-level mutants of the Commons Math 2.2 port, built "
                 "the same way as AutoMR's mutants but not the identical Java set; AutoMR's "
                 "per-function denominators are its own.")
    for note in ref["source"]["notes"]:
        lines.append(f"- {note}")

    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"\n[compare_benchmark] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
