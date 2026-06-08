from amr.benchmark import load_reference, build_comparison

REF = load_reference()


def test_per_program_automr_kills_sum_to_374():
    total = sum(p["automr_killed"] for p in REF["per_program"].values())
    assert total == REF["aggregate"]["automr"]["killed"] == 374


def test_per_program_seeded_faults_sum_recorded_discrepancy():
    # 953 vs headline 625; pinned
    total = sum(p["seeded_faults"] for p in REF["per_program"].values())
    assert total == REF["aggregate"]["per_program_seeded_faults_sum"] == 953
    assert REF["aggregate"]["total_mutants_headline"] == 625


def test_per_program_rates_match_kills_over_faults():
    for name, p in REF["per_program"].items():
        expected = round(p["automr_killed"] / p["seeded_faults"] * 100, 1)
        assert abs(expected - p["automr_kill_rate_pct"]) < 0.05, name


def test_aggregate_headline_rates_match_counts():
    agg = REF["aggregate"]
    total = agg["total_mutants_headline"]
    assert abs(round(agg["mri"]["killed"] / total * 100, 2) - agg["mri"]["kill_rate_pct_vs_625"]) < 0.01
    assert abs(round(agg["automr"]["killed"] / total * 100, 2) - agg["automr"]["kill_rate_pct_vs_625"]) < 0.01


def test_eight_functions_present():
    assert set(REF["per_program"]) == {
        "abs", "asinh", "atan", "cos", "log1p", "log10", "sin", "tan"
    }


def test_build_comparison_pairs_ours_with_automr():
    our_summary = {"sin": {"kill_rate_pct": 71.79, "mr_count": 6, "runtime_seconds": 7.13}}
    rows = build_comparison(REF, our_summary)
    sin_row = next(r for r in rows if r["function"] == "sin")
    assert sin_row["automr_killed"] == 18
    assert sin_row["ours_kill_rate_pct"] == 71.79
    # missing result -> none
    abs_row = next(r for r in rows if r["function"] == "abs")
    assert abs_row["ours_kill_rate_pct"] is None
