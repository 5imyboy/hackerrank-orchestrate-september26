"""Score the pipeline against dataset/sample_requests.csv.

amount_safe_to_pay is scored relative to the expected value, with sign: (got - expected) / expected.
A positive error recommends spending more than expected, which is the risky direction for personal finances.

Usage:
    python3 code/evaluation/main.py                 # per-field accuracy + mismatches for the default config
    python3 code/evaluation/main.py --grid          # sweep forecast configs, print the best
    python3 code/evaluation/main.py --evidence cache
"""

import argparse
import itertools
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import DATASET_DIR, Dataset, read_csv  # noqa: E402
from main import CONFIG, build_row, load_facts  # noqa: E402
from validate import validate_row  # noqa: E402

FIELDS = ["amount_safe_to_pay", "affordability_status", "recommended_payment_method", "payment_plan",
          "earliest_date_for_full_payment", "spending_changes_needed"]
DECISION_FIELDS = FIELDS[1:]


def _plan(text):
    return [] if text == "none" else [(p.split(":")[0], round(float(p.split(":")[1]), 2)) for p in text.split("|")]


def relative_error(sample, row):
    expected, got = float(sample["amount_safe_to_pay"]), float(row["amount_safe_to_pay"])
    if abs(got - expected) <= 0.01:
        return 0.0
    return (got - expected) / expected if expected else float("inf")


def field_matches(sample, row):
    return {
        "amount_safe_to_pay": abs(relative_error(sample, row)) <= 0.01,
        "affordability_status": row["affordability_status"] == sample["affordability_status"],
        "recommended_payment_method": row["recommended_payment_method"] == sample["recommended_payment_method"],
        "payment_plan": _plan(row["payment_plan"]) == _plan(sample["payment_plan"]),
        "earliest_date_for_full_payment": row["earliest_date_for_full_payment"] == sample["earliest_date_for_full_payment"],
        "spending_changes_needed": row["spending_changes_needed"] == sample["spending_changes_needed"],
    }


def score(ds, samples, facts, config, verbose=False):
    """Returns (field match counts, {request_id: signed relative error of amount_safe_to_pay})."""
    counts, errors = Counter(), {}
    for s in samples:
        row = build_row(ds, s, facts.get(s["user_id"], []), config)
        matches = field_matches(s, row)
        counts.update(k for k, ok in matches.items() if ok)
        errors[s["request_id"]] = relative_error(s, row)
        if verbose:
            problems = validate_row(ds, s, row)
            bad = [k for k, ok in matches.items() if not ok]
            if bad or problems:
                print(f"\n{s['request_id']}  wrong: {', '.join(bad) or '-'}")
                for k in bad:
                    print(f"   {k:<32} got {row[k] or '-':<45} want {s[k] or '-'}")
                for p in problems:
                    print(f"   [validate] {p}")
                print(f"   explanation: {row['decision_explanation']}")
    return counts, errors


def safe_amount_summary(errors):
    n = len(errors)
    over = {k: e for k, e in errors.items() if e > 0}
    bands = "  ".join(f"within {int(t * 100)}%={sum(abs(e) <= t for e in errors.values())}/{n}" for t in (0.01, 0.05, 0.10))
    worst = max(over.items(), key=lambda kv: kv[1], default=(None, 0.0))
    return (f"exact={sum(e == 0 for e in errors.values())}/{n}  {bands}  over-estimates={len(over)}/{n}"
            + (f"  worst over={worst[1]:+.1%} ({worst[0]})" if worst[0] else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--evidence", choices=["none", "cache", "extract"], default="cache")
    args = parser.parse_args()

    ds = Dataset()
    samples = read_csv(DATASET_DIR / "sample_requests.csv")
    facts, _ = load_facts(ds, samples, args.evidence)
    n = len(samples)

    if args.grid:
        results = []
        for var, buffer, fixed, same_day, failed, irregular in itertools.product(
            ["median", "mean", "mean3", "max3"], [1.0, 1.1, 1.2, 1.25, 1.3], ["last", "max"],
            ["debit_first", "credit_first"], ["suppress", "reserve"], [False, True],
        ):
            cfg = replace(CONFIG, variable_estimator=var, variable_buffer=buffer, fixed_estimator=fixed,
                          same_day=same_day, failed_debit_mode=failed, project_irregular_income=irregular)
            counts, errors = score(ds, samples, facts, cfg)
            overs = sum(e > 0 for e in errors.values())
            results.append((sum(counts[k] for k in DECISION_FIELDS), overs, max(errors.values()), counts, errors, cfg))
        results.sort(key=lambda r: (-r[0], r[1], r[2]))
        for decisions, overs, worst, counts, errors, cfg in results[:12]:
            print(f"decisions={decisions}/{n * len(DECISION_FIELDS)} overs={overs}/{n} worst={worst:+.1%}  "
                  f"| var={cfg.variable_estimator} buffer={cfg.variable_buffer} fixed={cfg.fixed_estimator} same_day={cfg.same_day} "
                  f"failed={cfg.failed_debit_mode} irregular={cfg.project_irregular_income}")
        return

    counts, errors = score(ds, samples, facts, CONFIG, verbose=True)
    print()
    for k in DECISION_FIELDS:
        print(f"{k:<32} {counts[k]:>2}/{n}")
    print(f"{'amount_safe_to_pay':<32} {safe_amount_summary(errors)}")


if __name__ == "__main__":
    main()
