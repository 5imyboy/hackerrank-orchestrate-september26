"""Score the pipeline against dataset/sample_requests.csv.

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


def _plan(text):
    return [] if text == "none" else [(p.split(":")[0], round(float(p.split(":")[1]), 2)) for p in text.split("|")]


def field_matches(sample, row):
    requested = float(sample["requested_amount"])
    return {
        "amount_safe_to_pay": abs(float(row["amount_safe_to_pay"]) - float(sample["amount_safe_to_pay"])) <= 0.01 * requested,
        "affordability_status": row["affordability_status"] == sample["affordability_status"],
        "recommended_payment_method": row["recommended_payment_method"] == sample["recommended_payment_method"],
        "payment_plan": _plan(row["payment_plan"]) == _plan(sample["payment_plan"]),
        "earliest_date_for_full_payment": row["earliest_date_for_full_payment"] == sample["earliest_date_for_full_payment"],
        "spending_changes_needed": row["spending_changes_needed"] == sample["spending_changes_needed"],
    }


def score(ds, samples, facts, config, verbose=False):
    counts = Counter()
    for s in samples:
        row = build_row(ds, s, facts.get(s["user_id"], []), config)
        matches = field_matches(s, row)
        counts.update(k for k, ok in matches.items() if ok)
        if verbose:
            errors = validate_row(ds, s, row)
            bad = [k for k, ok in matches.items() if not ok]
            if bad or errors:
                print(f"\n{s['request_id']}  wrong: {', '.join(bad) or '-'}")
                for k in bad:
                    print(f"   {k:<32} got {row[k] or '-':<45} want {s[k] or '-'}")
                for e in errors:
                    print(f"   [validate] {e}")
                print(f"   explanation: {row['decision_explanation']}")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--evidence", choices=["none", "cache", "extract"], default="none")
    args = parser.parse_args()

    ds = Dataset()
    samples = read_csv(DATASET_DIR / "sample_requests.csv")
    facts, _ = load_facts(ds, samples, args.evidence)
    n = len(samples)

    if args.grid:
        results = []
        for var, same_day, failed, irregular in itertools.product(
            ["last", "mean", "median", "mean3"], ["debit_first", "credit_first"],
            ["suppress", "reserve", "ignore"], [False, True],
        ):
            cfg = replace(CONFIG, variable_estimator=var, same_day=same_day,
                          failed_debit_mode=failed, project_irregular_income=irregular)
            counts = score(ds, samples, facts, cfg)
            results.append((sum(counts.values()), counts, cfg))
        results.sort(key=lambda r: r[0], reverse=True)
        for total, counts, cfg in results[:8]:
            print(f"{total:>3}/{n * len(FIELDS)}  " + "  ".join(f"{k[:12]}={counts[k]}" for k in FIELDS)
                  + f"  | var={cfg.variable_estimator} same_day={cfg.same_day} failed={cfg.failed_debit_mode} irregular={cfg.project_irregular_income}")
        return

    counts = score(ds, samples, facts, CONFIG, verbose=True)
    print("\n" + "\n".join(f"{k:<32} {counts[k]:>2}/{n}" for k in FIELDS))


if __name__ == "__main__":
    main()
