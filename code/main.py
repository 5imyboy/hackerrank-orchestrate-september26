"""Buy or Wait? — decide how each request in dataset/requests.csv should be paid and write output.csv.

Usage:
    python3 code/main.py                       # use cached evidence (code/cache/evidence.json)
    python3 code/main.py --evidence extract    # call Claude for uncached messages/images, write usage report
    python3 code/main.py --evidence none       # ignore messages and images
"""

import argparse
import csv
import sys
from pathlib import Path

from data import DATASET_DIR, REPO_ROOT, Dataset
from decide import decide
from explain import explain, format_changes, format_plan, plain_amount
from forecast import ForecastConfig, build_forecast
from validate import validate_row

COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
    "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation",
]
USAGE_REPORT = Path(__file__).resolve().parent / "evaluation" / "usage_report.md"

# Chosen by sweeping configs against dataset/sample_requests.csv (code/evaluation/main.py --grid).
CONFIG = ForecastConfig(variable_estimator="median", same_day="credit_first",
                        failed_debit_mode="suppress", project_irregular_income=True)


def build_row(ds, request, facts=(), config=CONFIG):
    forecast = build_forecast(ds, request, config, facts)
    decision = decide(ds, request, forecast)
    return {
        "request_id": request["request_id"],
        "amount_safe_to_pay": plain_amount(decision.safe_amount),
        "affordability_status": decision.status,
        "recommended_payment_method": decision.method,
        "payment_plan": format_plan(decision.plan),
        "earliest_date_for_full_payment": decision.earliest.isoformat() if decision.earliest else "",
        "spending_changes_needed": format_changes(decision.change_details),
        "decision_explanation": explain(request, ds.profiles[request["user_id"]], decision),
    }


def load_facts(ds, requests, mode):
    """Returns (facts_by_user, evidence_item_count)."""
    if mode == "none":
        return {}, 0
    import evidence
    facts, n_items, _ = evidence.facts_by_user(ds, {r["user_id"] for r in requests}, extract=(mode == "extract"))
    return facts, n_items


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", default=str(DATASET_DIR / "requests.csv"))
    parser.add_argument("--out", default=str(REPO_ROOT / "output.csv"))
    parser.add_argument("--evidence", choices=["none", "cache", "extract"], default="cache")
    args = parser.parse_args()

    ds = Dataset()
    requests = ds.load_requests(args.requests)
    facts, n_items = load_facts(ds, requests, args.evidence)

    rows, problems = [], 0
    for request in requests:
        row = build_row(ds, request, facts.get(request["user_id"], []))
        for error in validate_row(ds, request, row):
            problems += 1
            print(f"[validate] {request['request_id']}: {error}", file=sys.stderr)
        rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out} ({problems} validation issues)")

    if args.evidence == "extract":
        import usage
        command = "python3 code/main.py --evidence extract" + ("" if args.requests.endswith("requests.csv") else f" --requests {args.requests}")
        usage.write_report(USAGE_REPORT, len(requests), n_items, command)
        print(f"wrote token usage report to {USAGE_REPORT}")


if __name__ == "__main__":
    main()
