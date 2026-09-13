"""Check an output row against the Buy or Wait? output contract."""

import re

from data import parse_date
from decide import Rules, installment_schedule

STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
CHANGE_RE = re.compile(r"^(stop:(event_\d+)|reduce_to:(event_\d+):(\d+(\.\d+)?))$")


def _parse_plan(text):
    if text == "none":
        return []
    payments = []
    for part in text.split("|"):
        date, amount = part.split(":")
        payments.append((parse_date(date), float(amount)))
    return payments


def validate_row(ds, request, row):
    errors = []
    profile = ds.profiles[request["user_id"]]
    rules = Rules.for_request(profile, request)
    rd = parse_date(request["request_date"])
    safe = float(row["amount_safe_to_pay"])
    status, method = row["affordability_status"], row["recommended_payment_method"]
    earliest = parse_date(row["earliest_date_for_full_payment"])

    if not 0 <= safe <= rules.requested + 1e-9:
        errors.append(f"amount_safe_to_pay {safe} outside [0, {rules.requested}]")
    if status not in STATUSES:
        errors.append(f"bad status {status}")
    if method not in METHODS:
        errors.append(f"bad method {method}")

    try:
        payments = _parse_plan(row["payment_plan"])
    except ValueError:
        return errors + [f"unparseable payment_plan {row['payment_plan']}"]
    if payments != sorted(payments, key=lambda p: p[0]):
        errors.append("payment_plan not chronological")
    if (method == "not_recommended") != (not payments):
        errors.append("payment_plan must be none exactly when not_recommended")
    if method not in ("not_recommended", "wait") and method not in rules.methods:
        errors.append(f"{method} not accepted by user")

    if status == "affordable_now" and (earliest != rd or payments != [(rd, rules.requested)]):
        errors.append("affordable_now requires earliest == request_date and a single full payment today")
    if method == "partial_payment":
        if not rules.allows_partial or not 0 < safe < rules.requested:
            errors.append("partial_payment not permitted")
        if (len(payments) != 2 or payments[0] != (rd, safe) or payments[1][0] != earliest
                or abs(sum(a for _, a in payments) - rules.requested) > 0.011 or earliest > rules.due):
            errors.append("partial_payment plan must be today:safe | earliest:remainder within the deadline")
    if method == "installments":
        schedules = [installment_schedule(o) for o in ds.options_by_request[request["request_id"]]
                     if o["payment_method"] == "installments"]
        if not any(len(s) == len(payments) and all(d1 == d2 and abs(a1 - a2) < 0.005 for (d1, a1), (d2, a2) in zip(s, payments))
                   for s in schedules):
            errors.append("installment plan does not match a supplied option")

    changes = [] if row["spending_changes_needed"] == "none" else row["spending_changes_needed"].split("|")
    if len(changes) > 3:
        errors.append("more than three spending changes")
    seen = set()
    for change in changes:
        m = CHANGE_RE.match(change)
        if not m:
            errors.append(f"bad change {change}")
            continue
        event_id = m.group(2) or m.group(3)
        event = ds.events_by_id.get(event_id)
        if event_id in seen:
            errors.append(f"multiple changes for {event_id}")
        seen.add(event_id)
        if not event or event["user_id"] != request["user_id"] or event["flexibility"] == "fixed":
            errors.append(f"{event_id} is not a flexible event of this user")
            continue
        allowed = rules.can_stop if m.group(2) else rules.can_reduce
        if event["category"] not in allowed or event["category"] in rules.protect:
            errors.append(f"{change} targets a category the user does not permit")
    return errors
