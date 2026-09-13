"""Output formatting and grounded decision explanations in the style of the solved samples."""


def plain_amount(x):
    """Amount for CSV fields: 25256, 17229139.2, 603.3."""
    return f"{x:.2f}".rstrip("0").rstrip(".")


def plan_amount(x):
    """Amount inside payment plans and reduce_to actions: 25256, 620.40."""
    return str(round(x)) if abs(x - round(x)) < 0.005 else f"{x:.2f}"


def money(currency, x):
    return f"{currency} {round(x):,}" if abs(x - round(x)) < 0.005 else f"{currency} {x:,.2f}"


def long_date(d):
    return f"{d.day} {d.strftime('%B')} {d.year}"


def format_plan(plan):
    if plan is None:
        return "none"
    return "|".join(f"{d.isoformat()}:{plan_amount(a)}" for d, a in plan.payments)


def format_changes(details):
    if not details:
        return "none"
    return "|".join(f"stop:{event_id}" if action == "stop" else f"reduce_to:{event_id}:{plan_amount(amount)}"
                    for action, event_id, amount, _ in details)


def _change_phrase(currency, details):
    stops = [f"the {label.lower()}" for action, _, _, label in details if action == "stop"]
    reduces = [f"the {label.lower()} to {money(currency, amount)}" for action, _, amount, label in details if action != "stop"]
    parts = []
    if stops:
        parts.append("Stop " + " and ".join(stops))
    if reduces:
        parts.append(("reduce " if stops else "Reduce ") + " and ".join(reduces))
    return " and ".join(parts)


def explain(request, profile, decision):
    c = profile["home_currency"]
    minimum = float(profile["minimum_balance_to_keep"])
    requested = float(request["requested_amount"])
    plan = decision.plan

    if plan is None:
        methods = set(profile["payment_methods_user_will_consider"].split("|"))
        if decision.safe_amount > 0 and methods == {"partial_payment"}:
            return (f"Do not proceed with the {money(c, requested)} request. Although {money(c, decision.safe_amount)} "
                    f"is available today, the full amount cannot be completed safely within 90 days.")
        due = request["desired_completion_date"]
        from data import parse_date
        return (f"Do not make this payment by {long_date(parse_date(due))}. None of the available options "
                f"keeps the {money(c, minimum)} minimum protected.")

    if plan.method == "wait":
        return (f"Pay {money(c, requested)} in full on {long_date(plan.payments[0][0])}. "
                f"Paying earlier would take the balance below the {money(c, minimum)} minimum.")

    if plan.method == "full_payment":
        action = f"pay {money(c, requested)} today"
    elif plan.method == "installments":
        action = (f"use {len(plan.payments)} installments of {money(c, plan.payments[0][1])}, "
                  f"starting {long_date(plan.payments[0][0])}")
    else:
        (_, first), (second_date, second) = plan.payments
        action = f"pay {money(c, first)} today and the remaining {money(c, second)} on {long_date(second_date)}"

    if decision.change_details:
        return f"{_change_phrase(c, decision.change_details)}, then {action}. This leaves at least {money(c, minimum)} available."
    if plan.method == "full_payment":
        return f"Pay {money(c, requested)} today. This leaves at least {money(c, minimum)} available over the next 90 days."
    if plan.method == "partial_payment":
        return (f"{action[0].upper()}{action[1:]}. This completes the full request and keeps the "
                f"{money(c, minimum)} minimum protected.")
    return f"{action[0].upper()}{action[1:]}. This leaves at least {money(c, minimum)} available."
