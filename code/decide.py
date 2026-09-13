"""Choose a recommendation: candidate plans, safety checks, ranking, and spending changes."""

import datetime as dt
import itertools
from dataclasses import dataclass, field

from data import parse_date

MAX_CHANGES = 3


@dataclass
class Plan:
    method: str
    payments: list                                  # [(date, amount)] in chronological order
    changes: dict = field(default_factory=dict)     # event_id -> new amount (0 = stop)
    option_id: str = ""

    @property
    def total(self):
        return sum(amount for _, amount in self.payments)

    def rank_key(self, due):
        # Problem statement order: meets deadline, no changes, lowest total, earlier start, fewer payments.
        return (self.payments[-1][0] > due, bool(self.changes), round(self.total, 2),
                self.payments[0][0], len(self.payments))


@dataclass
class Decision:
    safe_amount: float
    earliest: dt.date | None
    status: str
    method: str
    plan: Plan | None
    change_details: list = field(default_factory=list)  # [(action, event_id, new_amount, description)]


@dataclass
class Rules:
    requested: float
    due: dt.date
    methods: set
    allows_partial: bool
    max_months: int
    protect: set
    can_stop: set
    can_reduce: set

    @classmethod
    def for_request(cls, profile, request):
        split = lambda value: set(filter(None, value.split("|")))  # noqa: E731
        return cls(
            requested=float(request["requested_amount"]),
            due=parse_date(request["desired_completion_date"]),
            methods=split(profile["payment_methods_user_will_consider"]),
            allows_partial=request["allows_partial_payment"].strip().lower() == "true",
            max_months=int(profile["max_installment_months"]) if profile["max_installment_months"] else 0,
            protect=split(profile["expense_categories_to_protect"]),
            can_stop=split(profile["expense_categories_user_is_willing_to_stop"]),
            can_reduce=split(profile["expense_categories_user_is_willing_to_reduce"]),
        )


def installment_schedule(option):
    first = parse_date(option["first_payment_date"])
    step = int(option["payment_frequency_days"] or 0)
    amount = float(option["payment_amount"])
    return [(first + dt.timedelta(days=step * k), amount) for k in range(int(option["number_of_payments"]))]


def candidate_plans(options, forecast, rules, include_wait=True):
    rd, requested = forecast.request_date, rules.requested
    plans = []
    if "full_payment" in rules.methods and forecast.is_safe([(rd, requested)]):
        plans.append(Plan("full_payment", [(rd, requested)]))

    wants_partial = "partial_payment" in rules.methods and rules.allows_partial
    earliest = forecast.earliest_full_payment(requested) if (wants_partial or include_wait) else None
    if wants_partial:
        safe = round(forecast.safe_amount_today(requested), 2)
        if 0 < safe < requested and earliest and earliest <= rules.due:
            payments = [(rd, safe), (earliest, round(requested - safe, 2))]
            if forecast.is_safe(payments):
                plans.append(Plan("partial_payment", payments))

    if "installments" in rules.methods and rules.max_months:
        for option in options:
            if option["payment_method"] != "installments" or int(option["number_of_payments"]) > rules.max_months:
                continue
            payments = installment_schedule(option)
            if payments[0][0] >= rd and forecast.is_safe(payments):
                plans.append(Plan("installments", payments, option_id=option["payment_option_id"]))

    if include_wait and "full_payment" in rules.methods and earliest and earliest > rd:
        plans.append(Plan("wait", [(earliest, requested)]))
    return plans


def change_options(forecast, rules):
    """Allowed stop/reduce actions on projected flexible streams, keyed by their latest event."""
    streams = {}
    for f in forecast.flows:
        if f.projected and f.amount < 0 and f.flexibility != "fixed" and f.category not in rules.protect:
            streams.setdefault(f.event_id, f)
    options = []
    for event_id, f in streams.items():
        if f.flexibility in ("stoppable", "reducible_or_stoppable") and f.category in rules.can_stop:
            options.append(("stop", event_id, 0.0, f.label))
        if (f.flexibility in ("reducible", "reducible_or_stoppable") and f.category in rules.can_reduce
                and f.min_allowed is not None and f.min_allowed < -f.amount):
            options.append(("reduce_to", event_id, f.min_allowed, f.label))
    return options


def best_change_plan(options, forecast, rules):
    found = []
    actions = change_options(forecast, rules)
    for size in range(1, MAX_CHANGES + 1):
        for combo in itertools.combinations(actions, size):
            if len({a[1] for a in combo}) < size:
                continue  # stop and reduce on the same event are mutually exclusive
            changes = {a[1]: a[2] for a in combo}
            changed = forecast.with_changes(changes)
            cut = sum(-f.amount - changes[f.event_id] for f in forecast.flows
                      if f.projected and f.amount < 0 and f.event_id in changes)
            for plan in candidate_plans(options, changed, rules, include_wait=False):
                if plan.payments[-1][0] > rules.due:
                    continue
                plan.changes = changes
                found.append(((*plan.rank_key(rules.due), round(cut, 2), size, plan.option_id), plan, combo))
    if not found:
        return None, []
    _, plan, combo = min(found, key=lambda item: item[0])
    return plan, list(combo)


def decide(ds, request, forecast):
    rules = Rules.for_request(ds.profiles[request["user_id"]], request)
    options = ds.options_by_request[request["request_id"]]
    safe = round(forecast.safe_amount_today(rules.requested), 2)
    earliest = forecast.earliest_full_payment(rules.requested)

    plans = candidate_plans(options, forecast, rules)
    best = min(plans, key=lambda p: (p.rank_key(rules.due), p.option_id)) if plans else None
    details = []
    if best is None or best.payments[-1][0] > rules.due:
        change_plan, combo = best_change_plan(options, forecast, rules)
        if change_plan and (best is None or change_plan.rank_key(rules.due) < best.rank_key(rules.due)):
            best, details = change_plan, combo

    if best is None:
        return Decision(safe, earliest, "not_affordable", "not_recommended", None)
    if best.changes:
        status = "affordable_with_plan"
    elif best.method == "full_payment":
        status = "affordable_now"
    elif best.method == "wait":
        status = "affordable_later"
    else:
        status = "affordable_with_plan"
    return Decision(safe, earliest, status, best.method, best, details)
