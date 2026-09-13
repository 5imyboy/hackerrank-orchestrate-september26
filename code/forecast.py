"""90-day cash-flow forecast: recurring streams, known future events, and safety checks."""

import calendar
import datetime as dt
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, replace

from data import parse_date

HORIZON_DAYS = 90
EPS = 1e-6
VARIABLE_CATEGORIES = frozenset({"groceries", "transport", "dining"})
NON_STREAM_TYPES = frozenset({"refund", "investment_purchase", "investment_sale", "investment_valuation"})


@dataclass(frozen=True)
class ForecastConfig:
    variable_estimator: str = "mean"   # amount for pooled variable spending
    fixed_estimator: str = "last"      # amount for monthly debits
    same_day: str = "debit_first"      # debit_first: bills before salary on the same day
    include_due_today: bool = True     # project occurrences due on request_date
    salary_recurs_after_scheduled: bool = True
    failed_debit_mode: str = "suppress"  # suppress: an unretried failed debit replaces that cycle | reserve | ignore
    project_irregular_income: bool = False
    monthly_grace_days: int = 5
    pool_stale_factor: float = 2.0


@dataclass
class Flow:
    date: dt.date
    amount: float                      # signed: debit < 0, credit > 0
    category: str
    label: str
    event_id: str = ""                 # latest historical event of the stream
    flexibility: str = "fixed"
    min_allowed: float | None = None
    projected: bool = False
    kind: str = "known"                # known | monthly | pool | evidence


def add_months(d, n):
    y, m = divmod(d.month - 1 + n, 12)
    year, month = d.year + y, m + 1
    return dt.date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def estimate(values, how):
    if how == "last":
        return values[-1]
    if how == "max":
        return max(values)
    if how == "min":
        return min(values)
    if how == "median":
        return statistics.median(values)
    if how == "mean3":
        return statistics.mean(values[-3:])
    if how == "max3":
        return max(values[-3:])
    return statistics.mean(values)


def _is_monthly(dates):
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    return bool(gaps) and all(27 <= g <= 32 for g in gaps)


@dataclass
class Forecast:
    request_date: dt.date
    end: dt.date
    balance: float
    minimum: float
    flows: list = field(default_factory=list)
    config: ForecastConfig = ForecastConfig()

    def lowest_balance(self, payments=()):
        """Lowest balance from request_date through the horizon, with extra payments applied."""
        by_day = defaultdict(lambda: [0.0, 0.0, 0.0])  # debits, credits, payments
        for f in self.flows:
            by_day[f.date][0 if f.amount < 0 else 1] += f.amount
        for d, amount in payments:
            by_day[d][2] += amount
        b = low = self.balance
        for d in sorted(by_day):
            if d > self.end:
                break
            debits, credits, paid = by_day[d]
            if self.config.same_day == "debit_first":
                b += debits
                low = min(low, b)
                b += credits - paid
            else:
                b += debits + credits - paid
            low = min(low, b)
        return low

    def is_safe(self, payments=()):
        return self.lowest_balance(payments) >= self.minimum - EPS

    def safe_amount_today(self, requested):
        return max(0.0, min(requested, self.lowest_balance() - self.minimum))

    def earliest_full_payment(self, requested):
        for i in range(HORIZON_DAYS + 1):
            d = self.request_date + dt.timedelta(days=i)
            if self.is_safe([(d, requested)]):
                return d
        return None

    def with_changes(self, changes):
        """changes: {event_id: new_amount} applied to projected debits of that stream (0 = stop)."""
        flows = [
            replace(f, amount=-changes[f.event_id]) if f.projected and f.amount < 0 and f.event_id in changes else f
            for f in self.flows
        ]
        return replace(self, flows=flows)


def build_forecast(ds, request, config=ForecastConfig(), facts=()):
    user = request["user_id"]
    profile = ds.profiles[user]
    home = profile["home_currency"]
    rd = parse_date(request["request_date"])
    end = rd + dt.timedelta(days=HORIZON_DAYS)
    fact_amounts = {f["event_id"]: f["amount"] for f in facts if f.get("kind") == "event_amount" and f.get("amount")}

    def amount_of(e):
        raw = e["amount"] or fact_amounts.get(e["event_id"])
        if raw in (None, ""):
            return None
        return ds.to_home(float(raw), e["currency"], e["settlement_date"] or e["event_date"], home)

    events = ds.events_by_user[user]
    flows = []

    # Known future cash: pending/scheduled debits are reserved; only scheduled income counts.
    retried = {e["linked_event_id"] for e in events if e["linked_event_id"] and e["status"] == "scheduled"}
    failed = []  # (category, date) of failed debits with no scheduled retry
    for e in events:
        amount = amount_of(e)
        if amount is None or e["direction"] == "non_cash":
            continue
        d = parse_date(e["settlement_date"] or e["event_date"])
        if e["status"] in ("pending", "scheduled") and e["direction"] == "debit":
            if d <= end:
                flows.append(Flow(max(d, rd), -amount, e["category"], e["description"], e["event_id"]))
        elif e["status"] == "scheduled" and e["direction"] == "credit" and e["event_type"] == "income" and rd <= d <= end:
            flows.append(Flow(d, amount, e["category"], e["description"], e["event_id"]))
            if config.salary_recurs_after_scheduled and e["category"] == "salary":
                k = 1
                while add_months(d, k) <= end:
                    flows.append(Flow(add_months(d, k), amount, e["category"], e["description"], e["event_id"],
                                      projected=True, kind="monthly"))
                    k += 1
        elif e["status"] == "failed" and e["direction"] == "debit" and e["event_id"] not in retried:
            failed.append((e["category"], parse_date(e["event_date"])))
            if config.failed_debit_mode == "reserve" and 0 <= (rd - failed[-1][1]).days <= 14:
                flows.append(Flow(rd, -amount, e["category"], e["description"], e["event_id"]))
    known = list(flows)

    # Recurring streams from settled history.
    history = [
        e for e in events
        if e["status"] == "settled" and e["direction"] in ("debit", "credit")
        and parse_date(e["event_date"]) <= rd and e["event_type"] not in NON_STREAM_TYPES
        and not e["linked_event_id"] and e["event_id"] not in ds.linked_parents
    ]
    by_category = defaultdict(list)
    for e in history:
        by_category[(e["category"], e["direction"])].append(e)

    streams = []
    for (category, direction), group in by_category.items():
        by_description = defaultdict(list)
        for e in group:
            by_description[e["description"]].append(e)
        rest = []
        for description_events in by_description.values():
            dates = [parse_date(e["event_date"]) for e in description_events]
            if category not in VARIABLE_CATEGORIES and len(description_events) >= 2 and _is_monthly(dates):
                streams.append(("monthly", category, direction, description_events))
            else:
                rest.extend(description_events)
        if len(rest) >= 3 and (direction == "debit" or config.project_irregular_income):
            streams.append(("pool", category, direction, sorted(rest, key=lambda e: e["event_date"])))

    for kind, category, direction, stream in streams:
        dates = [parse_date(e["event_date"]) for e in stream]
        values = [v for v in (amount_of(e) for e in stream) if v is not None]
        latest = stream[-1]
        if not values or "final" in latest["description"].lower():
            continue
        sign = -1 if direction == "debit" else 1
        if kind == "monthly":
            if add_months(dates[-1], 1) + dt.timedelta(days=config.monthly_grace_days) < rd:
                continue  # an expected occurrence was missed: the stream has ended
            value = estimate(values, config.fixed_estimator if direction == "debit" else "last")
            upcoming = (add_months(dates[-1], k) for k in range(1, 5))
        else:
            gap = round(statistics.median((b - a).days for a, b in zip(dates, dates[1:])))
            if gap <= 0 or dates[-1] + dt.timedelta(days=gap * config.pool_stale_factor) < rd:
                continue
            value = estimate(values, config.variable_estimator if direction == "debit" else "min")
            upcoming = (dates[-1] + dt.timedelta(days=gap * k) for k in range(1, HORIZON_DAYS // gap + 3))
        min_allowed = float(latest["minimum_allowed_amount"]) if latest["minimum_allowed_amount"] else None
        for nd in upcoming:
            if nd > end:
                break
            if nd < rd or (nd == rd and not config.include_due_today):
                continue
            if any(k.category == category and (k.amount < 0) == (sign < 0) and abs((k.date - nd).days) <= 5 for k in known):
                continue
            if (config.failed_debit_mode == "suppress" and sign < 0
                    and any(c == category and abs((d - nd).days) <= 7 for c, d in failed)):
                continue
            flows.append(Flow(nd, sign * value, category, latest["description"], latest["event_id"],
                              latest["flexibility"], min_allowed, projected=True, kind=kind))

    flows = apply_facts(flows, facts, rd, end)
    flows.sort(key=lambda f: (f.date, f.amount))
    return Forecast(rd, end, float(profile["current_available_balance"]),
                    float(profile["minimum_balance_to_keep"]), flows, config)


def apply_facts(flows, facts, rd, end):
    """Apply structured evidence (home-currency amounts) extracted from messages and images."""

    def salary_credits():
        return [f for f in flows if f.category == "salary" and f.amount > 0]

    def add_monthly_salary(start, amount, label):
        k = 0
        while add_months(start, k) <= end:
            if add_months(start, k) >= rd:
                flows.append(Flow(add_months(start, k), amount, "salary", label, projected=True, kind="evidence"))
            k += 1

    for fact in facts:
        kind = fact.get("kind")
        amount = float(fact["amount"]) if fact.get("amount") else None
        effective = parse_date(fact.get("effective_date"))

        if kind == "salary_confirmed" and amount and effective:
            cutoff = effective - dt.timedelta(days=10)
            flows = [f for f in flows if not (f.category == "salary" and f.amount > 0 and f.date >= cutoff)]
            add_monthly_salary(effective, amount, "confirmed salary (evidence)")
        elif kind == "salary_change" and amount:
            targets = [f for f in salary_credits() if f.date >= (effective or rd)]
            named = [f for f in targets if f.label == fact.get("income_description")]
            for f in named or targets:
                f.amount = amount
            if not targets and effective:
                add_monthly_salary(effective, amount, "changed salary (evidence)")
        elif kind == "salary_date_change" and fact.get("new_date"):
            new_date = parse_date(fact["new_date"])
            nearest = min(salary_credits(), key=lambda f: abs((f.date - new_date).days), default=None)
            if nearest and abs((nearest.date - new_date).days) <= 20:
                shift, start, label = new_date - nearest.date, nearest.date, nearest.label
                for f in salary_credits():
                    if f.label == label and f.date >= start:
                        f.date += shift
        elif kind == "salary_only" and amount:
            credits = salary_credits()
            if credits:
                keep = min(credits, key=lambda f: abs(f.amount - amount)).label
                flows = [f for f in flows if not (f.category == "salary" and f.amount > 0 and f.label != keep)]
                for f in salary_credits():
                    f.amount = amount
        elif kind == "income_stopped":
            flows = [f for f in flows if not (f.amount > 0 and f.date >= (effective or rd))]
        elif kind == "income_unconfirmed":
            description = fact.get("income_description")
            flows = [f for f in flows
                     if not (f.amount > 0 and f.projected and (f.label == description if description else f.kind == "pool"))]
        elif kind == "one_off_credit" and amount:
            upcoming_salary = sorted(f.date for f in salary_credits() if f.date >= rd)
            date = effective or (upcoming_salary[0] if upcoming_salary else rd)
            flows.append(Flow(max(date, rd), amount, "income", "confirmed one-off credit (evidence)", kind="evidence"))
        elif kind == "expense_change":
            category = fact.get("category")
            categories = {"rent", "housing"} if category in ("rent", "housing") else {category}
            for f in flows:
                if f.amount < 0 and f.projected and f.category in categories and f.date >= (effective or rd):
                    if amount:
                        f.amount = -amount
                    elif fact.get("pct"):
                        f.amount *= 1 + float(fact["pct"]) / 100
        elif kind == "one_off_debit" and amount:
            flows.append(Flow(max(effective or rd, rd), -amount, fact.get("category") or "other",
                              "outstanding debit (evidence)", kind="evidence"))
    return [f for f in flows if rd <= f.date <= end]
