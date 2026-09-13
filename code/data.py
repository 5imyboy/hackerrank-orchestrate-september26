"""Load the Buy or Wait? dataset CSVs into per-user and per-request indexes."""

import csv
import datetime as dt
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"


def parse_date(value):
    return dt.date.fromisoformat(value) if value else None


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Dataset:
    def __init__(self, dataset_dir=DATASET_DIR):
        self.dir = Path(dataset_dir)
        self.profiles = {r["user_id"]: r for r in read_csv(self.dir / "financial_profiles.csv")}

        self.events = read_csv(self.dir / "financial_events.csv")
        self.events_by_id = {e["event_id"]: e for e in self.events}
        self.events_by_user = defaultdict(list)
        for e in self.events:
            self.events_by_user[e["user_id"]].append(e)
        for events in self.events_by_user.values():
            events.sort(key=lambda e: (e["event_date"], e["event_id"]))
        self.linked_parents = {e["linked_event_id"] for e in self.events if e["linked_event_id"]}

        self.fx = {
            (r["rate_date"], r["from_currency"], r["to_currency"]): float(r["rate"])
            for r in read_csv(self.dir / "exchange_rates.csv")
        }

        self.options_by_request = defaultdict(list)
        for o in read_csv(self.dir / "request_payment_options.csv"):
            self.options_by_request[o["request_id"]].append(o)

        self.messages = read_csv(self.dir / "messages.csv")
        self.messages_by_user = defaultdict(list)
        for m in self.messages:
            self.messages_by_user[m["user_id"]].append(m)

        self.images = read_csv(self.dir / "images.csv")
        self.images_by_user = defaultdict(list)
        for i in self.images:
            self.images_by_user[i["user_id"]].append(i)

    def load_requests(self, path):
        return read_csv(path)

    def image_path(self, image_id):
        return self.dir / "media" / "images" / f"{image_id}.png"

    def to_home(self, amount, currency, date, home):
        """Convert using the rate dated `date` (YYYY-MM-DD) in either stated direction."""
        if currency == home:
            return amount
        if (date, currency, home) in self.fx:
            return amount * self.fx[(date, currency, home)]
        if (date, home, currency) in self.fx:
            return amount / self.fx[(date, home, currency)]
        raise KeyError(f"no exchange rate for {currency}->{home} on {date}")

    def to_home_nearest(self, amount, currency, date, home):
        """Convert with the latest rate dated on or before `date` (the earliest rate if none precede it)."""
        if currency == home:
            return amount
        dates = sorted(d for d, src, dst in self.fx if {src, dst} == {currency, home})
        if not dates:
            raise KeyError(f"no exchange rate between {currency} and {home}")
        prior = [d for d in dates if d <= date]
        return self.to_home(amount, currency, prior[-1] if prior else dates[0], home)
