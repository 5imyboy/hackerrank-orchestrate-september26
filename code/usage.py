"""Record Claude API token usage and write the token usage report."""

import datetime as dt
import threading
from collections import defaultdict

# Anthropic first-party list prices, USD per 1M tokens: (input, output).
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
}
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

_lock = threading.Lock()
_calls = []


def record(model, item_type, item_id, response_usage):
    with _lock:
        _calls.append({
            "model": model,
            "item": f"{item_type}:{item_id}",
            "input": response_usage.input_tokens or 0,
            "output": response_usage.output_tokens or 0,
            "cache_read": getattr(response_usage, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(response_usage, "cache_creation_input_tokens", 0) or 0,
        })


def _cost(model, c):
    price_in, price_out = PRICING.get(model, PRICING["claude-opus-5"])
    return (c["input"] * price_in + c["output"] * price_out
            + c["cache_read"] * price_in * CACHE_READ_MULTIPLIER
            + c["cache_write"] * price_in * CACHE_WRITE_MULTIPLIER) / 1_000_000


def write_report(path, n_requests, n_items, command):
    per_model = defaultdict(lambda: defaultdict(float))
    for c in _calls:
        m = per_model[c["model"]]
        m["calls"] += 1
        for k in ("input", "output", "cache_read", "cache_write"):
            m[k] += c[k]
        m["cost"] += _cost(c["model"], c)

    overall = defaultdict(float)
    for m in per_model.values():
        for k, v in m.items():
            overall[k] += v
    total_tokens = overall["input"] + overall["output"] + overall["cache_read"] + overall["cache_write"]

    def row(name, m):
        tokens = m["input"] + m["output"] + m["cache_read"] + m["cache_write"]
        return (f"| {name} | {int(m['calls'])} | {int(m['input']):,} | {int(m['output']):,} | "
                f"{int(m['cache_read']):,} | {int(m['cache_write']):,} | {int(tokens):,} | ${m['cost']:.4f} |")

    lines = [
        "# Token Usage Report",
        "",
        f"Final full-dataset run: `{command}` on {dt.date.today().isoformat()}, producing `output.csv` for "
        f"{n_requests} requests.",
        "",
        "Only the unstructured evidence (messages and images) is sent to a model. Forecasting, plan selection, "
        "explanations and validation are deterministic Python and use no tokens.",
        "",
        "## Per model",
        "",
        "| Provider / model | Calls | Input tokens | Output tokens | Cache read | Cache write | Total tokens | Est. cost (USD) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [row(f"Anthropic / `{model}`", m) for model, m in sorted(per_model.items())]
    lines += [row("**Overall**", overall), ""]
    lines += [
        "## Summary",
        "",
        f"- Evidence items (messages + images) for the evaluated users: {n_items}",
        f"- Model calls: {int(overall['calls'])}",
        f"- Input tokens: {int(overall['input']):,}; output tokens (includes thinking): {int(overall['output']):,}",
        f"- Total tokens: {int(total_tokens):,}",
        f"- Average tokens per request: {total_tokens / max(n_requests, 1):,.1f}",
        f"- Estimated total cost: ${overall['cost']:.4f}",
        f"- Estimated cost per request: ${overall['cost'] / max(n_requests, 1):.5f}",
        "",
        "Pricing: Anthropic first-party list prices per 1M tokens — "
        + "; ".join(f"`{m}` ${p[0]:.2f} input / ${p[1]:.2f} output" for m, p in PRICING.items())
        + f"; cache reads {CACHE_READ_MULTIPLIER}x and cache writes {CACHE_WRITE_MULTIPLIER}x the input price.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
