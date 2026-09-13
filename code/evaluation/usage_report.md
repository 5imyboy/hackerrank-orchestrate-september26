# Token Usage Report

Final full-dataset run: `python3 code/main.py --evidence extract` on 2026-09-12, producing `output.csv` for 250 requests.

Only the unstructured evidence (messages and images) is sent to a model. Forecasting, plan selection, explanations and validation are deterministic Python and use no tokens.

## Per model

| Provider / model | Calls | Input tokens | Output tokens | Cache read | Cache write | Total tokens | Est. cost (USD) |
|---|---|---|---|---|---|---|---|
| Anthropic / `claude-haiku-4-5-20251001` | 13 | 67,188 | 14,359 | 0 | 0 | 81,547 | $0.1390 |
| **Overall** | 13 | 67,188 | 14,359 | 0 | 0 | 81,547 | $0.1390 |

## Summary

- Evidence items (messages + images) for the evaluated users: 209
- Model calls: 13
- Input tokens: 67,188; output tokens (includes thinking): 14,359
- Total tokens: 81,547
- Average tokens per request: 326.2
- Estimated total cost: $0.1390
- Estimated cost per request: $0.00056

Pricing: Anthropic first-party list prices per 1M tokens — `claude-haiku-4-5` $1.00 input / $5.00 output; cache reads 0.1x and cache writes 1.25x the input price.
