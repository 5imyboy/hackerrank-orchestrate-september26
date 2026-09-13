# Buy or Wait? — AI financial decision agent

For every request in `dataset/requests.csv` the agent decides whether the user should pay in full, pay partially, use a supplied installment option, wait, or not proceed, and writes `output.csv` in the repository root.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r code/requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env      # only needed to extract evidence; never commit .env
```

Python 3.10+ is required. Apart from the `anthropic` SDK, only the standard library is used.

## Run

```bash
.venv/bin/python code/main.py --evidence extract   # full run: calls Claude for uncached messages/images, writes evaluation/usage_report.md
.venv/bin/python code/main.py                      # reuse cached evidence (code/cache/evidence.json), no API calls
.venv/bin/python code/main.py --evidence none      # ignore messages and images entirely
.venv/bin/python code/evaluation/main.py           # score against dataset/sample_requests.csv, list mismatches
.venv/bin/python code/evaluation/main.py --grid    # sweep forecast settings against the samples
```

Options: `--requests <csv>` and `--out <csv>` change the input and output paths.

## How it works

| Step | Module | What it does |
|---|---|---|
| Load | `data.py` | Reads all dataset CSVs, indexes by user and request, converts currencies with the dated rates. |
| Evidence | `evidence.py` | Sends each message and image to `claude-opus-5` once, using a JSON schema for structured output. It returns typed facts: salary change, confirmed or moved salary, income stopped or unconfirmed, confirmed one-off credit, rent change, outstanding bill, or the amount of a blank-amount event. Message and image text is treated as untrusted data; embedded instructions are ignored. Results are cached by content hash, so reruns are deterministic. |
| Forecast | `forecast.py` | Builds a 90-day daily balance from the current balance. It projects recurring streams detected from settled history: monthly bills and income by description, and variable groceries, transport and dining by category cadence. It reserves pending and scheduled debits and counts scheduled income. It ignores pending credits, failed or cancelled rows, unrealized investments and lifecycle duplicates, then applies the evidence facts. |
| Decide | `decide.py` | Computes the safe amount today (lowest projected balance minus the minimum to keep) and the earliest safe full-payment date. It builds the eligible plans (full payment, partial payment, supplied installment options, wait), keeps only those where every payment keeps the balance above the minimum, and ranks them: meets the deadline, needs no spending changes, lowest total paid, earlier start, fewer payments, lowest option id. If nothing meets the deadline, it searches up to three permitted stop/reduce changes on flexible, non-protected expenses and picks the smallest sufficient cut. |
| Explain | `explain.py` | Formats plans and amounts and writes a short, grounded explanation. |
| Validate | `validate.py` | Checks every row against the output contract before it is written: bounds, allowed values, plan shape, exact installment match, and permitted spending changes. |

Only messages and images use a model. Everything else is deterministic Python, which keeps token cost low; see `evaluation/usage_report.md`.
