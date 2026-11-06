# AI Analytics PoC

Ask a question in plain English, get a chart back, and be able to check how the answer was produced.

The AI never writes SQL. It emits a structured **semantic query** naming metrics and dimensions from a governed model, and a deterministic compiler turns that into SQL. That constraint is the point: answers use defined business logic, every answer shows the query and SQL behind it, and accuracy is measured with an evaluation set rather than asserted.

Runs entirely on a laptop against Microsoft's public AdventureWorksDW sample (Internet Sales). The default model is Llama 3.1 8B running locally through Ollama, free; Claude Haiku is a one-flag switch for comparison.

## Status

| Milestone | State |
| --- | --- |
| 1. Data load into DuckDB | Done |
| 2. Semantic model + compiler, no LLM | Done |
| 3. Agent loop as a CLI | Done: 10/10 smoke questions on both models (see below) |
| 4. Streamlit chat | Built; needs a run-through by someone other than the author |
| 5. Evaluation harness | |
| 6. Saved gallery, sharing between demo users, correction log | |

## Quick start

Needs Python 3.11+. For the local model, install [Ollama](https://ollama.com) and run `ollama pull llama3.1:8b` (about 5 GB). For Claude, copy `.env.example` to `.env` and add an Anthropic API key.

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\activate                 # if scripts are blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python data/load.py --verify            # downloads the sample (~17 MB) and builds data/adventureworks.duckdb
pytest                                  # 232 tests, a few seconds
streamlit run app/main.py               # opens the chat in your browser
```

macOS / Linux: the same, with `python3 -m venv .venv` and `source .venv/bin/activate`.

## The chat app

```powershell
streamlit run app/main.py
```

Run it from the project folder. It opens at http://localhost:8501; stop it with Ctrl+C in the terminal.

- **Pick the model in the sidebar.** Llama runs locally and is free but takes a minute or two per answer on a laptop CPU; Claude Haiku takes a few seconds and costs about a tenth of a cent. The sidebar says if a model isn't available (Ollama not running, or no API key in `.env`), and the chat box is disabled until it is.
- **Each answer shows** any assumption the model made, the written answer, a chart for breakdowns, the data behind it, and a footer with model calls, tokens, cost and time. **How this was answered** shows the semantic query, the generated SQL, and any query the semantic layer rejected along the way, with the reason.
- **While it works**, the status line shows what the agent is doing: running a query, fixing a rejected one, drawing a chart.
- **Follow-ups work.** The agent sees the last two questions in the chat, each with its answer and the semantic query behind it (not the data rows), so "now by month" or "plot those side by side" builds on the previous query. It always re-queries rather than reusing old numbers. **New chat** starts fresh. `history_turns` in `config.yaml` sets how many turns are passed.
- **Figures are checked against the data.** Every number in the written answer is matched against the query result at the precision it was written (rounded or truncated). Anything that doesn't match is flagged above the chart: in testing, Llama reported a $11,505,796.50 margin as $11,506,197. Flagged figures are logged too (`unverified` in `question_log`).
- Every question is logged to `data/app_state.sqlite`, the same log the CLI uses.

## Asking questions from the terminal

```powershell
python -m agent "Show me sales by product category for 2013"          # Llama, local
python -m agent -p anthropic "Show me sales by product category for 2013"   # Claude Haiku
python -m agent                                     # interactive, with follow-ups: 'new' resets, 'quit' exits
python -m agent --sql "Top 3 product models by revenue"   # also print the semantic query and SQL
python -m agent --chart charts/out.html "Monthly revenue in 2013"   # save the chart, open in a browser
python -m agent --smoke --chart charts              # the ten smoke-test questions, with expected answers
python -m agent --summary                           # questions, cost and latency per model so far
python -m agent --inspect                           # every query tried for the last question, and why any were rejected
```

Each answer prints any assumption the agent made, the written answer, the result table, and a line like:

```
[answered] llama3.1:8b | 3 model calls, 2 tool calls | 6,412 in / 188 out tokens | $0.0000 | 41.3s
```

Every question is logged to `data/app_state.sqlite` (table `question_log`) with tokens, cost, latency, every query the agent tried, and the full transcript, so a wrong answer can be traced to the step that went wrong.

Expect Llama on a laptop CPU to take 30 to 90 seconds per question: each model call re-reads a prompt of around 2,000 tokens. Haiku answers in a few seconds for well under a cent.

## Running semantic queries by hand

To check the compiler without any LLM, pass a query file (inline JSON gets mangled by Windows shell quoting):

```powershell
python -m semantic examples/revenue_by_year.json
python -m semantic --sql-only examples/top_models_2013.json
```

## How it fits together

```
question ─▶ agent ─▶ semantic query (JSON) ─▶ compiler ─▶ SQL ─▶ DuckDB
               ▲             │                    │
               │             └─ readable errors ◀─┘
               └─ rows ◀──────────────────────────────────────────┘
```

| Path | What it is |
| --- | --- |
| `data/load.py` | Rebuilds the DuckDB file from Microsoft's install script. Column names and types are read from their `CREATE TABLE` statements rather than hand-typed. |
| `model/semantic_model.yaml` | Tables, joins, dimensions, measures, descriptions, synonyms and context for the AI. The only place business definitions live. |
| `semantic/` | Model loader, query validation, SQL compiler and DuckDB runner. No LLM involvement. |
| `agent/providers.py` | One interface over Ollama and the Anthropic API, using plain HTTP so the translation is all in one file. |
| `agent/tools.py` | The three tools: `get_model`, `run_query`, `make_chart`. |
| `agent/loop.py` | Ask, run tools, repeat until a text answer or the tool-call cap. |
| `agent/charts.py` | Turns a chart request into a Plotly JSON spec. No generated plotting code. |
| `agent/verify.py` | Checks the figures in a written answer against the query result. |
| `agent/log.py` | SQLite question log. |
| `app/main.py`, `app/ui.py` | Streamlit chat: the page, and how an answer is rendered. |
| `config.yaml` | Models, row limits, tool-call cap, paths. |
| `agent/prices.yaml` | Per-token prices. |

## The agent

One agent, three tools, a cap of six tool calls per question.

- **The model description is in the system prompt**, not only behind `get_model`. That saves a model round trip per question (10 to 20 seconds on CPU) and stops small models guessing field names because they skipped the lookup. `get_model` stays available as a keyword search.
- **Compiler errors go back to the model as text** listing valid alternatives, so it can fix the query and retry within the same question.
- **The model sees at most 50 rows** (configurable), with numbers rounded. The app and the log keep every row.
- **Small formatting slips are repaired before validation**: a list sent as a JSON string, a query wrapped in a `query` key, a tool call written as JSON text instead of through the tool channel. These are Llama tool-calling quirks rather than wrong answers. Wrong field names are never repaired; the compiler rejects them.
- **If the agent doesn't ask for a chart, the code draws one** for any multi-row breakdown: a line over time, otherwise a bar, with every measure in the same unit side by side (revenue and margin as clustered columns). The app says when a chart was a default.
- **Known gaps are declined in code.** `not_available` in the model YAML lists topics the model can't answer (reseller sales, employees, targets, inventory, fiscal periods, forecasts), with the words that identify them and a reason. A question mentioning one is declined before the LLM sees it, and logged as `declined` by `guard`. In one run Llama declined "reseller sales"; in the next it answered with internet revenue, so this can't be left to the prompt. The limit: it only catches gaps someone has written down. Other out-of-scope questions rely on the model declining, logged as declined by `model`.
- **Statuses**: `answered` (backed by at least one successful query), `declined`, `no_query` (answered without querying, which is suspect), `capped`, `error`. `answered` does not mean correct; scoring is the evaluation harness's job.

## The semantic query

```json
{
  "measures": ["revenue", "order_count"],
  "dimensions": ["product_category", {"field": "order_date", "grain": "month"}],
  "filters": [
    {"field": "order_date", "op": "between", "value": ["2013-01-01", "2013-12-31"]},
    {"field": "revenue", "op": ">", "value": 10000}
  ],
  "sort": [{"field": "revenue", "dir": "desc"}],
  "limit": 100,
  "assumptions": ["Interpreted 'last year' as calendar 2013, the latest full year in the data"]
}
```

Dates in filters can be a day (`2013-01-31`) or a whole period (`2013`, `2013-Q2`, `2013-03`); `= "2013"` means the whole year and `between ["2012", "2013"]` covers both years. Time dimensions can be written `order_date__month` or `{"field": "order_date", "grain": "month"}`.

Filter ops depend on the field: `=`, `!=`, `in`, `not_in`, `contains`, `is_null`, `is_not_null` for text; comparisons and `between` for numbers, dates and measures. Time dimensions take a grain of `day`, `week`, `month`, `quarter` or `year`, and come back as a column named like `order_date__month`.

## Compiler rules, and why

- **Unknown fields are rejected with the closest valid names**, and a synonym gets pointed at the real field. Every problem comes back in one message so the agent can fix them in one retry.
- **Only the tables a query needs are joined**, walking many-to-one paths out from the fact table. The model can only contain many-to-one joins away from the single fact, so a query can never fan out and double count.
- **LEFT JOINs.** A missing dimension row should show as a blank group, not silently remove sales from the total.
- **Dimension filters go in WHERE, measure filters in HAVING.**
- **Two `=`, `between` or `in` filters on the same field are rejected**, because filters are combined with AND and they can never both match. The error suggests one `between` or `in` instead.
- **Time grains compile to `DATE_TRUNC`.** Date filters always apply to the underlying date, not the truncated one.
- **Derived measures expand into their base aggregates**, so `gross_margin_pct` means the same thing in every query.
- **Sums only on the fact table.** The model refuses a `sum` or `avg` defined on a dimension table, because at fact grain it would be repeated per order line. `count_distinct` is allowed because it is the one aggregation that stays correct.
- **Filter values are bound parameters**, never pasted into SQL. That matters more than usual here because the values come from an LLM. The SQL shown to users has values inlined for readability; the SQL that runs does not.
- **Every query has a limit and a full sort order**, so a limited result is the same every time.

## Notes on the data

- 60,398 order lines, $29,358,677.22 revenue, orders from 2010-12-29 to 2014-01-28. 2013 is the latest full year; 2010 and 2014 are a few days and one month.
- All amounts are USD regardless of the order's currency.
- No Components were sold online, so only three of the four categories appear in results.
- **Fiscal year is deliberately not modelled.** `DimDate.FiscalYear` in this version of the sample is inconsistent (July–December 2010 is FY2011 but January–June 2011 is FY2010), most likely a side effect of Microsoft shifting the sample's dates forward. Exposing it would give confidently wrong answers, so the agent is told to say fiscal periods aren't available.

## Out of scope for v1

Reseller Sales (a second fact at a different grain, which needs fan/chasm-trap handling), letting the agent propose new metrics, dashboards, multiple agents, logins and deployment. Each is in the handoff doc with the reason it was left out.
