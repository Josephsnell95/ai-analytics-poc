"""Ask the agent questions from the terminal.

    python -m agent "What was total revenue in 2013?"          # Llama via Ollama (default)
    python -m agent -p anthropic "Top 5 products by revenue"   # Claude Haiku
    python -m agent                                             # interactive, type questions
    python -m agent --smoke                                     # run the ten smoke questions
    python -m agent --summary                                   # cost and latency per model so far
    python -m agent --inspect                                   # every query tried for the last question
    python -m agent --inspect 12                                # ...or for logged question 12

Add --sql to print the semantic query and SQL, and --chart out.html to save the chart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from semantic import SemanticEngine

from .charts import to_html
from .config import load_settings
from .log import QuestionLog
from .loop import Agent, AgentResult
from .providers import ProviderError, make_provider

MAX_PRINTED_ROWS = 20


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m agent")
    parser.add_argument("question", nargs="*", help="question to ask (omit for interactive mode)")
    parser.add_argument("-p", "--provider", help="ollama or anthropic (default from config.yaml)")
    parser.add_argument("--sql", action="store_true", help="show the semantic query and generated SQL")
    parser.add_argument("--chart", metavar="FILE", help="save the chart as an HTML file")
    parser.add_argument("--smoke", action="store_true", help="run agent/smoke_questions.yaml")
    parser.add_argument("--summary", action="store_true", help="print per-model totals from the log")
    parser.add_argument("--inspect", nargs="?", const="last", metavar="ID",
                        help="show every query tried for a logged question (default: the last one)")
    args = parser.parse_args()

    settings = load_settings()
    log = QuestionLog(settings.path("app_state"))
    if args.summary:
        print_summary(log)
        return 0
    if args.inspect:
        return inspect(log, args.inspect)

    provider_name = args.provider or settings.agent["default_provider"]
    try:
        provider = make_provider(provider_name, settings)
    except ProviderError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    engine = SemanticEngine.from_config(settings.root / "config.yaml")
    agent = Agent(provider, engine, settings)
    print(f"Using {provider.name} / {provider.model}\n")

    if args.smoke:
        return run_smoke(agent, log, settings.root / "agent" / "smoke_questions.yaml", args)
    if args.question:
        result = ask(agent, log, " ".join(args.question), args)
        return 0 if result.status != "error" else 1

    print("Type a question, 'new' to start a fresh conversation, or 'quit' to exit.")
    history: list[AgentResult] = []
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.lower() in ("quit", "exit", "q"):
            return 0
        if q.lower() == "new":
            history = []
            print("Started a new conversation.")
        elif q:
            history.append(ask(agent, log, q, args, history=history))


def ask(agent: Agent, log: QuestionLog, question: str, args, chart_path: str | None = None,
        history: list[AgentResult] | None = None) -> AgentResult:
    print("Thinking...", flush=True)
    result = agent.ask(question, history=history)
    log.write(result)
    print_result(result, show_sql=args.sql)
    path = chart_path or args.chart
    if path and result.chart:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(to_html(result.chart, question), encoding="utf-8")
        print(f"Chart saved to {path}")
    return result


def print_result(r: AgentResult, show_sql: bool = False) -> None:
    if r.assumptions:
        print("Assumed: " + "; ".join(r.assumptions))
    print()
    print(r.answer or "(no answer)")
    if r.unverified:
        print(f"\n!! Not in the query results: {', '.join(r.unverified)}. Check the table below.")
    print()
    if r.rows:
        print_table(r.columns, r.rows, r.column_formats)
    if r.chart:
        traces = r.chart["data"]
        who = "default, the agent didn't ask for one" if r.chart_auto else "requested by the agent"
        print(f"Chart ({who}): {r.chart['layout']['meta']['chart_type']} with {len(traces)} series - "
              f"\"{r.chart['layout']['title']['text']}\"")
    if show_sql and r.semantic_query:
        print("\nSemantic query:")
        print(json.dumps(r.semantic_query, indent=2))
        print("\nSQL:")
        print(r.sql)
    failed = sum(1 for a in r.attempts if a.error)
    u = r.usage
    status = f"{r.status} by {r.declined_by}" if r.declined_by else r.status
    print(f"\n[{status}] {r.model} | {u.llm_calls} model calls, {u.tool_calls} tool calls"
          f"{f' ({failed} rejected queries)' if failed else ''} | "
          f"{u.input_tokens:,} in / {u.output_tokens:,} out tokens | ${u.cost_usd:.4f} | {u.latency_s:.1f}s")


def print_table(columns: list[str], rows: list[dict], formats: dict | None = None) -> None:
    shown = rows[:MAX_PRINTED_ROWS]
    formats = formats or {}
    cells = [[_fmt(row[c], formats.get(c)) for c in columns] for row in shown]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for r in cells:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)))
    if len(rows) > len(shown):
        print(f"... {len(rows) - len(shown)} more rows")


def run_smoke(agent: Agent, log: QuestionLog, path: Path, args) -> int:
    with open(path, encoding="utf-8") as f:
        questions = yaml.safe_load(f)
    charts_dir = Path(args.chart) if args.chart else None
    summary = []
    for i, item in enumerate(questions, 1):
        print("=" * 78)
        print(f"Q{i}: {item['question']}")
        print(f"    tests: {item['tests']}")
        chart_path = str(charts_dir / f"q{i:02d}.html") if charts_dir else None
        r = ask(agent, log, item["question"], args, chart_path=chart_path)
        print(f"\nEXPECTED: {item['expected']}")
        summary.append((i, r))

    print("=" * 78)
    print("\nSummary (check each answer against EXPECTED above):")
    total_cost = sum(r.usage.cost_usd for _, r in summary)
    total_time = sum(r.usage.latency_s for _, r in summary)
    for i, r in summary:
        print(f"  Q{i:<3} {r.status:<9} {r.usage.tool_calls} tools  {r.usage.latency_s:>6.1f}s  ${r.usage.cost_usd:.4f}")
    print(f"  Total: ${total_cost:.4f}, {total_time:.0f}s, "
          f"average ${total_cost / len(summary):.4f} and {total_time / len(summary):.1f}s per question")
    return 0


def print_summary(log: QuestionLog) -> None:
    rows = log.summary()
    if not rows:
        print("No questions logged yet.")
        return
    header = ["provider", "model", "questions", "answered", "declined", "avg s", "avg tokens", "total $", "avg $"]
    print_table(header, [dict(zip(header, r)) for r in rows])


def inspect(log: QuestionLog, which: str) -> int:
    row = log.get(None if which == "last" else int(which))
    if row is None:
        print("No such question in the log.")
        return 1
    print(f"#{row['id']} {row['asked_at']} {row['provider']}/{row['model']} [{row['status']}]")
    print(f"Q: {row['question']}\n")
    for i, a in enumerate(json.loads(row["attempts"]), 1):
        print(f"Query {i}: {json.dumps(a['query'])}")
        if a["error"]:
            print("  REJECTED: " + a["error"].replace("\n", "\n            "))
        else:
            print(f"  ran, {a['row_count']} rows")
    print(f"\nAnswer: {row['answer']}")
    print(f"\n{row['llm_calls']} model calls, {row['tool_calls']} tool calls, {row['latency_s']}s, "
          f"${row['cost_usd']:.4f}")
    if row["error"]:
        print(f"Error: {row['error']}")
    return 0


def _fmt(v, fmt: str | None = None) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if fmt == "currency":
            return f"${v:,.2f}"
        if fmt == "percent":
            return f"{v:.1%}"
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return str(v)


if __name__ == "__main__":
    sys.exit(main())
