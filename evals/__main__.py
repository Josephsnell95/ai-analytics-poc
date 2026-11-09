"""Run the evaluation.

    python -m evals check                      # validate the questions and show expected answers
    python -m evals run                        # tuning set on the default model (Llama: about an hour)
    python -m evals run -p anthropic           # ...on Claude Haiku (a few minutes, a few pence)
    python -m evals run --set heldout          # the set the reported accuracy comes from
    python -m evals run --only ranking_top_promotion,ambiguous_last_year
    python -m evals run --resume evals/results/<file>.json   # carry on after an interruption
    python -m evals report                     # results table, written to evals/results/summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.config import load_settings
from agent.loop import Agent
from agent.providers import ProviderError, make_provider, provider_status
from semantic import SemanticEngine

from .runner import (RESULTS_DIR, SETS, check_questions, latest_results, load_questions, needs_reference,
                     report, results_path, run_set)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m evals")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="validate questions and print expected answers")
    p_check.add_argument("--set", choices=SETS, default="tuning")

    p_run = sub.add_parser("run", help="run a question set against a model")
    p_run.add_argument("--set", choices=SETS, default="tuning")
    p_run.add_argument("-p", "--provider", help="ollama or anthropic (default from config.yaml)")
    p_run.add_argument("--only", help="comma-separated question ids")
    p_run.add_argument("--resume", metavar="FILE", help="continue an interrupted run")

    sub.add_parser("report", help="summarise the latest run per model")
    args = parser.parse_args()

    settings = load_settings()
    engine = SemanticEngine.from_config(settings.root / "config.yaml")

    if args.command == "check":
        return check(args.set, engine)
    if args.command == "report":
        text = report(latest_results())
        (RESULTS_DIR / "summary.md").parent.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "summary.md").write_text(text, encoding="utf-8")
        print(text)
        print(f"Written to {RESULTS_DIR / 'summary.md'}")
        return 0

    all_items = load_questions(args.set)
    problems = check_questions(all_items, engine)
    if problems:
        print("Fix these first (python -m evals check):\n- " + "\n- ".join(problems), file=sys.stderr)
        return 1
    items = all_items
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {i["id"] for i in all_items}
        if unknown:
            print(f"Unknown ids: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 1
        items = [i for i in all_items if i["id"] in wanted]

    provider_name = args.provider or settings.agent["default_provider"]
    available, why = provider_status(provider_name, settings)
    if not available:
        print(f"Error: {why}", file=sys.stderr)
        return 1
    try:
        provider = make_provider(provider_name, settings)
    except ProviderError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    agent = Agent(provider, engine, settings)

    resume = None
    if args.resume:
        out = Path(args.resume)
        resume = json.loads(out.read_text(encoding="utf-8"))
    else:
        out = results_path(args.set, provider.model)
    ready = [i for i in items if not needs_reference(i)]
    print(f"{args.set} set, {provider.name} / {provider.model}: {len(ready)} questions"
          f"{f', {len(items) - len(ready)} awaiting a reference' if len(items) > len(ready) else ''}")
    print(f"Writing {out}\n")

    data = run_set(agent, items, out, resume=resume, all_items=all_items)
    s = data["summary"]
    print(f"\n{s['passed']}/{s['questions']} passed ({s['accuracy']:.0%}) · avg {s['avg_latency_s']}s · "
          f"${s['total_cost_usd']:.4f} total")
    print("Run 'python -m evals report' for the full table.")
    return 0


def check(set_name: str, engine: SemanticEngine) -> int:
    items = load_questions(set_name)
    problems = check_questions(items, engine)
    for item in items:
        print(f"{item['id']}  [{item['category']}]  {item['question']}")
        if item.get("follow_up_of"):
            print(f"    follows: {item['follow_up_of']}")
        if item.get("expect") == "decline":
            gap = engine.model.unavailable_topic(item["question"])
            print(f"    expect: decline ({'guard: ' + gap.topic if gap else 'model must decline'})")
        elif needs_reference(item):
            print("    expect: AWAITING REFERENCE")
        else:
            try:
                rows = engine.run(item["reference"]).records()
                print(f"    expect: {rows[:6]}{' ...' if len(rows) > 6 else ''}")
            except Exception as e:  # reported in problems below
                print(f"    expect: ERROR {e}")
    print(f"\n{len(items)} questions, {sum(needs_reference(i) for i in items)} awaiting a reference")
    if problems:
        print("\nProblems:\n- " + "\n- ".join(problems))
        return 1
    print("No problems.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
