"""Load question sets, run them through the agent, score them, write results.

A results file is written after every question, so a long Llama run that is
interrupted can carry on with --resume instead of starting again.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from agent.loop import Agent, AgentResult
from semantic import QueryValidationError, SemanticEngine

from .scoring import PASSING, Score, score

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"
MAX_ERRORS_IN_A_ROW = 3   # the model has gone away; stop rather than score the rest as errors
SETS = {"tuning": EVALS_DIR / "questions.yaml", "heldout": EVALS_DIR / "heldout.yaml"}


# ----------------------------------------------------------------- questions
def load_questions(set_name: str) -> list[dict]:
    with open(SETS[set_name], encoding="utf-8") as f:
        items = yaml.safe_load(f) or []
    for item in items:
        item["set"] = set_name
    return items


def check_questions(items: list[dict], engine: SemanticEngine) -> list[str]:
    """Problems with a question file. Questions still waiting for a reference are
    not problems; they're reported separately and skipped by the runner."""
    problems, seen = [], set()
    for item in items:
        qid = item.get("id")
        where = f"{qid or '(no id)'}"
        if not qid:
            problems.append(f"{where}: missing id")
        elif qid in seen:
            problems.append(f"{where}: duplicate id")
        if not item.get("question"):
            problems.append(f"{where}: missing question")
        parent = item.get("follow_up_of")
        if parent and parent not in seen:
            problems.append(f"{where}: follow_up_of '{parent}' must be an earlier question in the file")
        if item.get("expect") == "decline":
            if item.get("reference"):
                problems.append(f"{where}: a decline question takes no reference")
        else:
            for ref in [item.get("reference"), *item.get("alternatives", [])]:
                if ref is None:
                    continue
                try:
                    if not engine.run(ref).rows:
                        problems.append(f"{where}: reference returns no rows")
                except QueryValidationError as e:
                    problems.append(f"{where}: reference is invalid: {'; '.join(e.errors)}")
        seen.add(qid)
    return problems


def needs_reference(item: dict) -> bool:
    return item.get("expect") != "decline" and not item.get("reference")


# ----------------------------------------------------------------------- run
def run_set(agent: Agent, items: list[dict], out_path: Path, *, resume: dict | None = None,
            all_items: list[dict] | None = None, progress: Callable[[str], None] = print) -> dict:
    """Run and score every question in items that is ready, writing out_path as it
    goes. all_items is the whole set, used to find a follow-up's parent question
    when the parent isn't being scored in this run (--only, or a resumed run)."""
    data = resume or {"meta": run_meta(agent, all_items or items), "results": []}
    done = {r["id"] for r in data["results"]}
    by_id = {i["id"]: i for i in (all_items or items)}
    answers: dict[str, AgentResult] = {}   # this run's results, as history for follow-ups

    ready = [i for i in items if not needs_reference(i)]
    errors_in_a_row = 0
    for n, item in enumerate(ready, 1):
        if item["id"] in done:
            continue
        history = []
        parent_id = item.get("follow_up_of")
        if parent_id:
            if parent_id not in answers:   # resumed run, or --only: ask the parent first, unscored
                answers[parent_id] = agent.ask(by_id[parent_id]["question"])
            history = [answers[parent_id]]

        result = agent.ask(item["question"], history=history)
        answers[item["id"]] = result
        s = score(item, result, agent.engine)
        data["results"].append(record(item, result, s))
        _write(out_path, data)
        errors_in_a_row = errors_in_a_row + 1 if s.outcome == "error" else 0
        if errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
            progress(f"Stopping: {errors_in_a_row} provider errors in a row ({result.error}). "
                     f"Fix it, then carry on with --resume {out_path}")
            data["summary"] = summarise(data["results"])
            _write(out_path, data)
            return data
        mark = "pass" if s.passed else "FAIL"
        progress(f"[{n}/{len(ready)}] {mark} {item['id']}: {s.outcome}"
                 f"{' - ' + s.reason if s.reason else ''} ({result.usage.latency_s:.0f}s)")

    data["meta"]["finished"] = _now()
    data["summary"] = summarise(data["results"])
    _write(out_path, data)
    return data


def record(item: dict, r: AgentResult, s: Score) -> dict:
    u = r.usage
    return {
        "id": item["id"], "set": item["set"], "category": item["category"], "question": item["question"],
        "follow_up_of": item.get("follow_up_of"),
        "outcome": s.outcome, "passed": s.passed, "reason": s.reason, "review": s.review,
        "status": r.status, "declined_by": r.declined_by, "answer": r.answer,
        "assumptions": r.assumptions, "semantic_query": r.semantic_query, "unverified": r.unverified,
        "row_count": len(r.rows), "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
        "cost_usd": u.cost_usd, "latency_s": u.latency_s, "llm_calls": u.llm_calls, "tool_calls": u.tool_calls,
        "rejected_queries": sum(1 for a in r.attempts if a.error), "error": r.error,
        "attempts": [{"query": a.query, "error": a.error, "row_count": a.row_count} for a in r.attempts],
    }


def run_meta(agent: Agent, items: list[dict]) -> dict:
    """Everything needed to say which configuration produced a score."""
    model_file = agent.settings.path("model")
    return {
        "provider": agent.provider.name, "model": agent.provider.model,
        "set": items[0]["set"] if items else "", "questions": len(items),
        "awaiting_reference": [i["id"] for i in items if needs_reference(i)],
        "started": _now(), "finished": None, "git_commit": _git_commit(),
        "prompt_sha": _sha(agent.system), "semantic_model_sha": _sha(model_file.read_text(encoding="utf-8")),
        "config": {k: agent.settings.agent[k] for k in ("max_tool_calls", "rows_to_llm", "history_turns")
                   if k in agent.settings.agent},
    }


# ------------------------------------------------------------------- summary
def summarise(results: list[dict]) -> dict:
    n = len(results)
    if not n:
        return {"questions": 0}
    passed = sum(r["passed"] for r in results)
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        by_cat[r["category"]][0] += r["passed"]
        by_cat[r["category"]][1] += 1
    total_cost = sum(r["cost_usd"] for r in results)
    return {
        "questions": n,
        "passed": passed,
        "accuracy": round(passed / n, 3),
        "outcomes": dict(Counter(r["outcome"] for r in results)),
        "by_category": {c: {"passed": p, "of": t} for c, (p, t) in sorted(by_cat.items())},
        "declined_by": dict(Counter(r["declined_by"] for r in results if r["declined_by"])),
        "flagged_figure_answers": sum(1 for r in results if r["unverified"]),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / n, 1),
        "total_cost_usd": round(total_cost, 6),
        "cost_per_question_usd": round(total_cost / n, 6),
        "avg_tool_calls": round(sum(r["tool_calls"] for r in results) / n, 2),
        "rejected_queries": sum(r["rejected_queries"] for r in results),
    }


# -------------------------------------------------------------------- report
def latest_results(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """The most recent finished run for each (set, provider, model)."""
    latest: dict[tuple, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        if not meta.get("finished"):
            continue
        key = (meta["set"], meta["provider"], meta["model"])
        if key not in latest or meta["finished"] > latest[key]["meta"]["finished"]:
            data["path"] = path.name
            latest[key] = data
    return sorted(latest.values(), key=lambda d: (d["meta"]["set"] != "heldout", d["meta"]["model"]))


def report(runs: list[dict]) -> str:
    if not runs:
        return "No finished runs yet. Run: python -m evals run"
    lines = ["# Evaluation results", ""]
    lines += ["| Set | Model | Accuracy | Passed | Avg time | Cost per question | Rejected queries | Answers with flagged figures |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for d in runs:
        m, s = d["meta"], d["summary"]
        cost = "free" if s["total_cost_usd"] == 0 else f"${s['cost_per_question_usd']:.4f}"
        lines.append(f"| {m['set']} | {m['model']} | {s['accuracy']:.0%} | {s['passed']}/{s['questions']} | "
                     f"{s['avg_latency_s']:.0f}s | {cost} | {s['rejected_queries']} | {s['flagged_figure_answers']} |")

    categories = sorted({c for d in runs for c in d["summary"]["by_category"]})
    lines += ["", "## By category", "", "| Category | " + " | ".join(f"{d['meta']['model']} ({d['meta']['set']})" for d in runs) + " |",
              "| --- |" + " --- |" * len(runs)]
    for c in categories:
        cells = []
        for d in runs:
            v = d["summary"]["by_category"].get(c)
            cells.append(f"{v['passed']}/{v['of']}" if v else "")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    for d in runs:
        m = d["meta"]
        failures = [r for r in d["results"] if not r["passed"]]
        reviews = [r for r in d["results"] if r["review"]]
        lines += ["", f"## {m['model']}, {m['set']} set", "",
                  f"Run {m['started'][:16].replace('T', ' ')} UTC · commit `{m.get('git_commit') or 'n/a'}` · "
                  f"prompt `{m['prompt_sha']}` · model `{m['semantic_model_sha']}` · file `{d['path']}`"]
        if m.get("awaiting_reference"):
            lines.append(f"\nNot run, awaiting a reference query: {', '.join(m['awaiting_reference'])}")
        if failures:
            lines += ["", "Failures:", ""]
            lines += [f"- **{r['id']}** ({r['outcome']}): {r['reason'] or r['status']}" for r in failures]
        if reviews:
            lines += ["", "To review (is the stated assumption reasonable?):", ""]
            for r in reviews:
                assumed = "; ".join(r["assumptions"]) or "no assumption stated"
                lines.append(f"- **{r['id']}** ({r['outcome']}): {assumed}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- helpers
def results_path(meta_set: str, model: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_model = model.replace(":", "-").replace("/", "-")
    return RESULTS_DIR / f"{stamp}_{meta_set}_{safe_model}.json"


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)   # never leave a half-written results file


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=EVALS_DIR, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
