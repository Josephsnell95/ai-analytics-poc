import json
from pathlib import Path

import pytest

from agent.config import load_settings
from agent.loop import Agent
from agent.tests.conftest import ScriptedProvider, say, tool
from evals.runner import latest_results, report, run_set
from semantic import SemanticEngine

ROOT = Path(__file__).resolve().parents[2]
CAT_2012 = {"measures": ["revenue"], "dimensions": ["product_category"],
            "filters": [{"field": "order_date", "op": "=", "value": "2012"}]}
ITEMS = [
    {"id": "parent", "set": "tuning", "category": "breakdown", "question": "Revenue by category in 2012",
     "reference": CAT_2012},
    {"id": "child", "set": "tuning", "category": "follow_up", "question": "Now split that by quarter",
     "follow_up_of": "parent", "reference": dict(CAT_2012, dimensions=["product_category", "order_date__quarter"])},
    {"id": "resellers", "set": "tuning", "category": "out_of_scope_listed",
     "question": "Reseller sales in 2013?", "expect": "decline"},
    {"id": "later", "set": "tuning", "category": "lookup", "question": "Some later question"},   # no reference yet
]


@pytest.fixture
def setup():
    if not (ROOT / "data" / "adventureworks.duckdb").exists():
        pytest.skip("Database not built")
    settings = load_settings(ROOT)
    return settings, SemanticEngine.from_config(ROOT / "config.yaml")


def script():
    return [
        tool("run_query", **CAT_2012), say("Bikes dominated 2012."),
        tool("run_query", **dict(CAT_2012, dimensions=["product_category"])),   # forgot the quarter
        say("Here it is by quarter."),
    ]


def test_run_scores_follow_up_with_history_and_writes_file(setup, tmp_path):
    settings, engine = setup
    provider = ScriptedProvider(script())
    out = tmp_path / "run.json"
    lines = []
    data = run_set(Agent(provider, engine, settings), ITEMS, out, progress=lines.append)

    saved = json.loads(out.read_text())
    assert [r["id"] for r in saved["results"]] == ["parent", "child", "resellers"]
    assert [r["outcome"] for r in saved["results"]] == ["correct", "wrong", "declined_correctly"]
    assert "different breakdown" in saved["results"][1]["reason"]
    assert saved["results"][2]["declined_by"] == "guard"
    assert saved["meta"]["awaiting_reference"] == ["later"]
    assert saved["meta"]["prompt_sha"] and saved["meta"]["finished"]

    # The follow-up was asked with the parent's turn as history.
    child_call = provider.calls[2]["messages"]
    assert child_call[0]["content"] == "Revenue by category in 2012" and child_call[-1]["content"] == "Now split that by quarter"

    s = data["summary"]
    assert (s["questions"], s["passed"], s["accuracy"]) == (3, 2, 0.667)
    assert s["by_category"]["follow_up"] == {"passed": 0, "of": 1}
    assert lines[1].startswith("[2/3] FAIL child")


def test_resume_skips_finished_questions_and_asks_parent_unscored(setup, tmp_path):
    settings, engine = setup
    out = tmp_path / "run.json"
    run_set(Agent(ScriptedProvider(script()), engine, settings), ITEMS, out, progress=lambda _: None)
    partial = json.loads(out.read_text())
    partial["results"] = partial["results"][:1]   # pretend it stopped after the parent
    partial["meta"]["finished"] = None

    provider = ScriptedProvider(script())   # parent re-asked for history, then the child
    data = run_set(Agent(provider, engine, settings), ITEMS, out, resume=partial, progress=lambda _: None)
    assert [r["id"] for r in data["results"]] == ["parent", "child", "resellers"]
    assert len(provider.calls) == 4   # parent (2 calls, unscored) + child (2 calls); resellers is guarded


def test_report_table(setup, tmp_path):
    settings, engine = setup
    run_set(Agent(ScriptedProvider(script()), engine, settings), ITEMS, tmp_path / "20261106-120000_tuning_m.json",
            progress=lambda _: None)
    text = report(latest_results(tmp_path))
    assert "| tuning | llama3.1:8b | 67% | 2/3 |" in text
    assert "| follow_up | 0/1 |" in text
    assert "**child** (wrong): different breakdown" in text
    assert "awaiting a reference query: later" in text


def test_report_ignores_unfinished_runs(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"meta": {"finished": None}, "results": []}))
    assert "No finished runs" in report(latest_results(tmp_path))


def test_perfect_agent_scores_100_percent_on_the_tuning_set(setup, tmp_path):
    """An agent that runs exactly each reference query (and declines what it should)
    must pass everything. If not, the scorer or a reference is wrong, not the model."""
    from agent.prompt import CANNOT_ANSWER
    from agent.providers import LLMResponse, ToolCall
    from evals.runner import load_questions

    settings, engine = setup
    items = load_questions("tuning")
    by_question = {i["question"]: i for i in items}

    class Oracle:
        name, model = "oracle", "llama3.1:8b"

        def chat(self, system, messages, tools):
            q = by_question[[m for m in messages if m["role"] == "user"][-1]["content"]]
            answered = messages[-1]["role"] == "tool"
            if q.get("expect") == "decline":
                return LLMResponse(f"{CANNOT_ANSWER}: not in the model.")
            if answered:
                return LLMResponse("Done.")
            return LLMResponse("", [ToolCall("c", "run_query", dict(q["reference"]))])

    data = run_set(Agent(Oracle(), engine, settings), items, tmp_path / "oracle.json", progress=lambda _: None)
    failures = [(r["id"], r["reason"]) for r in data["results"] if not r["passed"]]
    assert failures == []
    assert data["summary"]["declined_by"] == {"guard": 2, "model": 3}


def test_run_stops_after_repeated_provider_errors(setup, tmp_path):
    from agent.providers import ProviderError
    from evals.runner import MAX_ERRORS_IN_A_ROW, load_questions

    class Gone:
        name, model = "ollama", "llama3.1:8b"

        def chat(self, *a):
            raise ProviderError("Can't reach Ollama")

    settings, engine = setup
    answerable = [i for i in load_questions("tuning") if i.get("expect") != "decline"]
    lines = []
    data = run_set(Agent(Gone(), engine, settings), answerable, tmp_path / "r.json", progress=lines.append)
    assert len(data["results"]) == MAX_ERRORS_IN_A_ROW and data["meta"]["finished"] is None
    assert "--resume" in lines[-1]
