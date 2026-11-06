import json

import pytest

from agent.loop import Agent
from agent.prompt import CANNOT_ANSWER
from agent.tests.conftest import ScriptedProvider, say, tool

REVENUE_2013 = {
    "measures": ["revenue"],
    "filters": [{"field": "order_date", "op": "between", "value": ["2013-01-01", "2013-12-31"]}],
    "assumptions": ["2013 means calendar 2013"],
}


def make_agent(engine, settings, responses):
    provider = ScriptedProvider(responses)
    return Agent(provider, engine, settings), provider


def test_query_then_answer(engine, settings):
    agent, provider = make_agent(engine, settings, [
        tool("run_query", **REVENUE_2013),
        say("Revenue in 2013 was $16,351,550."),
    ])
    r = agent.ask("What was revenue in 2013?")

    assert r.status == "answered"
    assert r.answer == "Revenue in 2013 was $16,351,550."
    assert r.rows == [{"revenue": pytest.approx(16_351_550.34)}]
    assert r.assumptions == ["2013 means calendar 2013"]
    assert "BETWEEN DATE '2013-01-01' AND DATE '2013-12-31'" in r.sql
    assert (r.usage.llm_calls, r.usage.tool_calls) == (2, 1)
    assert (r.usage.input_tokens, r.usage.output_tokens) == (2200, 130)
    assert r.usage.cost_usd == 0

    # The tool result fed back to the model is compact JSON with rounded numbers.
    tool_msg = provider.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["rows"] == [{"revenue": 16351550.34}]


def test_system_prompt_contains_model(engine, settings):
    agent, provider = make_agent(engine, settings, [say("hi")])
    agent.ask("hello")
    system = provider.calls[0]["system"]
    assert "gross_margin_pct" in system and "Fiscal periods are not modelled" in system


def test_validation_error_is_fed_back_and_retried(engine, settings):
    agent, provider = make_agent(engine, settings, [
        tool("run_query", measures=["sales"]),
        tool("run_query", measures=["revenue"]),
        say("Total revenue is $29,358,677."),
    ])
    r = agent.ask("Total sales?")
    assert r.status == "answered"
    assert [a.error is not None for a in r.attempts] == [True, False]
    feedback = provider.calls[1]["messages"][-1]["content"]
    assert "synonym of 'revenue'" in feedback and "call run_query again" in feedback


def test_chart_from_last_result(engine, settings):
    agent, _ = make_agent(engine, settings, [
        tool("run_query", measures=["revenue"], dimensions=["product_category"]),
        tool("make_chart", chart_type="bar", x="product_category", y=["revenue"]),
        say("Bikes dominate."),
    ])
    r = agent.ask("Revenue by category")
    assert r.chart["data"][0]["type"] == "bar"
    assert r.chart["data"][0]["x"] == ["Accessories", "Bikes", "Clothing"]


def test_bad_chart_returns_error_to_model(engine, settings):
    agent, provider = make_agent(engine, settings, [
        tool("run_query", measures=["revenue"], dimensions=["product_category"]),
        tool("make_chart", chart_type="bar", x="category", y=["revenue"]),
        say("Done."),
    ])
    r = agent.ask("Revenue by category")
    assert "not a column of the last result" in provider.calls[2]["messages"][-1]["content"]
    assert r.chart_auto and r.chart["data"][0]["x"] == ["Accessories", "Bikes", "Clothing"]


def test_default_chart_when_agent_skips_make_chart(engine, settings):
    agent, _ = make_agent(engine, settings, [
        tool("run_query", measures=["revenue"], dimensions=["order_date__month"],
             filters=[{"field": "order_date", "op": "=", "value": "2013"}]),
        say("Revenue grew through 2013."),
    ])
    r = agent.ask("Plot monthly revenue for 2013")
    assert r.chart_auto and r.chart["layout"]["meta"]["chart_type"] == "line"
    assert len(r.chart["data"][0]["x"]) == 12


def test_no_default_chart_for_single_number(engine, settings):
    agent, _ = make_agent(engine, settings, [tool("run_query", measures=["revenue"]), say("$29.4M")])
    r = agent.ask("Total revenue")
    assert r.chart is None and not r.chart_auto


def test_agent_chart_is_not_replaced(engine, settings):
    agent, _ = make_agent(engine, settings, [
        tool("run_query", measures=["revenue"], dimensions=["product_category"]),
        tool("make_chart", chart_type="pie", x="product_category", y=["revenue"]),
        say("Bikes dominate."),
    ])
    r = agent.ask("Revenue share by category")
    assert not r.chart_auto and r.chart["data"][0]["type"] == "pie"


def test_column_formats_reported(engine, settings):
    agent, _ = make_agent(engine, settings, [
        tool("run_query", measures=["revenue", "gross_margin_pct", "order_count"]), say("ok")])
    r = agent.ask("x")
    assert r.column_formats == {"revenue": "currency", "gross_margin_pct": "percent", "order_count": "number"}


def test_declined(engine, settings):
    agent, _ = make_agent(engine, settings, [say(f"{CANNOT_ANSWER}: reseller sales are not in the model.")])
    r = agent.ask("Reseller sales last year?")
    assert r.status == "declined" and r.semantic_query is None


def test_answer_without_query_is_flagged(engine, settings):
    agent, _ = make_agent(engine, settings, [say("Revenue was about $30 million.")])
    assert agent.ask("Revenue?").status == "no_query"


def test_tool_call_cap(engine, settings):
    cap = settings.agent["max_tool_calls"]
    agent, provider = make_agent(engine, settings, [tool("get_model") for _ in range(cap + 1)])
    r = agent.ask("Loop forever")
    assert r.status == "capped"
    assert r.usage.tool_calls == cap
    assert len(provider.calls) == cap + 1  # the call that would exceed the cap is not run


def test_provider_error_is_reported_not_raised(engine, settings):
    from agent.providers import ProviderError

    class Broken:
        name, model = "fake", "llama3.1:8b"

        def chat(self, *a):
            raise ProviderError("Can't reach Ollama")

    r = Agent(Broken(), engine, settings).ask("Revenue?")
    assert r.status == "error" and "Can't reach Ollama" in r.error


def test_last_successful_query_is_reported(engine, settings):
    agent, _ = make_agent(engine, settings, [
        tool("run_query", measures=["revenue"]),
        tool("run_query", measures=["order_count"]),
        tool("run_query", measures=["nope"]),
        say("There were 27,659 orders."),
    ])
    r = agent.ask("How many orders?")
    assert r.semantic_query == {"measures": ["order_count"]}
    assert len(r.attempts) == 3


def test_state_resets_between_questions(engine, settings):
    agent, _ = make_agent(engine, settings, [
        tool("run_query", measures=["revenue"]), say("A"),
        say("B"),
    ])
    agent.ask("first")
    r = agent.ask("second")
    assert r.semantic_query is None and r.attempts == []


def test_guard_declines_without_calling_the_model(engine, settings):
    agent, provider = make_agent(engine, settings, [])   # any model call would fail the test
    r = agent.ask("What were our reseller sales last year?")
    assert r.status == "declined" and r.declined_by == "guard"
    assert r.answer.startswith(CANNOT_ANSWER) and "internet (direct-to-consumer) sales only" in r.answer
    assert provider.calls == [] and r.usage.llm_calls == 0 and r.usage.cost_usd == 0


def test_model_decline_is_labelled(engine, settings):
    agent, _ = make_agent(engine, settings, [say(f"{CANNOT_ANSWER}: no weather data.")])
    r = agent.ask("Did rain affect bike sales?")
    assert r.status == "declined" and r.declined_by == "model"


def test_prompt_lists_unavailable_topics(engine, settings):
    agent, provider = make_agent(engine, settings, [say("hi")])
    agent.ask("hello")
    assert "NOT AVAILABLE" in provider.calls[0]["system"] and "Reseller sales:" in provider.calls[0]["system"]


def test_progress_events(engine, settings):
    agent, _ = make_agent(engine, settings, [
        tool("run_query", measures=["sales"]),
        tool("run_query", measures=["revenue"], dimensions=["product_category"]),
        tool("make_chart", chart_type="bar", x="product_category", y=["revenue"]),
        say("Done."),
    ])
    events = []
    agent.ask("Revenue by category", on_event=events.append)
    assert events == [
        "Thinking", "Running a semantic query", "Query rejected by the semantic layer; the model is fixing it",
        "Reading the results", "Running a semantic query",
        "Reading the results", "Drawing a chart",
        "Reading the results",
    ]


def test_misread_figure_flagged_on_result(engine, settings):
    agent, _ = make_agent(engine, settings, [
        tool("run_query", measures=["gross_margin"], dimensions=["product_category"]),
        say("Bikes made $11,506,197 of gross margin; Clothing made $136,413."),
    ])
    r = agent.ask("Margin by category")
    assert r.unverified == ["$11,506,197"]


def test_correct_answer_not_flagged(engine, settings):
    agent, _ = make_agent(engine, settings, [
        tool("run_query", measures=["revenue"], filters=[{"field": "order_date", "op": "=", "value": "2013"}]),
        say("Revenue in 2013 was $16.35 million ($16,351,550.34)."),
    ])
    assert agent.ask("Revenue 2013").unverified == []


# ------------------------------------------------------------ follow-ups
TOP5 = {"measures": ["revenue", "gross_margin"], "dimensions": ["product_name"],
        "sort": [{"field": "revenue", "dir": "desc"}], "limit": 5}


def test_follow_up_sees_previous_question_answer_and_query(engine, settings):
    agent, provider = make_agent(engine, settings, [
        tool("run_query", **TOP5), say("Mountain-200 Black, 46 leads with $1,373,469.55."),
        tool("run_query", **TOP5), say("Shown side by side."),
    ])
    first = agent.ask("Top 5 items by sales with their profit")
    agent.ask("plot those side by side as a clustered column chart", history=[first])

    sent = provider.calls[2]["messages"]   # first model call of the second question
    assert [m["role"] for m in sent[:3]] == ["user", "assistant", "user"]
    assert sent[0]["content"] == "Top 5 items by sales with their profit"
    assert "Mountain-200 Black, 46 leads" in sent[1]["content"]
    assert '[Query behind this answer: {"measures": ["revenue", "gross_margin"]' in sent[1]["content"]
    assert sent[2]["content"] == "plot those side by side as a clustered column chart"
    assert "rows" not in sent[1]["content"]   # no result rows carried over


def test_history_limited_to_configured_turns(engine, settings):
    agent, provider = make_agent(engine, settings, [say("a"), say("b"), say("c"), say("d")])
    history = [agent.ask(q) for q in ("q1", "q2", "q3")]
    agent.ask("q4", history=history)
    users = [m["content"] for m in provider.calls[-1]["messages"] if m["role"] == "user"]
    assert users == ["q2", "q3", "q4"]   # history_turns: 2


def test_no_history_by_default(engine, settings):
    agent, provider = make_agent(engine, settings, [say("x")])
    agent.ask("q")
    assert len(provider.calls[0]["messages"]) == 1


def test_prompt_has_follow_up_rule(engine, settings):
    agent, provider = make_agent(engine, settings, [say("x")])
    agent.ask("q")
    assert "never reuse numbers from an earlier answer" in provider.calls[0]["system"]
