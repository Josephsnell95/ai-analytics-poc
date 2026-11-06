import json

import pytest

from agent.log import QuestionLog
from agent.loop import Agent
from agent.tests.conftest import ScriptedProvider, say, tool


def test_haiku_cost(settings):
    # 10,000 in and 1,000 out at $0.10 / $0.50 per million
    assert settings.cost("claude-haiku-5-5", 10_000, 1_000) == pytest.approx(0.0015)


def test_long_context_rate_applies_to_whole_request(settings):
    assert settings.cost("claude-haiku-5-5", 200_000, 1_000) == pytest.approx(0.1 + 0.0025)


def test_local_model_is_free(settings):
    assert settings.cost("llama3.1:8b", 50_000, 5_000) == 0


def test_unpriced_model_is_an_error_not_free(settings):
    with pytest.raises(KeyError, match="No prices for 'claude-opus-9'"):
        settings.cost("claude-opus-9", 1, 1)


def test_configured_models_all_have_prices(settings):
    for name in settings.raw["providers"]:
        assert settings.provider(name)["model"] in settings.prices


def test_log_round_trip(tmp_path, engine, settings):
    agent = Agent(ScriptedProvider([
        tool("run_query", measures=["nope"]),
        tool("run_query", measures=["revenue"], assumptions=["all time"]),
        say("Revenue is $29.4M."),
    ]), engine, settings)
    result = agent.ask("Revenue?")

    log = QuestionLog(tmp_path / "state.sqlite")
    log.write(result)
    row = log.con.execute(
        "SELECT status, failed_queries, tool_calls, input_tokens, assumptions, semantic_query, transcript "
        "FROM question_log").fetchone()
    assert row[:4] == ("answered", 1, 2, 3200)
    assert json.loads(row[4]) == ["all time"]
    assert json.loads(row[5])["measures"] == ["revenue"]
    assert len(json.loads(row[6])) == 6   # user, 2x(assistant+tool), final assistant

    summary = log.summary()
    assert summary[0][:4] == ("fake", "llama3.1:8b", 1, 1)
    log.close()


def test_old_log_file_gets_new_column(tmp_path):
    import sqlite3
    old = tmp_path / "old.sqlite"
    con = sqlite3.connect(old)
    con.execute("CREATE TABLE question_log (id INTEGER PRIMARY KEY, question TEXT)")
    con.commit(); con.close()
    log = QuestionLog(old)
    cols = {row[1] for row in log.con.execute("PRAGMA table_info(question_log)")}
    assert "declined_by" in cols
    log.close()


def test_guard_decline_logged(tmp_path, engine, settings):
    r = Agent(ScriptedProvider([]), engine, settings).ask("Reseller sales by year")
    log = QuestionLog(tmp_path / "s.sqlite")
    log.write(r)
    assert log.get()["declined_by"] == "guard"
    log.close()
