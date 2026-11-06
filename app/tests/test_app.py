"""Drive the real page script with Streamlit's AppTest and a scripted model."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agent import providers
from agent.tests.conftest import ScriptedProvider, say, tool

MAIN = str(Path(__file__).resolve().parents[1] / "main.py")
DB = Path(__file__).resolve().parents[2] / "data" / "adventureworks.duckdb"


@pytest.fixture
def app(monkeypatch, tmp_path):
    if not DB.exists():
        pytest.skip("Database not built; run python data/load.py")
    """App with Ollama reported as available and a scripted model behind it."""
    script = []
    monkeypatch.setattr(providers, "make_provider", lambda name, s: ScriptedProvider(script))
    monkeypatch.setattr("app.ui.provider_status", lambda name, s: (True, "test model"))
    monkeypatch.setattr("agent.config.Settings.path",
                        lambda self, key: tmp_path / "state.sqlite" if key == "app_state"
                        else self.root / self.raw["paths"][key])
    at = AppTest.from_file(MAIN, default_timeout=30)
    at.script = script
    return at


def test_empty_state_shows_examples(app):
    app.run()
    assert not app.exception
    assert "Ask about internet sales" in app.markdown[0].value
    assert len([b for b in app.button if b.key and b.key.startswith("example-")]) == 4


def test_question_renders_answer_chart_table_and_working(app):
    app.script.extend([
        tool("run_query", measures=["revenue", "gross_margin_pct"], dimensions=["order_date__quarter"],
             filters=[{"field": "order_date", "op": "=", "value": "2013"}], assumptions=["Calendar 2013"]),
        say("Revenue grew every quarter, from $2,678,946 to $5,328,574."),
    ])
    app.run()
    app.chat_input[0].set_value("Quarterly revenue in 2013").run()
    assert not app.exception

    md = " ".join(m.value for m in app.markdown)
    assert r"from \$2,678,946 to \$5,328,574" in md          # $ escaped, not LaTeX
    assert "Calendar 2013" in app.info[0].value
    assert any("Default chart" in c.value for c in app.caption)
    assert {"Data · 4 rows", "How this was answered"} <= {e.label for e in app.main.expander}
    sql_blocks = [c.value for c in app.code if "SELECT" in c.value]
    assert sql_blocks and "DATE_TRUNC('quarter'" in sql_blocks[0]
    cost = [c.value for c in app.caption if "model call" in c.value][0]
    assert "2 model calls" in cost and "free" in cost

    table = app.dataframe[0].value
    assert list(table.iloc[:, 0]) == ["2013 Q1", "2013 Q2", "2013 Q3", "2013 Q4"]


def test_rejected_query_shown_in_working(app):
    app.script.extend([tool("run_query", measures=["sales"]), tool("run_query", measures=["revenue"]),
                       say("Total revenue is $29,358,677.")])
    app.run()
    app.chat_input[0].set_value("Total sales").run()
    assert "1 rejected query" in " ".join(m.value for m in app.markdown)
    assert any("(1 rejected)" in c.value for c in app.caption)


def test_guard_decline_needs_no_model(app):
    app.run()
    app.chat_input[0].set_value("Reseller sales by year").run()   # script is empty: any model call would fail
    assert not app.exception
    assert "internet (direct-to-consumer) sales only" in app.warning[0].value
    assert any("no model call" in c.value for c in app.caption)


def test_example_button_asks_question(app):
    app.script.extend([tool("run_query", measures=["revenue"],
                            filters=[{"field": "order_date", "op": "=", "value": "2013"}]),
                       say("Total revenue in 2013 was $16,351,550.")])
    app.run()
    next(b for b in app.button if b.key == "example-0").click().run()
    assert not app.exception
    assert any("16,351,550" in m.value for m in app.markdown)
    assert app.sidebar.metric[0].value == "1"


def test_provider_unavailable_disables_input(app, monkeypatch):
    monkeypatch.setattr("app.ui.provider_status", lambda name, s: (False, "Ollama isn't running"))
    app.run()
    assert app.chat_input[0].disabled
    assert any("Ollama isn't running" in c.value for c in app.sidebar.caption)


def test_unverified_figure_warning(app):
    app.script.extend([tool("run_query", measures=["gross_margin"], dimensions=["product_category"]),
                       say("Bikes made $11,506,197 of gross margin.")])
    app.run()
    app.chat_input[0].set_value("Margin by category").run()
    assert r"\$11,506,197 doesn't match the query results" in app.warning[0].value


def test_follow_up_carries_previous_turn(app):
    top5 = {"measures": ["revenue", "gross_margin"], "dimensions": ["product_name"],
            "sort": [{"field": "revenue", "dir": "desc"}], "limit": 5}
    app.script.extend([tool("run_query", **top5), say("Mountain-200 Black, 46 leads with $1,373,469.55."),
                       tool("run_query", **top5), say("Here they are side by side.")])
    app.run()
    app.chat_input[0].set_value("top 5 items by sales and their profit").run()
    app.chat_input[0].set_value("plot those side by side as a clustered column chart").run()
    assert not app.exception
    # Default chart for the follow-up: both money measures, grouped.
    import json
    fig = json.loads(app.get("plotly_chart")[-1].proto.spec)
    assert fig["layout"]["barmode"] == "group"
    assert [t["name"] for t in fig["data"]] == ["Revenue", "Gross margin"]
    assert app.sidebar.metric[0].value == "2"
