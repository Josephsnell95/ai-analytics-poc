"""The scorer must mark right answers right and wrong answers wrong. Results are
built by running real queries, as the agent would."""

import pytest

from agent.loop import AgentResult
from evals.scoring import score
from semantic import SemanticEngine

REPO_CONFIG = __import__("pathlib").Path(__file__).resolve().parents[2] / "config.yaml"


@pytest.fixture(scope="module")
def engine():
    if not (REPO_CONFIG.parent / "data" / "adventureworks.duckdb").exists():
        pytest.skip("Database not built")
    e = SemanticEngine.from_config(REPO_CONFIG)
    yield e
    e.close()


def result(engine, query, status="answered", unverified=None, rows=None, declined_by=None):
    """An AgentResult as the agent would return it after running this query."""
    r = AgentResult(question="q", provider="fake", model="m", status=status, answer="a",
                    declined_by=declined_by, unverified=unverified or [])
    if query is not None:
        res = engine.run(query)
        r.semantic_query = query
        r.columns = [c.name for c in res.columns]
        r.column_formats = {c.name: c.format for c in res.columns}
        r.rows = rows if rows is not None else res.records()
    return r


def item(reference=None, **kw):
    return {"id": "x", "category": "c", "question": "q", "reference": reference, **kw}


BY_GROUP = {"measures": ["revenue"], "dimensions": ["sales_territory_group"]}
TOTAL_2013 = {"measures": ["revenue"], "filters": [{"field": "order_date", "op": "=", "value": "2013"}]}
TOP5_CITIES = {"measures": ["customer_count"], "dimensions": ["customer_city"],
               "sort": [{"field": "customer_count", "dir": "desc"}], "limit": 5}


# ------------------------------------------------------------------ exact
def test_same_query_is_correct(engine):
    assert score(item(BY_GROUP), result(engine, BY_GROUP), engine).outcome == "correct"


def test_differently_written_query_with_same_values_is_correct(engine):
    agent_query = {"measures": ["revenue", "order_count"], "dimensions": ["sales_territory_group"],
                   "sort": [{"field": "revenue", "dir": "desc"}]}   # extra measure, different order
    assert score(item(BY_GROUP), result(engine, agent_query), engine).outcome == "correct"


def test_different_breakdown_is_wrong(engine):
    s = score(item(BY_GROUP), result(engine, {"measures": ["revenue"], "dimensions": ["sales_territory_country"]}), engine)
    assert s.outcome == "wrong" and "not grouped by sales_territory_group" in s.reason


def test_extra_dimension_is_wrong(engine):
    q = {"measures": ["revenue"], "dimensions": ["sales_territory_group", "order_date__year"]}
    s = score(item(BY_GROUP), result(engine, q), engine)
    assert s.outcome == "wrong" and "different breakdown" in s.reason


def test_wrong_filter_values_differ(engine):
    q = dict(BY_GROUP, filters=[{"field": "order_date", "op": "=", "value": "2013"}])
    s = score(item(BY_GROUP), result(engine, q), engine)
    assert s.outcome == "wrong" and "values differ" in s.reason


# --------------------------------------------------------------- contains
def test_total_found_inside_a_breakdown_is_correct(engine):
    germany = {"measures": ["order_count"], "filters": [{"field": "customer_country", "op": "=", "value": "Germany"}]}
    by_country = {"measures": ["order_count"], "dimensions": ["customer_country"]}
    assert score(item(germany), result(engine, by_country), engine).outcome == "correct"


def test_single_day_instead_of_year_is_wrong(engine):
    # The first smoke run's Q1 failure.
    day = {"measures": ["revenue"], "filters": [{"field": "order_date", "op": "=", "value": "2013-01-01"}]}
    s = score(item(TOTAL_2013), result(engine, day), engine)
    assert s.outcome == "wrong" and "expected row not found" in s.reason


# ---------------------------------------------------------------- ordered
def test_top_n_correct(engine):
    assert score(item(TOP5_CITIES), result(engine, TOP5_CITIES), engine).outcome == "correct"


def test_top_n_tie_at_cut_off_accepts_either(engine):
    r = result(engine, TOP5_CITIES)
    swapped = [dict(row) for row in r.rows]
    tied = [row for row in swapped if row["customer_count"] == 210]
    assert len(tied) == 1   # one of Beaverton / Bellingham made the cut
    tied[0]["customer_city"] = "Bellingham" if tied[0]["customer_city"] == "Beaverton" else "Beaverton"
    r.rows = swapped
    assert score(item(TOP5_CITIES), r, engine).outcome == "correct"


def test_top_n_unsorted_rows_are_wrong(engine):
    # The 2026-11-06 follow-up failure: right year, but no sort or limit, five arbitrary rows.
    ref = {"measures": ["revenue"], "dimensions": ["product_name"],
           "filters": [{"field": "order_date", "op": "=", "value": "2013"}],
           "sort": [{"field": "revenue", "dir": "desc"}], "limit": 5}
    unsorted = {k: v for k, v in ref.items() if k not in ("sort", "limit")}
    s = score(item(ref), result(engine, unsorted), engine)
    assert s.outcome == "wrong" and "wrong ranking" in s.reason
    r = result(engine, unsorted)
    r.rows = r.rows[:5]
    assert "wrong ranking" in score(item(ref), r, engine).reason


def test_top_n_invented_row_is_wrong(engine):
    r = result(engine, TOP5_CITIES)
    r.rows = [dict(row) for row in r.rows]
    r.rows[0]["customer_city"] = "Warrington"
    assert "row not in the data" in score(item(TOP5_CITIES), r, engine).reason


# -------------------------------------------------- alternatives and flags
def test_alternative_reference_accepted(engine):
    ref = {"measures": ["revenue"], "filters": [{"field": "customer_country", "op": "=", "value": "Canada"}]}
    alt = {"measures": ["revenue"], "filters": [{"field": "sales_territory_country", "op": "=", "value": "Canada"}]}
    by_territory = {"measures": ["revenue"], "dimensions": ["sales_territory_country"]}
    assert score(item(ref, alternatives=[alt]), result(engine, by_territory), engine).passed


def test_correct_but_flagged_figure(engine):
    s = score(item(BY_GROUP), result(engine, BY_GROUP, unverified=["$1.2M"]), engine)
    assert s.outcome == "correct_flagged" and s.passed and "$1.2M" in s.reason


def test_no_query(engine):
    assert score(item(BY_GROUP), result(engine, None, status="no_query"), engine).outcome == "no_result"


# ----------------------------------------------------------------- declines
def test_decline_expected_and_given(engine):
    s = score(item(expect="decline"), result(engine, None, status="declined", declined_by="guard"), engine)
    assert s.outcome == "declined_correctly" and "guard" in s.reason


def test_decline_expected_but_answered(engine):
    s = score(item(expect="decline"), result(engine, TOTAL_2013), engine)
    assert s.outcome == "answered_out_of_scope" and not s.passed


def test_answerable_question_declined(engine):
    s = score(item(BY_GROUP), result(engine, None, status="declined", declined_by="model"), engine)
    assert s.outcome == "wrongly_declined"


def test_provider_error(engine):
    r = result(engine, None, status="error")
    r.error = "Can't reach Ollama"
    assert score(item(BY_GROUP), r, engine).outcome == "error"


def test_missing_measure(engine):
    s = score(item(BY_GROUP), result(engine, {"measures": ["order_count"], "dimensions": ["sales_territory_group"]}), engine)
    assert s.outcome == "wrong" and "missing measure(s) revenue" in s.reason


def test_top_n_with_extra_sorted_rows_is_correct(engine):
    # Haiku in the first eval run: "which promotion brought in the most revenue?" with all 4 listed.
    ref = {"measures": ["revenue"], "dimensions": ["promotion_name"], "sort": [{"field": "revenue", "dir": "desc"}], "limit": 1}
    assert score(item(ref), result(engine, dict(ref, limit=5)), engine).outcome == "correct"


def test_top_n_wrong_direction_is_wrong(engine):
    # Llama: "4 least profitable models" sorted descending.
    ref = {"measures": ["gross_margin"], "dimensions": ["product_model"], "sort": [{"field": "gross_margin", "dir": "asc"}], "limit": 4}
    s = score(item(ref), result(engine, dict(ref, sort=[{"field": "gross_margin", "dir": "desc"}])), engine)
    assert s.outcome == "wrong" and "wrong ranking" in s.reason
