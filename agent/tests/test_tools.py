import json

import pytest

from agent.charts import ChartError, build_chart
from agent.tools import Tools, describe_model, normalise_query_args


# --------------------------------------------------------- argument repair
@pytest.mark.parametrize("raw, fixed", [
    ({"measures": '["revenue"]'}, {"measures": ["revenue"]}),
    ({"measures": "revenue"}, {"measures": ["revenue"]}),
    ({"measures": "revenue, order_count"}, {"measures": ["revenue", "order_count"]}),
    ({"query": {"measures": ["revenue"]}}, {"measures": ["revenue"]}),
    ({"query": '{"measures": ["revenue"]}'}, {"measures": ["revenue"]}),
    ({"measures": ["revenue"], "filters": None, "sort": []}, {"measures": ["revenue"]}),
    ({"measures": ["revenue"], "limit": "5"}, {"measures": ["revenue"], "limit": 5}),
    ({"measures": ["revenue"], "dimensions": {"field": "order_date", "grain": "year"}},
     {"measures": ["revenue"], "dimensions": ["order_date__year"]}),
    # Slips seen in the first Llama smoke run:
    ({"measures": ["revenue"], "limit": 0}, {"measures": ["revenue"]}),
    ({"measures": ["revenue"], "limit": "null"}, {"measures": ["revenue"]}),
    ({"measures": ["revenue"], "dimensions": [{"field": "customer_country", "grain": None}]},
     {"measures": ["revenue"], "dimensions": ["customer_country"]}),
    ({"measures": ["revenue"], "dimensions": [{"field": "customer_country", "grain": ""}]},
     {"measures": ["revenue"], "dimensions": ["customer_country"]}),
    ({"measures": ["revenue"], "dimensions": [{"field": "product_category"}]},
     {"measures": ["revenue"], "dimensions": ["product_category"]}),    ({"measures": ["order_count"], "dimensions": [{"field": "customer_country", "grain": None, "op": None, "value": None}]},
     {"measures": ["order_count"], "dimensions": ["customer_country"]}),
])
def test_normalise_fixes_formatting_slips(raw, fixed):
    assert normalise_query_args(raw) == fixed


def test_normalise_leaves_wrong_fields_for_the_compiler():
    assert normalise_query_args({"measures": ["turnover"]}) == {"measures": ["turnover"]}


def test_normalise_leaves_filter_inside_dimension_for_the_compiler():
    # Guessing the intent of {"field", "op", "value"} in dimensions could change the answer.
    d = {"field": "customer_country", "op": "=", "value": "Germany"}
    assert normalise_query_args({"measures": ["order_count"], "dimensions": [d]})["dimensions"] == [d]


# ---------------------------------------------------------------- run_query
def test_run_query_caps_rows_shown_to_model(engine):
    tools = Tools(engine, rows_to_llm=5)
    out = json.loads(tools.run_query({"measures": ["revenue"], "dimensions": ["product_name"]}))
    assert len(out["rows"]) == 5 and out["row_count"] == 100
    assert "first 5 of 100" in out["note"] and "hit the limit" in out["note_limit"]
    assert len(tools.state.last_result.rows) == 100   # the app still gets every row


def test_run_query_empty_result_hint(engine):
    tools = Tools(engine, rows_to_llm=50)
    out = json.loads(tools.run_query({"measures": ["revenue"], "dimensions": ["product_category"],
                                      "filters": [{"field": "product_category", "op": "=", "value": "Components"}]}))
    assert out["row_count"] == 0 and "2010-12-29 to 2014-01-28" in out["note"]


def test_percent_rounded_to_four_places(engine):
    tools = Tools(engine, rows_to_llm=50)
    out = json.loads(tools.run_query({"measures": ["gross_margin_pct", "revenue"]}))
    exact = tools.state.last_result.records()[0]
    assert out["rows"][0]["gross_margin_pct"] == round(exact["gross_margin_pct"], 4)
    assert out["rows"][0]["revenue"] == round(exact["revenue"], 2)


def test_unknown_tool(engine):
    assert "Unknown tool 'sql'" in Tools(engine, 50).call("sql", {})


# --------------------------------------------------------------- get_model
def test_describe_model_full_and_search(engine):
    full = describe_model(engine.model)
    assert "revenue:" in full and "customer_country:" in full and "CONTEXT:" in full
    found = describe_model(engine.model, "margin")
    assert "gross_margin_pct" in found and "customer_country" not in found
    assert "No fields match" in describe_model(engine.model, "zzz")


def test_describe_model_search_matches_synonyms(engine):
    assert "revenue:" in describe_model(engine.model, "turnover")


# -------------------------------------------------------------------- charts
@pytest.fixture
def monthly(engine):
    return engine.run({"measures": ["revenue", "order_count"],
                       "dimensions": [{"field": "order_date", "grain": "month"}, "product_category"],
                       "filters": [{"field": "order_date", "op": "between", "value": ["2013-01-01", "2013-03-31"]}],
                       "sort": [{"field": "revenue", "dir": "desc"}]})


def test_line_chart_sorted_by_time_and_split_by_series(monthly):
    spec = build_chart(monthly, "line", "order_date__month", ["revenue"], series="product_category")
    assert {t["name"] for t in spec["data"]} == {"Bikes", "Accessories", "Clothing"}
    bikes = next(t for t in spec["data"] if t["name"] == "Bikes")
    assert bikes["x"] == ["2013-01-01", "2013-02-01", "2013-03-01"]   # time order, not revenue order
    assert spec["layout"]["yaxis"]["tickformat"] == "$,.0f"


def test_y_as_single_string(monthly):
    assert build_chart(monthly, "bar", "product_category", "revenue")["data"][0]["type"] == "bar"


def test_chart_errors_list_valid_columns(monthly):
    with pytest.raises(ChartError) as e:
        build_chart(monthly, "donut", "month", ["product_category"])
    msg = " ".join(e.value.errors)
    assert "must be one of" in msg and "Columns:" in msg and "not a measure column" in msg


def test_series_and_multiple_y_conflict(monthly):
    with pytest.raises(ChartError, match="not both"):
        build_chart(monthly, "bar", "order_date__month", ["revenue", "order_count"], series="product_category")


def test_chart_needs_a_result():
    with pytest.raises(ChartError, match="Call run_query first"):
        build_chart(None, "bar", "x", ["y"])


def test_pie_rules(monthly):
    with pytest.raises(ChartError, match="exactly one y"):
        build_chart(monthly, "pie", "product_category", ["revenue", "order_count"])


# ------------------------------------------- default chart (agent/charts.py)
from agent.charts import default_chart


def test_default_chart_bar_for_category(engine):
    spec = default_chart(engine.run({"measures": ["revenue"], "dimensions": ["product_category"]}))
    assert spec["layout"]["meta"] == {"chart_type": "bar", "auto": True}


def test_default_chart_line_with_series_for_time_and_category(engine):
    spec = default_chart(engine.run({"measures": ["revenue"], "dimensions": ["order_date__quarter", "product_category"],
                                     "filters": [{"field": "order_date", "op": "=", "value": "2013"}]}))
    assert spec["layout"]["meta"]["chart_type"] == "line" and len(spec["data"]) == 3


def test_no_default_chart_when_too_many_series(engine):
    assert default_chart(engine.run({"measures": ["revenue"],
                                     "dimensions": ["order_date__year", "product_subcategory"]})) is None


# ------------------------------------- replay of the first Llama smoke run
# Queries copied from the logged run (data/app_state.sqlite, 2026-11-05) that the
# compiler rejected. Each should now succeed, through the normal tool path.
LOGGED_REJECTIONS = [
    {"sort": [{"field": "product_category", "dir": "asc"}], "limit": "null", "measures": ["revenue"],
     "dimensions": [{"field": "product_category"}],
     "filters": [{"field": "order_date", "op": "between", "value": ["2013-01-01", "2013-12-31"]}]},
    {"measures": ["revenue"], "dimensions": [{"field": "order_date", "grain": "month"}],
     "filters": [{"field": "order_date", "op": "between", "value": ["2013-01-01", "2013-12-31"]}],
     "sort": [{"field": "order_date", "dir": "asc"}], "limit": 0},
]


@pytest.mark.parametrize("query", LOGGED_REJECTIONS)
def test_logged_llama_rejections_now_run(engine, query):
    out = json.loads(Tools(engine, 50).run_query(query))
    assert out["row_count"] > 0


def test_logged_q1_period_filter_gives_full_year(engine):
    # Llama's first instinct for "in 2013" was = "2013", which used to be rejected.
    out = json.loads(Tools(engine, 50).run_query(
        {"measures": ["revenue"], "filters": [{"field": "order_date", "op": "=", "value": "2013"}]}))
    assert out["rows"] == [{"revenue": 16351550.34}]


def test_logged_q9_two_ranges_now_explained(engine):
    out = Tools(engine, 50).run_query({"measures": ["revenue"], "dimensions": ["order_date__quarter"], "filters": [
        {"field": "order_date", "op": "between", "value": ["2012-01-01", "2012-12-31"]},
        {"field": "order_date", "op": "between", "value": ["2013-01-01", "2013-12-31"]}]})
    assert "combined with AND" in out   # rejected with a fix, instead of silently returning nothing


def test_default_chart_groups_measures_in_the_same_unit(engine):
    # From the app log: "plot total sales and the gross margin side by side as a clustered column chart"
    spec = default_chart(engine.run({"measures": ["revenue", "gross_margin", "gross_margin_pct"],
                                     "dimensions": ["product_category"]}))
    assert [t["name"] for t in spec["data"]] == ["Revenue", "Gross margin"]   # the % is a different unit
    assert spec["layout"]["barmode"] == "group"


def test_agent_requested_multi_measure_bar_is_grouped(engine):
    r = engine.run({"measures": ["revenue", "total_cost"], "dimensions": ["product_category"]})
    assert build_chart(r, "bar", "product_category", ["revenue", "total_cost"])["layout"]["barmode"] == "group"


# ------------------------------------------- follow-up carry-over (2026-11-06 app log)
TOP5_QUERY = {"sort": [{"field": "gross_margin", "dir": "desc"}], "limit": 5,
              "measures": ["revenue", "gross_margin"], "dimensions": ["product_name"]}
FOLLOW_UP_QUERY = {"dimensions": ["product_name"], "measures": ["revenue", "gross_margin"],
                   "filters": [{"field": "order_date", "op": "=", "value": "2013"}]}


def test_follow_up_that_drops_sort_and_limit_is_warned(engine):
    # "now show those for just 2013" after the top-5 question: Llama's actual query.
    tools = Tools(engine, 50)
    tools.reset(previous_query=TOP5_QUERY)
    note = json.loads(tools.run_query(FOLLOW_UP_QUERY))["note_follow_up"]
    assert "limit 5" in note and "sort gross_margin desc" in note and "same items" in note


def test_follow_up_warning_given_once_per_question(engine):
    tools = Tools(engine, 50)
    tools.reset(previous_query=TOP5_QUERY)
    tools.run_query(FOLLOW_UP_QUERY)
    assert "note_follow_up" not in json.loads(tools.run_query(FOLLOW_UP_QUERY))


def test_follow_up_that_keeps_everything_is_not_warned(engine):
    tools = Tools(engine, 50)
    tools.reset(previous_query=TOP5_QUERY)
    corrected = dict(TOP5_QUERY, filters=FOLLOW_UP_QUERY["filters"])
    assert "note_follow_up" not in json.loads(tools.run_query(corrected))


def test_follow_up_dropping_a_filter_is_warned(engine):
    tools = Tools(engine, 50)
    tools.reset(previous_query={"measures": ["revenue"], "filters": [{"field": "product_category", "op": "=", "value": "Bikes"}]})
    note = json.loads(tools.run_query({"measures": ["revenue"], "dimensions": ["order_date__year"]}))["note_follow_up"]
    assert 'filter product_category = "Bikes"' in note


def test_no_previous_query_no_warning(engine):
    assert "note_follow_up" not in json.loads(Tools(engine, 50).run_query(FOLLOW_UP_QUERY))


@pytest.mark.parametrize("bare", ["2013", 2013, "2013-Q2", "2013-03", "2013-03-15"])
def test_bare_period_filter_becomes_time_filter(bare):
    out = normalise_query_args({"measures": ["revenue"], "filters": [bare]}, time_dimension="order_date")
    assert out["filters"] == [{"field": "order_date", "op": "=", "value": str(bare)}]


def test_bare_text_filter_left_for_compiler():
    assert normalise_query_args({"measures": ["revenue"], "filters": ["Bikes"]}, "order_date")["filters"] == ["Bikes"]


def test_logged_bare_2013_filter_now_runs(engine):
    # Llama's first attempt at the follow-up, rejected in the logged run.
    out = json.loads(Tools(engine, 50).run_query(
        {"measures": ["revenue", "gross_margin"], "dimensions": ["product_name"], "filters": ["2013"]}))
    assert out["row_count"] > 0
