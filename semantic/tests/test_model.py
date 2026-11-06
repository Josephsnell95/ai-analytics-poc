"""The model is validated at load time. Each test breaks the real model in one way."""

import pytest

from semantic import ModelError, SemanticModel


def errors_for(d):
    with pytest.raises(ModelError) as exc:
        SemanticModel.from_dict(d)
    return " | ".join(exc.value.errors)


def test_real_model_loads(model):
    assert model.fact == "internet_sales"
    assert "revenue" in model.measures and "product_category" in model.dimensions


def test_join_paths_walk_out_from_fact(model):
    path = [join.to for _, join in model.join_paths["category"]]
    assert path == ["product", "subcategory", "category"]
    assert model.join_paths["internet_sales"] == []


def test_unknown_fact(mutable_model_dict):
    mutable_model_dict["fact"] = "sales"
    assert "Fact table 'sales' is not defined" in errors_for(mutable_model_dict)


def test_join_to_undefined_table(mutable_model_dict):
    mutable_model_dict["tables"]["category"]["joins"] = [
        {"to": "department", "from_key": "a", "to_key": "b", "relationship": "many_to_one"}
    ]
    assert "table 'department' is not defined" in errors_for(mutable_model_dict)


def test_only_many_to_one_joins(mutable_model_dict):
    mutable_model_dict["tables"]["internet_sales"]["joins"][0]["relationship"] = "one_to_many"
    assert "only many_to_one is supported" in errors_for(mutable_model_dict)


def test_table_reachable_twice_is_ambiguous(mutable_model_dict):
    # A second route to category (directly from the fact) makes the join path ambiguous.
    mutable_model_dict["tables"]["internet_sales"]["joins"].append(
        {"to": "category", "from_key": "X", "to_key": "Y", "relationship": "many_to_one"}
    )
    assert "reachable by more than one join path" in errors_for(mutable_model_dict)


def test_unreachable_table(mutable_model_dict):
    mutable_model_dict["tables"]["reseller"] = {"source": "DimReseller"}
    assert "Table 'reseller' cannot be reached" in errors_for(mutable_model_dict)


def test_dimension_on_undefined_table(mutable_model_dict):
    mutable_model_dict["dimensions"]["product_color"]["table"] = "colour"
    assert "Dimension 'product_color': table 'colour' is not defined" in errors_for(mutable_model_dict)


def test_expression_sql_must_use_table_placeholder(mutable_model_dict):
    mutable_model_dict["dimensions"]["customer_name"]["sql"] = "FirstName || ' ' || LastName"
    assert "expression using {TABLE}" in errors_for(mutable_model_dict)


def test_grains_only_on_time_dimensions(mutable_model_dict):
    mutable_model_dict["dimensions"]["product_color"]["grains"] = ["month"]
    assert "only time dimensions can have grains" in errors_for(mutable_model_dict)


def test_name_used_as_dimension_and_measure(mutable_model_dict):
    mutable_model_dict["dimensions"]["revenue"] = {"table": "internet_sales", "sql": "SalesAmount"}
    assert "'revenue' is defined as both" in errors_for(mutable_model_dict)


def test_sum_on_dimension_table_rejected(mutable_model_dict):
    # Summing product list price at fact grain would repeat it per order line.
    mutable_model_dict["measures"]["list_price"] = {"table": "product", "sql": "ListPrice", "agg": "sum"}
    assert "would be repeated once per fact row" in errors_for(mutable_model_dict)


def test_count_distinct_on_dimension_table_allowed(mutable_model_dict):
    mutable_model_dict["measures"]["product_count"] = {
        "table": "product", "sql": "ProductKey", "agg": "count_distinct"
    }
    assert "product_count" in SemanticModel.from_dict(mutable_model_dict).measures


def test_unknown_aggregation(mutable_model_dict):
    mutable_model_dict["measures"]["revenue"]["agg"] = "median"
    assert "agg 'median' must be one of" in errors_for(mutable_model_dict)


def test_derived_measure_unknown_reference(mutable_model_dict):
    mutable_model_dict["measures"]["gross_margin"]["sql"] = "revenue - cogs"
    assert "references unknown measures ['cogs']" in errors_for(mutable_model_dict)


def test_derived_measure_cycle(mutable_model_dict):
    m = mutable_model_dict["measures"]
    m["a"] = {"type": "derived", "sql": "b + revenue"}
    m["b"] = {"type": "derived", "sql": "a * 2"}
    assert "cycle" in errors_for(mutable_model_dict)


def test_duplicate_synonym(mutable_model_dict):
    mutable_model_dict["measures"]["total_cost"]["synonyms"].append("sales")
    assert "Synonym 'sales' is used by both" in errors_for(mutable_model_dict)


def test_question_mark_reserved(mutable_model_dict):
    mutable_model_dict["dimensions"]["customer_name"]["sql"] = "'?' || {TABLE}.FirstName"
    assert "must not contain '?'" in errors_for(mutable_model_dict)


def test_all_errors_reported_together(mutable_model_dict):
    mutable_model_dict["measures"]["revenue"]["agg"] = "median"
    mutable_model_dict["dimensions"]["product_color"]["table"] = "colour"
    msg = errors_for(mutable_model_dict)
    assert "median" in msg and "colour" in msg


def test_derived_measure_expands_to_base_aggregates(model):
    assert model.measure_sql("average_order_value") == (
        "SUM(internet_sales.SalesAmount) / NULLIF(COUNT(DISTINCT internet_sales.SalesOrderNumber), 0)"
    )
    assert model.measure_tables("gross_margin_pct") == {"internet_sales"}


# ------------------------------------------------------------ not_available
@pytest.mark.parametrize("question, topic", [
    ("What were our reseller sales last year?", "Reseller sales"),
    ("RESELLER revenue by year", "Reseller sales"),
    ("Revenue by fiscal year", "Fiscal periods"),
    ("Revenue in FY 2013", "Fiscal periods"),
    ("Did we hit target in 2013?", "Targets and quotas"),
    ("Which salesperson sold the most?", "Employees and sales reps"),
    ("Forecast revenue for 2014", "Forecasts"),
])
def test_unavailable_topics_detected(model, question, topic):
    assert model.unavailable_topic(question).topic == topic


@pytest.mark.parametrize("question", [
    "Show me sales by product category for 2013",
    "Top customers by revenue",
    "Revenue for Road bikes",
    "What is the stocking pattern of orders",   # whole words only: "stocking" isn't "in stock"
    "Revenue in the Pacific territory",
])
def test_answerable_questions_not_blocked(model, question):
    assert model.unavailable_topic(question) is None


def test_blocked_term_cannot_be_a_field_or_synonym(mutable_model_dict):
    mutable_model_dict["not_available"].append({"topic": "Oops", "terms": ["turnover"], "reason": "x"})
    assert "term 'turnover' is a field or synonym" in errors_for(mutable_model_dict)


def test_not_available_needs_terms(mutable_model_dict):
    mutable_model_dict["not_available"].append({"topic": "Empty", "terms": [], "reason": "x"})
    assert "terms must be a non-empty list" in errors_for(mutable_model_dict)
