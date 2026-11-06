"""Bad queries are rejected with messages the agent can act on."""

import pytest

from semantic import QueryValidationError, parse_query


def errors_for(q, model, **kw):
    with pytest.raises(QueryValidationError) as exc:
        parse_query(q, model, **kw)
    return " | ".join(exc.value.errors)


def test_valid_query_parses(model):
    q = parse_query(
        {"measures": ["revenue"], "dimensions": [{"field": "order_date", "grain": "year"}]}, model
    )
    assert q.dimensions[0].output_name == "order_date__year"
    assert q.limit == 100


def test_accepts_json_string(model):
    assert parse_query('{"measures": ["revenue"]}', model).measures == ["revenue"]


def test_invalid_json(model):
    assert "not valid JSON" in errors_for("{measures: revenue}", model)


def test_empty_query(model):
    assert "at least one measure or dimension" in errors_for({}, model)


def test_unknown_key_suggests_valid_key(model):
    msg = errors_for({"measures": ["revenue"], "dimension": ["product_category"]}, model)
    assert "Unknown query key 'dimension'" in msg and "'dimensions'" in msg


def test_unknown_measure_lists_close_names(model):
    msg = errors_for({"measures": ["revenu"]}, model)
    assert "Unknown measure 'revenu'" in msg and "'revenue'" in msg


def test_synonym_points_to_real_field(model):
    msg = errors_for({"measures": ["turnover"]}, model)
    assert "synonym of 'revenue'" in msg


def test_dimension_used_as_measure(model):
    assert "is a dimension, not a measure" in errors_for({"measures": ["product_category"]}, model)


def test_measure_used_as_dimension(model):
    assert "is a measure, not a dimension" in errors_for(
        {"measures": ["revenue"], "dimensions": ["order_count"]}, model
    )


def test_unknown_dimension(model):
    msg = errors_for({"measures": ["revenue"], "dimensions": ["product_categry"]}, model)
    assert "'product_category'" in msg


def test_grain_on_non_time_dimension(model):
    msg = errors_for({"dimensions": [{"field": "product_category", "grain": "month"}]}, model)
    assert "not a time dimension" in msg


def test_invalid_grain(model):
    msg = errors_for({"dimensions": [{"field": "order_date", "grain": "fortnight"}]}, model)
    assert "no grain 'fortnight'" in msg and "month" in msg


def test_time_dimension_defaults_to_day(model):
    q = parse_query({"measures": ["revenue"], "dimensions": ["order_date"]}, model)
    assert q.dimensions[0].grain == "day"


def test_duplicate_fields(model):
    msg = errors_for({"measures": ["revenue", "revenue"]}, model)
    assert "listed twice" in msg


def test_same_time_dimension_at_two_grains_is_fine(model):
    q = parse_query(
        {"measures": ["revenue"], "dimensions": [
            {"field": "order_date", "grain": "year"}, {"field": "order_date", "grain": "month"}
        ]}, model)
    assert [d.output_name for d in q.dimensions] == ["order_date__year", "order_date__month"]


def test_filter_unknown_field(model):
    msg = errors_for({"measures": ["revenue"], "filters": [{"field": "colour", "op": "=", "value": "Red"}]}, model)
    assert "synonym of 'product_color'" in msg


def test_filter_bad_op_lists_valid_ops(model):
    msg = errors_for({"measures": ["revenue"], "filters": [{"field": "revenue", "op": "contains", "value": "1"}]}, model)
    assert "not valid for a measure" in msg and "between" in msg


def test_filter_between_needs_two_values(model):
    msg = errors_for({"measures": ["revenue"], "filters": [
        {"field": "order_date", "op": "between", "value": ["2013-01-01"]}]}, model)
    assert "two items" in msg


def test_filter_in_needs_list(model):
    msg = errors_for({"measures": ["revenue"], "filters": [
        {"field": "product_category", "op": "in", "value": "Bikes"}]}, model)
    assert "non-empty list" in msg


def test_filter_bad_date(model):
    msg = errors_for({"measures": ["revenue"], "filters": [
        {"field": "order_date", "op": ">=", "value": "last year"}]}, model)
    assert "dates must be like" in msg


def test_filter_measure_needs_number(model):
    msg = errors_for({"measures": ["revenue"], "filters": [
        {"field": "revenue", "op": ">", "value": "10k"}]}, model)
    assert "must be a number" in msg


def test_filter_missing_value(model):
    msg = errors_for({"measures": ["revenue"], "filters": [{"field": "product_category", "op": "="}]}, model)
    assert "needs a value" in msg


def test_filter_is_null_takes_no_value(model):
    q = parse_query({"measures": ["revenue"], "filters": [{"field": "product_line", "op": "is_null"}]}, model)
    assert q.filters[0].value is None


def test_sort_on_unselected_field(model):
    msg = errors_for({"measures": ["revenue"], "dimensions": ["product_category"],
                      "sort": [{"field": "order_count", "dir": "desc"}]}, model)
    assert "not in the query's measures or dimensions" in msg


def test_sort_bad_direction(model):
    msg = errors_for({"measures": ["revenue"], "sort": [{"field": "revenue", "dir": "down"}]}, model)
    assert "'asc' or 'desc'" in msg


def test_sort_by_time_dimension_name_resolves_to_grain(model):
    q = parse_query({"measures": ["revenue"], "dimensions": [{"field": "order_date", "grain": "month"}],
                     "sort": [{"field": "order_date", "dir": "desc"}]}, model)
    assert q.sort[0].field == "order_date__month"


def test_limit_over_max(model):
    msg = errors_for({"measures": ["revenue"], "limit": 5000}, model, max_limit=1000)
    assert "exceeds the maximum of 1000" in msg


def test_limit_not_positive_int(model):
    assert "positive integer" in errors_for({"measures": ["revenue"], "limit": 0}, model)
    assert "positive integer" in errors_for({"measures": ["revenue"], "limit": "10"}, model)


def test_assumptions_must_be_strings(model):
    assert "list of strings" in errors_for({"measures": ["revenue"], "assumptions": [1]}, model)


def test_all_errors_reported_together(model):
    msg = errors_for({"measures": ["revenu"], "dimensions": ["countree"], "limit": -1}, model)
    assert "revenu" in msg and "countree" in msg and "positive integer" in msg


# ---------------------------------------------------------- date periods
import datetime as dt


def _filter(model, op, value):
    q = parse_query({"measures": ["revenue"], "filters": [{"field": "order_date", "op": op, "value": value}]}, model)
    return q.filters[0].op, q.filters[0].value


@pytest.mark.parametrize("op, value, expected", [
    ("=", "2013", ("between", [dt.date(2013, 1, 1), dt.date(2013, 12, 31)])),
    ("=", 2013, ("between", [dt.date(2013, 1, 1), dt.date(2013, 12, 31)])),
    ("=", "2012-Q2", ("between", [dt.date(2012, 4, 1), dt.date(2012, 6, 30)])),
    ("=", "2012Q4", ("between", [dt.date(2012, 10, 1), dt.date(2012, 12, 31)])),
    ("=", "2012-02", ("between", [dt.date(2012, 2, 1), dt.date(2012, 2, 29)])),   # leap year
    ("!=", "2013", ("not_between", [dt.date(2013, 1, 1), dt.date(2013, 12, 31)])),
    (">", "2012", (">", dt.date(2012, 12, 31))),
    (">=", "2012", (">=", dt.date(2012, 1, 1))),
    ("<", "2013-Q3", ("<", dt.date(2013, 7, 1))),
    ("<=", "2013-03", ("<=", dt.date(2013, 3, 31))),
    ("between", ["2012", "2013"], ("between", [dt.date(2012, 1, 1), dt.date(2013, 12, 31)])),
    ("between", ["2012-Q4", "2013-03-15"], ("between", [dt.date(2012, 10, 1), dt.date(2013, 3, 15)])),
    ("=", "2013-01-01", ("=", dt.date(2013, 1, 1))),   # a full date still means that one day
])
def test_periods_expand_to_date_ranges(model, op, value, expected):
    assert _filter(model, op, value) == expected


def test_period_in_list_rejected(model):
    msg = errors_for({"measures": ["revenue"], "filters": [
        {"field": "order_date", "op": "in", "value": ["2012", "2013"]}]}, model)
    assert "use one 'between'" in msg


def test_bad_month_rejected(model):
    assert "dates must be like" in errors_for({"measures": ["revenue"], "filters": [
        {"field": "order_date", "op": "=", "value": "2013-13"}]}, model)


# ---------------------------------------------------- conflicting filters
def test_two_date_ranges_on_one_field_rejected(model):
    msg = errors_for({"measures": ["revenue"], "filters": [
        {"field": "order_date", "op": "between", "value": ["2012-01-01", "2012-12-31"]},
        {"field": "order_date", "op": "between", "value": ["2013-01-01", "2013-12-31"]}]}, model)
    assert "combined with AND" in msg and "['2012', '2013']" in msg


def test_two_equals_on_text_field_suggests_in(model):
    msg = errors_for({"measures": ["revenue"], "filters": [
        {"field": "customer_country", "op": "=", "value": "France"},
        {"field": "customer_country", "op": "=", "value": "Germany"}]}, model)
    assert "one 'in' filter" in msg


def test_open_ended_range_on_one_field_is_fine(model):
    q = parse_query({"measures": ["revenue"], "filters": [
        {"field": "order_date", "op": ">=", "value": "2012-07-01"},
        {"field": "order_date", "op": "<", "value": "2013"}]}, model)
    assert len(q.filters) == 2


# ------------------------------------------------------ grain shorthand
def test_grain_shorthand_matches_object_form(model):
    a = parse_query({"measures": ["revenue"], "dimensions": ["order_date__month"]}, model)
    b = parse_query({"measures": ["revenue"], "dimensions": [{"field": "order_date", "grain": "month"}]}, model)
    assert a.dimensions == b.dimensions


def test_grain_shorthand_on_non_time_dimension(model):
    assert "not a time dimension" in errors_for({"dimensions": ["product_category__month"]}, model)


def test_grain_shorthand_bad_grain(model):
    assert "no grain 'decade'" in errors_for({"dimensions": ["order_date__decade"]}, model)
