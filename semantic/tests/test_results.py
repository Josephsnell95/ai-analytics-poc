"""Hand-written semantic queries against the real database, each checked against
SQL written independently by hand. If these pass, a wrong answer later on is the
agent's fault, not the compiler's."""

import pytest


def _as_float(v):
    # Decimal and int both compare as float, so SUM(money) vs a hand-written ratio
    # don't fail on type alone.
    return float(v) if isinstance(v, (int, float)) or hasattr(v, "as_tuple") else v


def check(engine, semantic_query, reference_sql):
    got = engine.run(semantic_query).rows
    want = engine.con.execute(reference_sql).fetchall()
    assert len(got) == len(want), f"{len(got)} rows vs {len(want)} expected"
    for g, w in zip(got, want):
        assert len(g) == len(w)
        for a, b in zip(map(_as_float, g), map(_as_float, w)):
            if isinstance(a, float) and isinstance(b, float):
                assert a == pytest.approx(b, rel=1e-9)
            else:
                assert a == b
    return got


def test_total_revenue_matches_known_figure(engine):
    rows = engine.run({"measures": ["revenue"]}).rows
    assert float(rows[0][0]) == pytest.approx(29_358_677.2207)


def test_revenue_and_orders_by_category_2013(engine):
    check(engine,
          {"measures": ["revenue", "order_count"], "dimensions": ["product_category"],
           "filters": [{"field": "order_date", "op": "between", "value": ["2013-01-01", "2013-12-31"]}]},
          """
          SELECT c.EnglishProductCategoryName, SUM(f.SalesAmount), COUNT(DISTINCT f.SalesOrderNumber)
          FROM FactInternetSales f
          JOIN DimProduct p ON f.ProductKey = p.ProductKey
          JOIN DimProductSubcategory s ON p.ProductSubcategoryKey = s.ProductSubcategoryKey
          JOIN DimProductCategory c ON s.ProductCategoryKey = c.ProductCategoryKey
          JOIN DimDate d ON f.OrderDateKey = d.DateKey
          WHERE d.CalendarYear = 2013
          GROUP BY 1 ORDER BY 1
          """)


def test_top_5_subcategories_by_margin_pct(engine):
    check(engine,
          {"measures": ["gross_margin_pct"], "dimensions": ["product_subcategory"],
           "sort": [{"field": "gross_margin_pct", "dir": "desc"}], "limit": 5},
          """
          SELECT s.EnglishProductSubcategoryName,
                 (SUM(f.SalesAmount) - SUM(f.TotalProductCost)) / SUM(f.SalesAmount) AS m
          FROM FactInternetSales f
          JOIN DimProduct p ON f.ProductKey = p.ProductKey
          JOIN DimProductSubcategory s ON p.ProductSubcategoryKey = s.ProductSubcategoryKey
          GROUP BY 1 ORDER BY m DESC, 1 LIMIT 5
          """)


def test_quarterly_revenue_grain(engine):
    check(engine,
          {"measures": ["revenue"], "dimensions": [{"field": "order_date", "grain": "quarter"}],
           "filters": [{"field": "order_date", "op": ">=", "value": "2012-01-01"},
                       {"field": "order_date", "op": "<", "value": "2014-01-01"}]},
          """
          SELECT make_date(d.CalendarYear, (d.CalendarQuarter - 1) * 3 + 1, 1), SUM(f.SalesAmount)
          FROM FactInternetSales f JOIN DimDate d ON f.OrderDateKey = d.DateKey
          WHERE d.CalendarYear IN (2012, 2013)
          GROUP BY 1 ORDER BY 1
          """)


def test_customer_geography_snowflake(engine):
    check(engine,
          {"measures": ["customer_count", "revenue_per_customer"], "dimensions": ["customer_country"]},
          """
          SELECT g.EnglishCountryRegionName, COUNT(DISTINCT f.CustomerKey),
                 SUM(f.SalesAmount) / COUNT(DISTINCT f.CustomerKey)
          FROM FactInternetSales f
          JOIN DimCustomer c ON f.CustomerKey = c.CustomerKey
          JOIN DimGeography g ON c.GeographyKey = g.GeographyKey
          GROUP BY 1 ORDER BY 1
          """)


def test_having_filter_on_measure(engine):
    check(engine,
          {"measures": ["revenue"], "dimensions": ["product_model"],
           "filters": [{"field": "revenue", "op": ">", "value": 1_000_000}],
           "sort": [{"field": "revenue", "dir": "desc"}]},
          """
          SELECT p.ModelName, SUM(f.SalesAmount) r
          FROM FactInternetSales f JOIN DimProduct p ON f.ProductKey = p.ProductKey
          GROUP BY 1 HAVING SUM(f.SalesAmount) > 1000000 ORDER BY r DESC, 1
          """)


def test_in_filter_and_expression_dimension(engine):
    check(engine,
          {"measures": ["units_sold"], "dimensions": ["product_line", "customer_gender"],
           "filters": [{"field": "sales_territory_group", "op": "in", "value": ["Europe", "Pacific"]}]},
          """
          SELECT CASE TRIM(p.ProductLine) WHEN 'R' THEN 'Road' WHEN 'M' THEN 'Mountain'
                   WHEN 'T' THEN 'Touring' WHEN 'S' THEN 'Standard' END,
                 CASE c.Gender WHEN 'M' THEN 'Male' WHEN 'F' THEN 'Female' END,
                 SUM(f.OrderQuantity)
          FROM FactInternetSales f
          JOIN DimProduct p ON f.ProductKey = p.ProductKey
          JOIN DimCustomer c ON f.CustomerKey = c.CustomerKey
          JOIN DimSalesTerritory t ON f.SalesTerritoryKey = t.SalesTerritoryKey
          WHERE t.SalesTerritoryGroup IN ('Europe', 'Pacific')
          GROUP BY 1, 2 ORDER BY 1, 2
          """)


def test_contains_filter_case_insensitive(engine):
    check(engine,
          {"measures": ["revenue"], "dimensions": ["product_subcategory"],
           "filters": [{"field": "product_subcategory", "op": "contains", "value": "BIKE"}]},
          """
          SELECT s.EnglishProductSubcategoryName, SUM(f.SalesAmount)
          FROM FactInternetSales f
          JOIN DimProduct p ON f.ProductKey = p.ProductKey
          JOIN DimProductSubcategory s ON p.ProductSubcategoryKey = s.ProductSubcategoryKey
          WHERE s.EnglishProductSubcategoryName ILIKE '%bike%'
          GROUP BY 1 ORDER BY 1
          """)


def test_dimension_only_query_lists_values_present_in_sales(engine):
    rows = engine.run({"dimensions": ["sales_territory_group"]}).rows
    assert rows == [("Europe",), ("North America",), ("Pacific",)]  # the "NA" territory has no sales


def test_injection_attempt_is_just_a_value(engine):
    # Rejected as an unknown value before anything runs; on a dimension too large to
    # check, it's a bound parameter that simply matches nothing.
    from semantic import QueryValidationError
    with pytest.raises(QueryValidationError, match="is not a value of product_category"):
        engine.run({"measures": ["revenue"], "filters": [
            {"field": "product_category", "op": "=", "value": "x'; DROP TABLE FactInternetSales; --"}]})
    rows = engine.run({"measures": ["revenue"], "filters": [
        {"field": "customer_city", "op": "=", "value": "x'; DROP TABLE FactInternetSales; --"}]}).rows
    assert rows == [(None,)]
    assert engine.con.execute("SELECT COUNT(*) FROM FactInternetSales").fetchone()[0] == 60398


# ------------------------------------------------------------ filter values
from semantic import QueryValidationError  # noqa: E402


def _error(engine, field, value, op="="):
    with pytest.raises(QueryValidationError) as e:
        engine.run({"measures": ["revenue"], "filters": [{"field": field, "op": op, "value": value}]})
    return " | ".join(e.value.errors)


def test_value_from_another_dimension_is_redirected(engine):
    # First eval run: Llama filtered customer_country = "Europe" and got nothing back.
    assert "It is a value of sales_territory_group" in _error(engine, "customer_country", "Europe")


def test_wrong_case_suggests_exact_value(engine):
    assert "Did you mean 'Bikes'? Values are case-sensitive" in _error(engine, "product_category", "bikes")


def test_typo_suggests_close_value(engine):
    assert "Did you mean 'Germany'" in _error(engine, "customer_country", "Germny")


def test_unknown_value_lists_valid_ones(engine):
    assert "Valid values: 'Europe', 'NA', 'North America', 'Pacific'" in _error(engine, "sales_territory_group", "Asia")


def test_every_value_in_a_list_is_checked(engine):
    assert "'Narnia' is not a value" in _error(engine, "customer_country", ["France", "Narnia"], op="in")


def test_real_value_with_no_sales_is_valid_and_gives_nothing(engine):
    rows = engine.run({"measures": ["revenue"], "filters": [
        {"field": "product_category", "op": "=", "value": "Components"}]}).rows
    assert rows == [(None,)]


def test_high_cardinality_dimension_not_checked(engine):
    assert engine.dimension_values("customer_city") is None
    assert engine.run({"measures": ["revenue"], "filters": [
        {"field": "customer_city", "op": "=", "value": "Nowhereville"}]}).rows == [(None,)]


def test_contains_filter_not_checked(engine):
    engine.run({"measures": ["revenue"], "filters": [
        {"field": "product_category", "op": "contains", "value": "bik"}]})


def test_records_are_json_friendly(engine):
    rec = engine.run({"measures": ["revenue"], "dimensions": [{"field": "order_date", "grain": "year"}],
                      "limit": 1}).records()
    assert rec == [{"order_date__year": "2010-01-01", "revenue": pytest.approx(43421.0364)}]


def test_year_period_filter_gives_full_year(engine):
    rows = engine.run({"measures": ["revenue"], "filters": [{"field": "order_date", "op": "=", "value": "2013"}]}).rows
    assert float(rows[0][0]) == pytest.approx(16_351_550.34)


def test_quarter_comparison_with_period_between(engine):
    check(engine,
          {"measures": ["revenue"], "dimensions": ["order_date__quarter"],
           "filters": [{"field": "order_date", "op": "between", "value": ["2012", "2013"]}]},
          """
          SELECT make_date(d.CalendarYear, (d.CalendarQuarter - 1) * 3 + 1, 1), SUM(f.SalesAmount)
          FROM FactInternetSales f JOIN DimDate d ON f.OrderDateKey = d.DateKey
          WHERE d.CalendarYear IN (2012, 2013) GROUP BY 1 ORDER BY 1
          """)


def test_not_equal_period_excludes_year(engine):
    total = float(engine.run({"measures": ["revenue"]}).rows[0][0])
    not_2013 = float(engine.run({"measures": ["revenue"], "filters": [
        {"field": "order_date", "op": "!=", "value": "2013"}]}).rows[0][0])
    assert total - not_2013 == pytest.approx(16_351_550.34)
