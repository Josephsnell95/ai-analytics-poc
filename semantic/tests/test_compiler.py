"""SQL generation rules, checked on the SQL text. No database needed."""

from semantic import compile


def sql_for(model, q, **kw):
    return compile(model, q, **kw)


def test_measure_only_query_has_no_joins_or_group_by(model):
    c = sql_for(model, {"measures": ["revenue"]})
    assert "JOIN" not in c.sql and "GROUP BY" not in c.sql
    assert "SUM(internet_sales.SalesAmount)" in c.sql


def test_only_needed_tables_are_joined(model):
    c = sql_for(model, {"measures": ["revenue"], "dimensions": ["customer_country"]})
    assert "AS customer ON" in c.sql and "AS customer_geography ON" in c.sql
    assert "DimProduct" not in c.sql and "DimDate" not in c.sql


def test_snowflake_chain_joined_in_order(model):
    c = sql_for(model, {"measures": ["revenue"], "dimensions": ["product_category"]})
    i_prod, i_sub, i_cat = (c.sql.index(f"AS {t} ON") for t in ("product", "subcategory", "category"))
    assert i_prod < i_sub < i_cat
    assert "ON subcategory.ProductCategoryKey = category.ProductCategoryKey" in c.sql


def test_joins_are_left_joins(model):
    c = sql_for(model, {"measures": ["revenue"], "dimensions": ["product_category"]})
    assert c.sql.count("LEFT JOIN") == c.sql.count("JOIN")


def test_filter_only_field_still_joins_its_table(model):
    c = sql_for(model, {"measures": ["revenue"],
                        "filters": [{"field": "sales_territory_group", "op": "=", "value": "Europe"}]})
    assert "AS sales_territory ON" in c.sql


def test_dimension_filter_in_where_measure_filter_in_having(model):
    c = sql_for(model, {
        "measures": ["revenue"], "dimensions": ["product_category"],
        "filters": [{"field": "product_category", "op": "=", "value": "Bikes"},
                    {"field": "revenue", "op": ">", "value": 1000}],
    })
    where = c.sql[c.sql.index("WHERE"):c.sql.index("GROUP BY")]
    having = c.sql[c.sql.index("HAVING"):]
    assert "category.EnglishProductCategoryName = ?" in where
    assert "SUM(internet_sales.SalesAmount) > ?" in having
    assert c.params == ["Bikes", 1000]


def test_derived_measure_filter_expands_in_having(model):
    c = sql_for(model, {"measures": ["revenue"], "dimensions": ["product_subcategory"],
                        "filters": [{"field": "gross_margin_pct", "op": ">=", "value": 0.4}]})
    assert "HAVING (SUM(internet_sales.SalesAmount) - SUM(internet_sales.TotalProductCost))" in c.sql


def test_time_grain_uses_date_trunc(model):
    c = sql_for(model, {"measures": ["revenue"], "dimensions": [{"field": "order_date", "grain": "quarter"}]})
    assert "CAST(DATE_TRUNC('quarter', order_date.FullDateAlternateKey) AS DATE) AS \"order_date__quarter\"" in c.sql


def test_time_filter_uses_raw_date_not_grain(model):
    c = sql_for(model, {"measures": ["revenue"], "dimensions": [{"field": "order_date", "grain": "year"}],
                        "filters": [{"field": "order_date", "op": ">=", "value": "2013-01-01"}]})
    assert "WHERE order_date.FullDateAlternateKey >= ?" in c.sql


def test_derived_measure_expanded(model):
    c = sql_for(model, {"measures": ["gross_margin_pct"]})
    assert ("(SUM(internet_sales.SalesAmount) - SUM(internet_sales.TotalProductCost)) "
            "/ NULLIF(SUM(internet_sales.SalesAmount), 0)") in c.sql


def test_nested_derived_measure(model, mutable_model_dict):
    from semantic import SemanticModel
    mutable_model_dict["measures"]["margin_per_order"] = {"type": "derived", "sql": "gross_margin / NULLIF(order_count, 0)"}
    m = SemanticModel.from_dict(mutable_model_dict)
    c = sql_for(m, {"measures": ["margin_per_order"]})
    assert ("(SUM(internet_sales.SalesAmount) - SUM(internet_sales.TotalProductCost)) "
            "/ NULLIF(COUNT(DISTINCT internet_sales.SalesOrderNumber), 0)") in c.sql


def test_default_limit_always_applied(model):
    assert sql_for(model, {"measures": ["revenue"]}, default_limit=100).sql.endswith("LIMIT 100")


def test_explicit_limit(model):
    assert sql_for(model, {"measures": ["revenue"], "limit": 5}).sql.endswith("LIMIT 5")


def test_order_is_always_fully_specified(model):
    c = sql_for(model, {"measures": ["revenue"], "dimensions": ["product_category", "customer_country"],
                        "sort": [{"field": "revenue", "dir": "desc"}]})
    assert 'ORDER BY "revenue" DESC, "product_category" ASC, "customer_country" ASC' in c.sql


def test_values_are_parameters_not_inlined(model):
    nasty = "Bikes'; DROP TABLE FactInternetSales; --"
    c = sql_for(model, {"measures": ["revenue"],
                        "filters": [{"field": "product_category", "op": "=", "value": nasty}]})
    assert "DROP TABLE" not in c.sql
    assert c.params == [nasty]


def test_in_and_between_parameters(model):
    c = sql_for(model, {"measures": ["revenue"], "filters": [
        {"field": "product_category", "op": "in", "value": ["Bikes", "Clothing"]},
        {"field": "order_date", "op": "between", "value": ["2013-01-01", "2013-12-31"]},
    ]})
    assert "IN (?, ?)" in c.sql and "BETWEEN ? AND ?" in c.sql
    assert len(c.params) == 4


def test_display_sql_inlines_escaped_literals(model):
    c = sql_for(model, {"measures": ["revenue"], "filters": [
        {"field": "customer_city", "op": "=", "value": "Saint-Jean d'Angély"},
        {"field": "order_date", "op": ">=", "value": "2013-01-01"},
    ]})
    shown = c.display_sql()
    assert "'Saint-Jean d''Angély'" in shown and "DATE '2013-01-01'" in shown
    assert "?" not in shown


def test_columns_metadata(model):
    c = sql_for(model, {"measures": ["revenue", "gross_margin_pct"],
                        "dimensions": [{"field": "order_date", "grain": "month"}]})
    assert [(col.name, col.role) for col in c.columns] == [
        ("order_date__month", "dimension"), ("revenue", "measure"), ("gross_margin_pct", "measure")]
    assert c.columns[0].type == "time" and c.columns[0].grain == "month"
    assert c.columns[2].format == "percent"


def test_compiler_is_deterministic(model):
    q = {"measures": ["revenue", "order_count"], "dimensions": ["customer_country", "product_category"]}
    assert sql_for(model, q).sql == sql_for(model, q).sql
