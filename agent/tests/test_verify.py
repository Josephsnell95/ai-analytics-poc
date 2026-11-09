import pytest

from agent.verify import unverified_figures

CATEGORY_ROWS = [
    {"product_category": "Accessories", "revenue": 700759.96, "gross_margin": 438674.57},
    {"product_category": "Bikes", "revenue": 28318144.65, "gross_margin": 11505796.5},
    {"product_category": "Clothing", "revenue": 339772.61, "gross_margin": 136412.58},
]
FMT = {"product_category": None, "revenue": "currency", "gross_margin": "currency"}


def test_logged_llama_misreading_is_caught():
    # The real answer from the app log, 2026-11-05: Bikes margin is 11,505,796.50, not 11,506,197.
    answer = ("The total sales by product category are: Accessories ($700,760), Bikes ($28,318,145), "
              "and Clothing ($339,773). The gross margin by product category is: Accessories ($438,675), "
              "Bikes ($11,506,197), and Clothing ($136,413).")
    assert unverified_figures(answer, CATEGORY_ROWS, FMT) == ["$11,506,197"]


@pytest.mark.parametrize("figure", [
    "$28,318,144.65", "$28,318,145", "$28.3 million", "$28.32M", "$28m", "$0.7 million", "$700,760",
    "$439k", "28,318,145",
])
def test_figures_at_written_precision_match(figure):
    assert unverified_figures(f"Revenue was {figure}.", CATEGORY_ROWS, FMT) == []


@pytest.mark.parametrize("figure", ["$28.4 million", "$700,761", "$29,358,677", "12.4%"])
def test_figures_not_in_the_data_are_flagged(figure):
    assert unverified_figures(f"Revenue was {figure}.", CATEGORY_ROWS, FMT) == [figure]


def test_percent_measure_written_as_percentage():
    rows = [{"product_subcategory": "Touring Bikes", "gross_margin_pct": 0.378399994}]
    fmt = {"product_subcategory": None, "gross_margin_pct": "percent"}
    assert unverified_figures("Touring Bikes has the lowest margin at 37.8%.", rows, fmt) == []
    assert unverified_figures("Touring Bikes has the lowest margin at 37.84%.", rows, fmt) == []
    assert unverified_figures("Touring Bikes has the lowest margin at 38.5%.", rows, fmt) == ["38.5%"]


def test_digits_inside_dimension_values_are_ignored():
    rows = [{"product_name": "Mountain-200 Black, 46", "gross_margin": 626621.5735}]
    answer = 'Your most profitable item is the "Mountain-200 Black, 46" with a gross margin of $626,621.57.'
    assert unverified_figures(answer, rows, {"product_name": None, "gross_margin": "currency"}) == []


def test_years_quarters_dates_list_markers_and_small_counts_ignored():
    rows = [{"order_date__quarter": "2013-01-01", "revenue": 2678946.04}]
    answer = ("1. In 2013 Q1 (from 2013-01-01) revenue was $2,678,946.04, the first of 4 quarters "
              "and the top 3 by growth.")
    assert unverified_figures(answer, rows, {"order_date__quarter": None, "revenue": "currency"}) == []


def test_counts_checked():
    rows = [{"customer_count": 3532}]
    assert unverified_figures("3,532 customers bought in 2013.", rows, {"customer_count": "number"}) == []
    assert unverified_figures("There were 152 customers.", rows, {"customer_count": "number"}) == ["152"]


def test_no_rows_means_nothing_to_check():
    assert unverified_figures("Revenue was $5.", [], {}) == []


def test_truncated_figure_matches():
    rows = [{"revenue": 668241.53}]
    assert unverified_figures("Accessories made $668,241.", rows, {"revenue": "currency"}) == []
    assert unverified_figures("Accessories made $668,240.", rows, {"revenue": "currency"}) == ["$668,240"]


@pytest.mark.parametrize("phrase", ["1 January to December 31", "31st December", "Dec 31", "from 15 March"])
def test_days_of_month_ignored(phrase):
    assert unverified_figures(f"Revenue was $5 over {phrase}.", [{"revenue": 5.0}], {"revenue": "currency"}) == []


# ------------------------------------------- from the first eval run (2026-11-06)
@pytest.mark.parametrize("answer, rows, fmt", [
    ("The gross margin percentage on clothing is 40.2%.",                       # 40.148%: last-digit slip
     [{"gross_margin_pct": 0.40148196848474627}], {"gross_margin_pct": "percent"}),
    ('"No Discount" brought in $27,307,608 in revenue.',                        # 27,307,607.08
     [{"promotion_name": "No Discount", "revenue": 27307607.0825}], {"promotion_name": None, "revenue": "currency"}),
    ("The Touring-1000 and Touring-3000 promotions contributed far less, at $30,992.91 and $14,847.00.",
     [{"promotion_name": "Touring-1000 Promotion", "revenue": 30992.91},
      {"promotion_name": "Touring-3000 Promotion", "revenue": 14847.0}], {"promotion_name": None, "revenue": "currency"}),
    ("The first week counts only 1–6 January ($156,777.83) and the last only 28–31 January.",
     [{"order_date__week": "2012-12-31", "revenue": 156777.83}], {"order_date__week": None, "revenue": "currency"}),
])
def test_logged_false_positives_now_pass(answer, rows, fmt):
    assert unverified_figures(answer, rows, fmt) == []


@pytest.mark.parametrize("answer, rows, flagged", [
    ("Bachelors brought in $5,628,848.91.", [{"revenue": 5625848.91}], ["$5,628,848.91"]),   # digit misread
    ("The revenue for 2012 was $5,983,419.41.", [{"revenue": 624502.17}], ["$5,983,419.41"]), # invented total
])
def test_logged_real_errors_still_flagged(answer, rows, flagged):
    assert unverified_figures(answer, rows, {"revenue": "currency"}) == flagged
