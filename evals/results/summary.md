# Evaluation results

| Set | Model | Accuracy | Passed | Avg time | Cost per question | Rejected queries | Answers with flagged figures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tuning | claude-haiku-5-5 | 97% | 31/32 | 3s | $0.0012 | 1 | 3 |
| tuning | llama3.1:8b | 69% | 22/32 | 71s | free | 8 | 4 |

## By category

| Category | claude-haiku-5-5 (tuning) | llama3.1:8b (tuning) |
| --- | --- | --- |
| ambiguous | 2/2 | 0/2 |
| breakdown | 6/6 | 6/6 |
| follow_up | 2/2 | 1/2 |
| lookup | 5/5 | 4/5 |
| out_of_scope_listed | 2/2 | 2/2 |
| out_of_scope_unlisted | 3/3 | 0/3 |
| ranking | 4/5 | 4/5 |
| synonym | 3/3 | 3/3 |
| time | 4/4 | 2/4 |

## claude-haiku-5-5, tuning set

Run 2026-11-06 16:49 UTC · commit `c2872a4` · prompt `6f711514ed` · model `b112db4ff9` · file `20261106-164918_tuning_claude-haiku-5-5.json`

Failures:

- **ranking_top_promotion** (wrong): 4 rows, expected top 1

To review (is the stated assumption reasonable?):

- **ambiguous_last_year** (correct): "Last year" interpreted as calendar year 2013, the latest full calendar year in the data (orders run to January 2014).; Revenue is invoiced sales value in USD, excluding tax and freight.
- **ambiguous_best_customer** (correct): 'Best customer' means the customer with the highest revenue across all time, since no period was given.

## llama3.1:8b, tuning set

Run 2026-11-06 16:52 UTC · commit `c2872a4` · prompt `6f711514ed` · model `b112db4ff9` · file `20261106-165257_tuning_llama3.1-8b.json`

Failures:

- **lookup_helmets_q3** (wrong): expected row not found: {'units_sold': 1622}
- **ranking_least_profitable_models** (wrong): wrong ranking: gross_margin values [3616725.4874, 2182219.4658, 1685641.3576, 1132175.7855], expected [3196.5336, 4518.8436, 4528.263, 4574.2985]
- **time_yearly_2011_2013** (no_result): no successful query (status capped)
- **time_quarterly_europe_orders** (wrong): different breakdown: grouped by ['customer_country', 'order_date__quarter'], expected ['order_date__quarter']
- **followup_aov_those_countries** (no_result): no successful query (status no_query)
- **ambiguous_last_year** (wrong): expected row not found: {'revenue': 16351550.34}
- **ambiguous_best_customer** (wrong): missing measure(s) revenue
- **scope_unlisted_market_share** (answered_out_of_scope): should have declined but answered
- **scope_unlisted_web_visits** (answered_out_of_scope): should have declined but answered
- **scope_unlisted_net_profit** (answered_out_of_scope): should have declined but answered

To review (is the stated assumption reasonable?):

- **ambiguous_last_year** (wrong): last year means calendar year 2013
- **ambiguous_best_customer** (wrong): no assumption stated
