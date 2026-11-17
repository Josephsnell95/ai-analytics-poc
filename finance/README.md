# Finance dataset (milestone F1)

Fictitious accounts receivable data shaped like SAP Business One, for the finance version of the PoC: invoices, credit notes (based on the invoice, standalone with a typed reference, or cancellations), payments allocated on receipt or on account, and the internal reconciliations that match them later. A few hundred standalone credits and payments on account are still waiting to be matched today: the backlog the matching milestone works on. The design and the reasons behind it are in the Claude doc "Finance PoC — Design".

```
python -m finance.generate     # writes data/finance.duckdb (about 10 seconds)
python -m finance.report       # writes finance/data_profile.md: what the data looks like
pytest finance/tests           # the reconciliation rules
```

| File | What it is |
| --- | --- |
| `generator.yaml` | Every assumption: customers, groups, terms, payer types, credit notes, seasonality. Change a number, regenerate, rerun the tests. |
| `generate.py` | The seeded generator. Same seed, same data. Amounts are whole pence until written. |
| `names.py` | Invented customer, supplier and item names. |
| `schema.sql` | The B1-shaped tables: OCRD (with WTLiable, WTCode and a U_CIS_Contractor user-defined field), OCRG, OCTG, OSLP, OWHT, OITB, OITM, OINV/INV1, ORIN/RIN1, ORCT/RCT2, OITR/ITR1. `_gen_` tables are not B1: each customer's hidden payer type, and the true match behind every standalone credit and payment on account (`_gen_match`), for tests, the report and scoring the matcher. |
| `views.sql` | Derived facts at clean grains (`ar_customer`, `ar_invoice`, `ar_credit_note`, `ar_receipt`, `ar_allocation`, `ar_sales_line`, `ar_ledger`, `ar_reconciliation`), and the as-at macros (`ar_open_items`, `ar_unallocated_receipts`, `ar_unallocated`, `ar_balance`, `ar_aged_debt`, `ar_matching_backlog`), which rebuild balances at any date from dated documents, allocations and reconciliations. |
| `report.py`, `data_profile.md` | The profile of the generated data, with the judgement calls to review. |
| `tests/test_reconciliation.py` | Rules 1 to 5 from the design doc at every month-end and today; cancellations net to zero; reconciliations balance and never overreach; the ground truth agrees with what was reconciled; references are as good as their quality says; payer behaviour; determinism. |

Try the as-at logic in DuckDB:

```sql
-- aged debt as at the end of September, per customer
SELECT * FROM ar_aged_debt(DATE '2026-09-30') ORDER BY open_invoices DESC LIMIT 10;
-- what is waiting to be matched today, with whatever was typed in Comments
SELECT * FROM ar_matching_backlog(DATE '2026-10-31') ORDER BY source_date DESC LIMIT 20;
-- one customer's balance at each month-end this year
SELECT d, (SELECT balance FROM ar_balance(d) WHERE customer_code = 'C00286') FROM
  (SELECT CAST(LAST_DAY(m) AS DATE) d FROM generate_series(DATE '2026-01-01', DATE '2026-10-01', INTERVAL 1 MONTH) t(m));
```

The macros are the hand-written reference the semantic-layer compiler has to match in milestone F2. The semantic model and the agent don't use this data yet.
