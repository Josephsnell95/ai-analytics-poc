"""The reconciliation rules from the design doc, run against a freshly generated dataset.

Rules 1 and 2 are checked at every month-end in the history and at today. Where a
rule can be checked two ways, the test uses the raw B1-shaped tables for one side
and the derived views or as-at macros for the other, so a bug in the views can't
hide behind itself.
"""

import hashlib
import re
from datetime import date

import duckdb
import pytest

from finance.generate import generate, load_settings

TODAY = date(2026, 10, 31)


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    path = tmp_path_factory.mktemp("finance") / "finance.duckdb"
    generate(path)
    c = duckdb.connect(str(path), read_only=True)
    yield c
    c.close()


def rows(con, sql, *params):
    return con.execute(sql, params).fetchall()


def as_at_dates(con):
    """Every month-end in the history, plus today."""
    ends = [r[0] for r in rows(con, """
        SELECT CAST(LAST_DAY(d) AS DATE) FROM generate_series(DATE '2023-11-01', DATE '2026-10-01', INTERVAL 1 MONTH) t(d)""")]
    return sorted(set(ends) | {TODAY})


# ------------------------------------------------------------------ rule 1
def test_balance_from_raw_documents_matches_ledger_at_every_month_end(con):
    for d in as_at_dates(con):
        mismatches = rows(con, """
            WITH raw AS (
                SELECT CardCode, SUM(amt) AS bal FROM (
                    SELECT CardCode, DocTotal AS amt FROM OINV WHERE DocDate <= ? AND CANCELED = 'N'
                    UNION ALL SELECT CardCode, -DocTotal FROM ORIN WHERE DocDate <= ? AND CANCELED = 'N'
                    UNION ALL SELECT CardCode, -DocTotal FROM ORCT WHERE DocDate <= ? AND Canceled = 'N')
                GROUP BY CardCode)
            SELECT COALESCE(r.CardCode, b.customer_code), r.bal, b.balance
            FROM raw r FULL JOIN ar_balance(CAST(? AS DATE)) b ON b.customer_code = r.CardCode
            WHERE COALESCE(r.bal, 0) <> COALESCE(b.balance, 0)""", d, d, d, d)
        assert mismatches == [], f"as at {d}: {mismatches[:5]}"


# ------------------------------------------------------------------ rule 2
def test_open_items_less_cash_on_account_equal_balance_at_every_month_end(con):
    """Open invoices, less unmatched credits, less unallocated cash, is what the customer owes."""
    for d in as_at_dates(con):
        mismatches = rows(con, """
            WITH items AS (SELECT customer_code, SUM(open_amount) AS open FROM ar_open_items(CAST(? AS DATE)) GROUP BY 1)
            SELECT b.customer_code, b.balance, i.open, u.unallocated
            FROM ar_balance(CAST(? AS DATE)) b
            LEFT JOIN items i USING (customer_code)
            LEFT JOIN ar_unallocated(CAST(? AS DATE)) u USING (customer_code)
            WHERE b.balance <> COALESCE(i.open, 0) - COALESCE(u.unallocated, 0)""", d, d, d)
        assert mismatches == [], f"as at {d}: {mismatches[:5]}"


def test_nothing_is_ever_over_settled(con):
    for d in as_at_dates(con):
        assert rows(con, """SELECT COUNT(*) FROM ar_open_items(CAST(? AS DATE))
                            WHERE (document_type = 'Invoice' AND (open_amount < 0 OR open_amount > gross_amount))
                               OR (document_type = 'Credit note' AND (open_amount > 0 OR open_amount < gross_amount))""",
                    d) == [(0,)], d
        assert rows(con, "SELECT COUNT(*) FROM ar_unallocated_receipts(CAST(? AS DATE)) WHERE unallocated < 0", d) == [(0,)], d


# ------------------------------------------------------------------ rule 3
def test_payments_allocate_exactly_what_they_received(con):
    bad = rows(con, """
        SELECT r.DocEntry FROM ORCT r LEFT JOIN (SELECT DocNum, SUM(SumApplied) s FROM RCT2 GROUP BY 1) a ON a.DocNum = r.DocEntry
        WHERE r.DocTotal <> COALESCE(a.s, 0) + r.NoDocSum OR r.DocTotal <> r.TrsfrSum""")
    assert bad == []


def test_reconciliations_balance_and_never_overreach(con):
    unbalanced = rows(con, """SELECT ReconNum FROM ITR1 GROUP BY 1
                              HAVING SUM(ReconSum) FILTER (WHERE IsCredit = 'C') <> SUM(ReconSum) FILTER (WHERE IsCredit = 'D')""")
    assert unbalanced == []
    assert rows(con, "SELECT COUNT(*) FROM OITR h WHERE Total <> (SELECT SUM(ReconSum) FROM ITR1 l "
                     "WHERE l.ReconNum = h.ReconNum AND IsCredit = 'C')") == [(0,)]
    # a credit or payment is never matched for more than it is worth, nor before it exists
    over = rows(con, """
        SELECT l.SrcObjTyp, l.SrcObjAbs FROM ITR1 l JOIN OITR h USING (ReconNum)
        LEFT JOIN ORIN c ON l.SrcObjTyp = '14' AND c.DocEntry = l.SrcObjAbs
        LEFT JOIN ORCT r ON l.SrcObjTyp = '24' AND r.DocEntry = l.SrcObjAbs
        LEFT JOIN OINV i ON l.SrcObjTyp = '13' AND i.DocEntry = l.SrcObjAbs
        GROUP BY ALL
        HAVING SUM(l.ReconSum) > MAX(COALESCE(c.DocTotal, r.NoDocSum, i.DocTotal))
            OR MIN(h.ReconDate) < MAX(COALESCE(c.DocDate, r.DocDate, i.DocDate))
            OR COUNT(DISTINCT l.ShortName) > 1
            OR MAX(l.ShortName) <> MAX(COALESCE(c.CardCode, r.CardCode, i.CardCode))""")
    assert over == []


def test_every_document_settles_against_its_own_customer(con):
    assert rows(con, "SELECT COUNT(*) FROM ar_allocation a JOIN ar_invoice i USING (invoice_entry) "
                     "WHERE a.customer_code <> i.customer_code OR a.allocation_date <= i.invoice_date") == [(0,)]


def test_nothing_dated_after_today_or_on_a_weekend(con):
    for table in ("OINV", "ORIN", "ORCT"):
        assert rows(con, f"SELECT COUNT(*) FROM {table} WHERE DocDate > ? OR ISODOW(DocDate) > 5", TODAY) == [(0,)], table
    assert rows(con, "SELECT COUNT(*) FROM OITR WHERE ReconDate > ? OR ISODOW(ReconDate) > 5", TODAY) == [(0,)]


# ------------------------------------------------------------------ rule 4
def test_stored_paid_to_date_matches_history(con):
    """PaidToDate and DocStatus describe today; the as-at rebuild must agree with them."""
    bad = rows(con, """
        SELECT i.DocEntry, i.DocTotal, i.PaidToDate, o.open_amount, i.DocStatus
        FROM OINV i
        LEFT JOIN (SELECT * FROM ar_open_items(CAST(? AS DATE)) WHERE document_type = 'Invoice') o
               ON o.document_entry = i.DocEntry
        WHERE i.CANCELED = 'N'
          AND (i.DocTotal - i.PaidToDate <> COALESCE(o.open_amount, 0)
               OR (i.DocStatus = 'C') <> (COALESCE(o.open_amount, 0) = 0))""", TODAY)
    assert bad == [], bad[:5]
    bad = rows(con, """
        SELECT c.DocEntry FROM ORIN c
        LEFT JOIN (SELECT * FROM ar_open_items(CAST(? AS DATE)) WHERE document_type = 'Credit note') o
               ON o.document_entry = c.DocEntry
        WHERE c.CANCELED = 'N'
          AND (c.DocTotal - c.PaidToDate <> -COALESCE(o.open_amount, 0)
               OR (c.DocStatus = 'C') <> (COALESCE(o.open_amount, 0) = 0))""", TODAY)
    assert bad == [], bad[:5]


# ------------------------------------------------------------------ rule 5
@pytest.mark.parametrize("head,lines", [("OINV", "INV1"), ("ORIN", "RIN1")])
def test_header_totals_equal_lines_plus_vat(con, head, lines):
    bad = rows(con, f"""
        SELECT h.DocEntry FROM {head} h JOIN (SELECT DocEntry, SUM(LineTotal) net FROM {lines} GROUP BY 1) l USING (DocEntry)
        WHERE h.DocTotal - h.VatSum <> l.net""")
    assert bad == []
    # VAT per line, rounded half up to the penny, then summed; compared in whole pence
    vat = rows(con, f"""
        SELECT h.DocEntry FROM {head} h JOIN (
            SELECT DocEntry, SUM(CAST(FLOOR((LineTotal * 100 * VatPrcnt + 50) / 100) AS BIGINT)) AS vat_pence
            FROM {lines} GROUP BY 1) l USING (DocEntry)
        WHERE CAST(h.VatSum * 100 AS BIGINT) <> l.vat_pence""")
    assert vat == []


def test_line_totals_are_quantity_times_price(con):
    for table in ("INV1", "RIN1"):
        assert rows(con, f"SELECT COUNT(*) FROM {table} WHERE LineTotal <> Quantity * Price") == [(0,)], table


def test_based_credit_lines_point_at_a_real_line_of_the_same_customers_invoice(con):
    bad = rows(con, """
        SELECT r.DocEntry FROM RIN1 r JOIN ORIN h ON h.DocEntry = r.DocEntry
        LEFT JOIN OINV i ON i.DocEntry = r.BaseEntry
        LEFT JOIN INV1 l ON l.DocEntry = r.BaseEntry AND l.LineNum = r.BaseLine
        WHERE r.BaseType = 13
          AND (i.DocEntry IS NULL OR i.CardCode <> h.CardCode OR l.LineNum IS NULL OR r.Quantity > l.Quantity
               OR r.ItemCode <> l.ItemCode OR h.DocDate < i.DocDate)""")
    assert bad == []
    assert rows(con, "SELECT COUNT(*) FROM RIN1 WHERE BaseType = -1 AND (BaseEntry IS NOT NULL OR BaseLine IS NOT NULL)") == [(0,)]
    mixed = rows(con, "SELECT DocEntry FROM RIN1 GROUP BY 1 HAVING COUNT(DISTINCT BaseType) > 1")
    assert mixed == []


def test_due_dates_follow_payment_terms(con):
    bad = rows(con, """
        SELECT i.DocEntry FROM OINV i JOIN OCRD c USING (CardCode) JOIN OCTG t ON t.GroupNum = c.GroupNum
        WHERE i.DocDueDate <> CASE WHEN t.ExtraMonth = 1 THEN CAST(LAST_DAY(i.DocDate) AS DATE) ELSE i.DocDate END
                              + t.ExtraDays""")
    assert bad == []


def test_document_numbers_unique_and_customers_known(con):
    for table in ("OINV", "ORIN", "ORCT"):
        assert rows(con, f"SELECT COUNT(*) - COUNT(DISTINCT DocNum) FROM {table}") == [(0,)]
        assert rows(con, f"SELECT COUNT(*) FROM {table} t LEFT JOIN OCRD c USING (CardCode) "
                         "WHERE c.CardType IS DISTINCT FROM 'C'") == [(0,)], table


# ------------------------------------------------------------------ cancellations
def test_each_cancelled_invoice_has_one_cancellation_that_nets_to_zero(con):
    bad = rows(con, """
        SELECT i.DocEntry FROM OINV i
        LEFT JOIN (SELECT r.BaseEntry, COUNT(DISTINCT h.DocEntry) n, MAX(h.DocTotal) total, MIN(h.DocDate) d
                   FROM ORIN h JOIN RIN1 r USING (DocEntry) WHERE h.CANCELED = 'C' GROUP BY 1) c ON c.BaseEntry = i.DocEntry
        WHERE i.CANCELED = 'Y' AND (c.n IS DISTINCT FROM 1 OR c.total <> i.DocTotal OR c.d < i.DocDate)""")
    assert bad == []
    assert rows(con, "SELECT COUNT(*) FROM OINV WHERE CANCELED = 'Y'")[0][0] > 0
    # nothing else ever touches a cancelled invoice
    touched = rows(con, """
        SELECT COUNT(*) FROM OINV i WHERE i.CANCELED = 'Y' AND (
            EXISTS (SELECT 1 FROM RCT2 a WHERE a.DocEntry = i.DocEntry)
            OR EXISTS (SELECT 1 FROM ITR1 l WHERE l.SrcObjTyp = '13' AND l.SrcObjAbs = i.DocEntry)
            OR EXISTS (SELECT 1 FROM RIN1 r JOIN ORIN h USING (DocEntry) WHERE r.BaseEntry = i.DocEntry AND h.CANCELED = 'N'))""")
    assert touched == [(0,)]


def test_cancelled_pairs_change_no_balance_or_sales(con):
    """Excluding both halves of a cancellation must give the same answer as including both."""
    with_pairs = rows(con, """
        SELECT SUM(amt) FROM (SELECT DocTotal amt FROM OINV UNION ALL SELECT -DocTotal FROM ORIN WHERE CANCELED = 'C')""")
    cancelled_only = rows(con, "SELECT SUM(DocTotal) FROM OINV WHERE CANCELED = 'Y'")
    all_invoices = rows(con, "SELECT SUM(DocTotal) FROM OINV")
    assert with_pairs[0][0] == all_invoices[0][0] - cancelled_only[0][0]


# ------------------------------------------------------------------ the matching ground truth
def test_ground_truth_agrees_with_what_was_reconciled(con):
    """Every true match reconciled by today appears in ITR1 for that source document, with its amount."""
    missing = rows(con, """
        SELECT m.* FROM _gen_match m
        WHERE m.recon_date <= ?
          AND NOT EXISTS (
            SELECT 1 FROM ITR1 src JOIN ITR1 inv ON inv.ReconNum = src.ReconNum
            WHERE src.SrcObjTyp = CASE m.source_type WHEN 'credit note' THEN '14' ELSE '24' END
              AND src.SrcObjAbs = m.source_entry
              AND inv.SrcObjTyp = '13' AND inv.SrcObjAbs = m.invoice_entry AND inv.ReconSum = m.amount)""", TODAY)
    assert missing == []
    extra = rows(con, """
        SELECT l.ReconNum FROM ITR1 l WHERE l.SrcObjTyp = '13' AND NOT EXISTS (
            SELECT 1 FROM _gen_match m WHERE m.invoice_entry = l.SrcObjAbs AND m.amount = l.ReconSum AND m.recon_date <= ?)""",
                 TODAY)
    assert extra == []


def test_references_are_as_good_as_their_quality_says(con):
    rows_ = rows(con, """
        SELECT m.source_type, m.source_entry, m.reference_quality, COALESCE(c.Comments, r.Comments, '') comments,
               LIST(i.DocNum::VARCHAR) nums, LIST(i.NumAtCard) refs
        FROM _gen_match m JOIN OINV i ON i.DocEntry = m.invoice_entry
        LEFT JOIN ORIN c ON m.source_type = 'credit note' AND c.DocEntry = m.source_entry
        LEFT JOIN ORCT r ON m.source_type = 'receipt' AND r.DocEntry = m.source_entry
        GROUP BY ALL""")
    seen = set()
    for kind, entry, quality, comment, nums, refs in rows_:
        found = set(re.findall(r"\d{5,}", comment))
        seen.add(quality)
        if quality == "clean":
            assert set(nums) <= found, (kind, entry, comment, nums)
        elif quality == "typo":
            assert not set(nums) <= found, (kind, entry, comment, nums)
        elif quality == "po_only":
            assert not found & set(nums) and any(ref in comment for ref in refs), (kind, entry, comment)
        else:
            assert not found & set(nums), (kind, entry, comment)
    assert seen == {"clean", "typo", "po_only", "none"}


def test_some_credits_cover_several_invoices_and_some_are_left_over(con):
    multi = rows(con, "SELECT COUNT(*) FROM (SELECT source_entry FROM _gen_match WHERE source_type = 'credit note' "
                      "GROUP BY 1 HAVING COUNT(*) > 1)")[0][0]
    leftover = rows(con, """SELECT COUNT(*) FROM ORIN c JOIN (SELECT source_entry, SUM(amount) s FROM _gen_match
                            WHERE source_type = 'credit note' GROUP BY 1) m ON m.source_entry = c.DocEntry
                            WHERE m.s < c.DocTotal""")[0][0]
    assert multi > 20 and leftover > 20


def test_there_is_a_real_backlog_to_match_today(con):
    backlog = dict(rows(con, "SELECT source_type, COUNT(*) FROM ar_matching_backlog(CAST(? AS DATE)) GROUP BY 1", TODAY))
    assert backlog.get("Credit note", 0) > 30 and backlog.get("Receipt", 0) > 30
    # unmatched credits leave their invoices looking overdue: the pain point
    overdue_but_credited = rows(con, """
        SELECT COUNT(*) FROM _gen_match m JOIN ar_open_items(CAST(? AS DATE)) o
          ON o.document_type = 'Invoice' AND o.document_entry = m.invoice_entry
        WHERE m.source_type = 'credit note' AND (m.recon_date IS NULL OR m.recon_date > ?) AND o.days_overdue > 0""",
                                TODAY, TODAY)[0][0]
    assert overdue_but_credited > 10


# ------------------------------------------------------------------ CIS and withholding tax
def test_cis_flags(con):
    assert rows(con, "SELECT COUNT(*) FROM OCRD WHERE WTLiable = 'Y' AND (CardType <> 'S' OR WTCode IS NULL "
                     "OR WTCode NOT IN (SELECT WTCode FROM OWHT))") == [(0,)]
    assert rows(con, "SELECT COUNT(*) FROM OCRD WHERE WTLiable = 'Y'")[0][0] > 10
    groups = rows(con, "SELECT DISTINCT customer_group FROM ar_customer WHERE cis_contractor")
    assert groups == [("Construction",)]


# ------------------------------------------------------------------ behaviour
def test_payer_types_behave_in_order(con):
    """Median days paid after the due date rise from prompt to slow to chronic payers."""
    med = dict(rows(con, """
        SELECT g.payer_type, MEDIAN(DATE_DIFF('day', i.due_date, a.allocation_date))
        FROM ar_allocation a JOIN ar_invoice i USING (invoice_entry) JOIN _gen_customer g ON g.CardCode = i.customer_code
        GROUP BY 1"""))
    assert med["prompt"] <= 6 and 15 <= med["slow"] <= 45 and 60 <= med["chronic"] <= 120


def test_bad_debts_stop_paying(con):
    assert rows(con, """
        SELECT COUNT(*) FROM ORCT r JOIN _gen_customer g ON g.CardCode = r.CardCode
        WHERE g.payer_type = 'bad_debt' AND r.DocDate >= g.stop_date AND r.NoDocSum = 0""") == [(0,)]


def test_nearly_every_customer_trades_and_some_balances_are_overdue(con):
    # a handful of accounts set up but never traded is realistic; more would be a generator bug
    dormant = rows(con, "SELECT COUNT(*) FROM ar_customer c WHERE NOT EXISTS "
                        "(SELECT 1 FROM ar_invoice i WHERE i.customer_code = c.customer_code)")[0][0]
    assert dormant <= 4
    buckets = dict(rows(con, "SELECT ageing_bucket, COUNT(*) FROM ar_open_items(CAST(? AS DATE)) GROUP BY 1", TODAY))
    assert set(buckets) == {"Not yet due", "1-30 days", "31-60 days", "61-90 days", "Over 90 days", "Unapplied credit"}


# ------------------------------------------------------------------ determinism
def _digest(path):
    c = duckdb.connect(str(path), read_only=True)
    h = hashlib.sha256()
    for table in ("OCRD", "OINV", "INV1", "ORIN", "RIN1", "ORCT", "RCT2", "OITR", "ITR1"):
        for r in c.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall():
            h.update(repr(r).encode())
    c.close()
    return h.hexdigest()


def test_same_seed_same_data_different_seed_different_data(tmp_path):
    small = load_settings() | {"customers": 40, "suppliers": 5, "unidentified_cash": 3}
    small["key_accounts"] = {"count": 1, "multiplier": 8}
    generate(tmp_path / "a.duckdb", settings=small)
    generate(tmp_path / "b.duckdb", settings=small)
    generate(tmp_path / "c.duckdb", seed=7, settings=small)
    assert _digest(tmp_path / "a.duckdb") == _digest(tmp_path / "b.duckdb")
    assert _digest(tmp_path / "a.duckdb") != _digest(tmp_path / "c.duckdb")
