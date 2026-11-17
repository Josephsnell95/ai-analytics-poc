-- Derived facts over the raw B1-shaped tables, and the reference "as at" logic.
--
-- Views turn B1's header/line/allocation tables into facts at clean grains. The
-- table macros rebuild balances and open items as at ANY date from dated
-- documents, allocations and reconciliations, never from PaidToDate or DocStatus,
-- which only describe today. They are the hand-written reference the semantic-layer
-- compiler must match (design doc, milestone F2), and what the reconciliation tests use.
--
-- Cancelled invoices (OINV.CANCELED = Y) and their cancellation documents
-- (ORIN.CANCELED = C) are excluded everywhere: each pair nets to zero.

-- ---------------------------------------------------------------- dimensions
CREATE VIEW ar_customer AS
SELECT c.CardCode               AS customer_code,
       c.CardName               AS customer_name,
       g.GroupName              AS customer_group,
       t.PymntGroup             AS payment_terms,
       s.SlpName                AS account_owner,
       c.Country                AS country,
       c.CreditLine             AS credit_limit,
       c.frozenFor = 'Y'        AS on_stop,
       c.U_CIS_Contractor = 'Y' AS cis_contractor,
       c.CreateDate             AS customer_since
FROM OCRD c
JOIN OCRG g ON g.GroupCode = c.GroupCode
JOIN OCTG t ON t.GroupNum = c.GroupNum
LEFT JOIN OSLP s ON s.SlpCode = c.SlpCode
WHERE c.CardType = 'C';

-- ---------------------------------------------------------------- facts
-- One row per A/R invoice. Amounts include VAT unless named net.
CREATE VIEW ar_invoice AS
SELECT DocEntry               AS invoice_entry,
       DocNum                 AS invoice_number,
       CardCode               AS customer_code,
       DocDate                AS invoice_date,
       DocDueDate             AS due_date,
       NumAtCard              AS customer_reference,
       DocTotal - VatSum      AS net_amount,
       VatSum                 AS vat_amount,
       DocTotal               AS gross_amount
FROM OINV
WHERE CANCELED = 'N';

-- One row per A/R credit note. invoice_entry is the invoice it is based on; it is
-- NULL for a standalone credit, which settles invoices only through reconciliation.
CREATE VIEW ar_credit_note AS
SELECT h.DocEntry             AS credit_note_entry,
       h.DocNum               AS credit_note_number,
       h.CardCode             AS customer_code,
       h.DocDate              AS credit_date,
       b.invoice_entry,
       b.invoice_entry IS NULL AS standalone,
       h.Comments             AS comments,
       h.DocTotal - h.VatSum  AS net_amount,
       h.VatSum               AS vat_amount,
       h.DocTotal             AS gross_amount
FROM ORIN h
LEFT JOIN (SELECT DocEntry, MIN(BaseEntry) AS invoice_entry FROM RIN1 WHERE BaseType = 13 GROUP BY DocEntry) b
  ON b.DocEntry = h.DocEntry
WHERE h.CANCELED = 'N';

-- One row per incoming payment.
CREATE VIEW ar_receipt AS
SELECT DocEntry               AS receipt_entry,
       DocNum                 AS receipt_number,
       CardCode               AS customer_code,
       DocDate                AS receipt_date,
       DocTotal               AS amount,
       NoDocSum               AS on_account_amount,
       Comments               AS comments
FROM ORCT
WHERE Canceled = 'N' AND DocType = 'C';

-- One row per allocation of a payment to an invoice when it was received (RCT2).
CREATE VIEW ar_allocation AS
SELECT r.DocEntry             AS receipt_entry,
       a.DocEntry             AS invoice_entry,
       r.CardCode             AS customer_code,
       r.DocDate              AS allocation_date,
       a.SumApplied           AS amount
FROM RCT2 a
JOIN ORCT r ON r.DocEntry = a.DocNum
WHERE a.InvType = '13' AND r.Canceled = 'N';

-- One row per document in an internal reconciliation, dated on the reconciliation.
CREATE VIEW ar_reconciliation AS
SELECT h.ReconNum             AS recon_number,
       h.ReconDate            AS recon_date,
       l.ShortName            AS customer_code,
       CASE l.SrcObjTyp WHEN '13' THEN 'Invoice' WHEN '14' THEN 'Credit note' WHEN '24' THEN 'Receipt' END AS document_type,
       l.SrcObjAbs            AS document_entry,
       l.ReconSum             AS amount,
       l.IsCredit = 'C'       AS credit_side
FROM ITR1 l
JOIN OITR h ON h.ReconNum = l.ReconNum
WHERE h.Canceled = 'N';

-- Sales at line grain, net of VAT: invoice lines positive, credit note lines negative.
CREATE VIEW ar_sales_line AS
SELECT 'Invoice' AS document_type, h.DocEntry AS document_entry, h.CardCode AS customer_code,
       h.DocDate AS document_date, l.ItemCode AS item_code, i.ItemName AS item_name, g.ItmsGrpNam AS item_group,
       l.Quantity AS quantity, l.LineTotal AS net_amount
FROM INV1 l JOIN OINV h ON h.DocEntry = l.DocEntry
JOIN OITM i ON i.ItemCode = l.ItemCode JOIN OITB g ON g.ItmsGrpCod = i.ItmsGrpCod
WHERE h.CANCELED = 'N'
UNION ALL
SELECT 'Credit note', h.DocEntry, h.CardCode, h.DocDate, l.ItemCode, i.ItemName, g.ItmsGrpNam,
       -l.Quantity, -l.LineTotal
FROM RIN1 l JOIN ORIN h ON h.DocEntry = l.DocEntry
JOIN OITM i ON i.ItemCode = l.ItemCode JOIN OITB g ON g.ItmsGrpCod = i.ItmsGrpCod
WHERE h.CANCELED = 'N';

-- Every movement on a customer's account, signed as it affects what they owe.
-- Balances come from here by summing up to a date: additive across customers,
-- no joins, so nothing can fan out. Reconciliations move nothing: they only say
-- which open items cancel which, so they never touch the balance.
CREATE VIEW ar_ledger AS
SELECT customer_code, invoice_date AS movement_date, 'Invoice' AS movement_type, invoice_entry AS document_entry,
       gross_amount AS amount FROM ar_invoice
UNION ALL
SELECT customer_code, credit_date, 'Credit note', credit_note_entry, -gross_amount FROM ar_credit_note
UNION ALL
SELECT customer_code, receipt_date, 'Receipt', receipt_entry, -amount FROM ar_receipt;

-- ---------------------------------------------------------------- as at a date
-- Open items as at a date: invoices not yet settled, and standalone credits not yet
-- matched (shown negative, as on an aged debt report). Everything that settles an
-- invoice (a credit based on it, a payment allocated on receipt, a reconciliation)
-- is summed per invoice BEFORE the join (aggregate before join), so each invoice
-- meets one row of "settled so far" and an invoice paid in three parts appears once.
CREATE MACRO ar_open_items(as_of) AS TABLE
WITH invoice_settled AS (
    SELECT invoice_entry, SUM(amount) AS settled
    FROM (SELECT invoice_entry, gross_amount AS amount FROM ar_credit_note
          WHERE NOT standalone AND credit_date <= as_of
          UNION ALL
          SELECT invoice_entry, amount FROM ar_allocation WHERE allocation_date <= as_of
          UNION ALL
          SELECT document_entry, amount FROM ar_reconciliation WHERE document_type = 'Invoice' AND recon_date <= as_of)
    GROUP BY invoice_entry
),
credit_matched AS (
    SELECT document_entry AS credit_note_entry, SUM(amount) AS matched
    FROM ar_reconciliation WHERE document_type = 'Credit note' AND recon_date <= as_of
    GROUP BY 1
)
SELECT 'Invoice' AS document_type, i.invoice_entry AS document_entry, i.invoice_number AS document_number,
       i.customer_code, i.invoice_date AS document_date, i.due_date, i.customer_reference, NULL AS comments,
       i.gross_amount,
       i.gross_amount - COALESCE(s.settled, 0)                     AS open_amount,
       DATE_DIFF('day', i.due_date, CAST(as_of AS DATE))           AS days_overdue,
       CASE WHEN DATE_DIFF('day', i.due_date, CAST(as_of AS DATE)) <= 0  THEN 'Not yet due'
            WHEN DATE_DIFF('day', i.due_date, CAST(as_of AS DATE)) <= 30 THEN '1-30 days'
            WHEN DATE_DIFF('day', i.due_date, CAST(as_of AS DATE)) <= 60 THEN '31-60 days'
            WHEN DATE_DIFF('day', i.due_date, CAST(as_of AS DATE)) <= 90 THEN '61-90 days'
            ELSE 'Over 90 days' END                                AS ageing_bucket
FROM ar_invoice i
LEFT JOIN invoice_settled s ON s.invoice_entry = i.invoice_entry
WHERE i.invoice_date <= as_of
  AND i.gross_amount - COALESCE(s.settled, 0) <> 0
UNION ALL
SELECT 'Credit note', c.credit_note_entry, c.credit_note_number, c.customer_code, c.credit_date, NULL, NULL,
       c.comments, -c.gross_amount,
       -(c.gross_amount - COALESCE(m.matched, 0)),
       NULL, 'Unapplied credit'
FROM ar_credit_note c
LEFT JOIN credit_matched m USING (credit_note_entry)
WHERE c.standalone AND c.credit_date <= as_of
  AND c.gross_amount - COALESCE(m.matched, 0) <> 0;

-- Cash received but not allocated or reconciled to any invoice, per payment, as at a date.
CREATE MACRO ar_unallocated_receipts(as_of) AS TABLE
WITH used AS (
    SELECT receipt_entry, SUM(amount) AS used
    FROM (SELECT receipt_entry, amount FROM ar_allocation WHERE allocation_date <= as_of
          UNION ALL
          SELECT document_entry, amount FROM ar_reconciliation WHERE document_type = 'Receipt' AND recon_date <= as_of)
    GROUP BY 1
)
SELECT r.receipt_entry, r.receipt_number, r.customer_code, r.receipt_date, r.comments,
       r.amount - COALESCE(u.used, 0) AS unallocated
FROM ar_receipt r
LEFT JOIN used u USING (receipt_entry)
WHERE r.receipt_date <= as_of AND r.amount - COALESCE(u.used, 0) <> 0;

-- The same, per customer.
CREATE MACRO ar_unallocated(as_of) AS TABLE
SELECT customer_code, SUM(unallocated) AS unallocated FROM ar_unallocated_receipts(as_of) GROUP BY 1;

-- Each customer's balance as at a date, straight from the ledger.
CREATE MACRO ar_balance(as_of) AS TABLE
SELECT customer_code, SUM(amount) AS balance
FROM ar_ledger
WHERE movement_date <= as_of
GROUP BY customer_code;

-- Aged debt as at a date, per customer: the familiar aged debtors report.
CREATE MACRO ar_aged_debt(as_of) AS TABLE
SELECT o.customer_code,
       SUM(o.open_amount) FILTER (WHERE ageing_bucket = 'Not yet due')      AS not_yet_due,
       SUM(o.open_amount) FILTER (WHERE ageing_bucket = '1-30 days')        AS days_1_30,
       SUM(o.open_amount) FILTER (WHERE ageing_bucket = '31-60 days')       AS days_31_60,
       SUM(o.open_amount) FILTER (WHERE ageing_bucket = '61-90 days')       AS days_61_90,
       SUM(o.open_amount) FILTER (WHERE ageing_bucket = 'Over 90 days')     AS over_90,
       SUM(o.open_amount) FILTER (WHERE document_type = 'Invoice')          AS open_invoices,
       SUM(o.open_amount) FILTER (WHERE ageing_bucket = 'Unapplied credit') AS unapplied_credits,
       SUM(o.open_amount)                                                  AS net_open_items
FROM ar_open_items(as_of) o
GROUP BY o.customer_code;

-- The matching backlog as at a date: every standalone credit and payment on account
-- with something left to match, and whatever reference was typed with it. This is
-- what the matching engine works through.
CREATE MACRO ar_matching_backlog(as_of) AS TABLE
SELECT 'Credit note' AS source_type, document_entry AS source_entry, document_number AS source_number,
       customer_code, document_date AS source_date, -open_amount AS amount_to_match, comments
FROM ar_open_items(as_of) WHERE document_type = 'Credit note'
UNION ALL
SELECT 'Receipt', receipt_entry, receipt_number, customer_code, receipt_date, unallocated, comments
FROM ar_unallocated_receipts(as_of);
