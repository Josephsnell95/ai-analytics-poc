-- SAP Business One-shaped tables for the fictitious finance dataset.
--
-- Table and field names follow SAP Business One so the data looks like a B1
-- extract; only the fields the finance questions need are included. Check them
-- against a real instance before relying on any one of them. Amounts are GBP.
-- Tables starting _gen_ are not B1: they record how the generator built the data,
-- for tests and the data profile report only. The semantic model never uses them.

CREATE TABLE OCRG (             -- business partner groups
    GroupCode   INTEGER PRIMARY KEY,
    GroupName   VARCHAR NOT NULL,
    GroupType   VARCHAR NOT NULL   -- C customer, S supplier
);

CREATE TABLE OCTG (             -- payment terms
    GroupNum    INTEGER PRIMARY KEY,
    PymntGroup  VARCHAR NOT NULL,
    ExtraMonth  INTEGER NOT NULL,  -- 1 for "end of month" terms: count from the end of the invoice month
    ExtraDays   INTEGER NOT NULL
);

CREATE TABLE OSLP (             -- sales employees: the account owner
    SlpCode     INTEGER PRIMARY KEY,
    SlpName     VARCHAR NOT NULL
);

CREATE TABLE OCRD (             -- business partners: customers and suppliers
    CardCode    VARCHAR PRIMARY KEY,
    CardName    VARCHAR NOT NULL,
    CardType    VARCHAR NOT NULL,  -- C customer, S supplier
    GroupCode   INTEGER NOT NULL,
    GroupNum    INTEGER NOT NULL,  -- payment terms (OCTG)
    CreditLine  DECIMAL(19, 2) NOT NULL,
    SlpCode     INTEGER,           -- account owner (OSLP); customers only
    Country     VARCHAR NOT NULL,
    Currency    VARCHAR NOT NULL,
    CreateDate  DATE NOT NULL,
    validFor    VARCHAR NOT NULL,  -- Y active
    frozenFor   VARCHAR NOT NULL,  -- Y on stop
    WTLiable    VARCHAR NOT NULL,  -- Y subject to withholding tax: CIS subcontractors (suppliers only)
    WTCode      VARCHAR,           -- default withholding tax code (OWHT): CIS0, CIS20 or CIS30
    U_CIS_Contractor VARCHAR NOT NULL  -- user-defined field: Y customer is a CIS-registered contractor
);

CREATE TABLE OWHT (             -- withholding tax codes, used here for CIS
    WTCode      VARCHAR PRIMARY KEY,
    WTName      VARCHAR NOT NULL,
    Rate        DECIMAL(19, 2) NOT NULL
);

CREATE TABLE OITB (             -- item groups
    ItmsGrpCod  INTEGER PRIMARY KEY,
    ItmsGrpNam  VARCHAR NOT NULL
);

CREATE TABLE OITM (             -- items
    ItemCode    VARCHAR PRIMARY KEY,
    ItemName    VARCHAR NOT NULL,
    ItmsGrpCod  INTEGER NOT NULL
);

CREATE TABLE OINV (             -- A/R invoices
    DocEntry    INTEGER PRIMARY KEY,
    DocNum      INTEGER NOT NULL,
    CANCELED    VARCHAR NOT NULL,  -- Y cancelled; none are generated
    DocStatus   VARCHAR NOT NULL,  -- O open, C closed, as of today
    CardCode    VARCHAR NOT NULL,
    CardName    VARCHAR NOT NULL,
    DocDate     DATE NOT NULL,
    DocDueDate  DATE NOT NULL,
    TaxDate     DATE NOT NULL,
    NumAtCard   VARCHAR,           -- customer's order reference
    DocCur      VARCHAR NOT NULL,
    VatSum      DECIMAL(19, 2) NOT NULL,
    DocTotal    DECIMAL(19, 2) NOT NULL,  -- including VAT
    PaidToDate  DECIMAL(19, 2) NOT NULL,  -- as of today: payments allocated plus credit notes against it
    SlpCode     INTEGER,
    GroupNum    INTEGER NOT NULL,
    Comments    VARCHAR
);

CREATE TABLE INV1 (             -- A/R invoice lines
    DocEntry    INTEGER NOT NULL,
    LineNum     INTEGER NOT NULL,
    ItemCode    VARCHAR NOT NULL,
    Dscription  VARCHAR NOT NULL,
    Quantity    DECIMAL(19, 6) NOT NULL,
    Price       DECIMAL(19, 2) NOT NULL,
    LineTotal   DECIMAL(19, 2) NOT NULL,  -- excluding VAT
    VatPrcnt    DECIMAL(19, 2) NOT NULL,
    PRIMARY KEY (DocEntry, LineNum)
);

CREATE TABLE ORIN (             -- A/R credit notes; same shape as OINV
    DocEntry    INTEGER PRIMARY KEY,
    DocNum      INTEGER NOT NULL,
    CANCELED    VARCHAR NOT NULL,  -- N normal; C a cancellation document for a cancelled invoice
    DocStatus   VARCHAR NOT NULL,  -- O while a standalone credit is not fully reconciled
    CardCode    VARCHAR NOT NULL,
    CardName    VARCHAR NOT NULL,
    DocDate     DATE NOT NULL,
    DocDueDate  DATE NOT NULL,
    TaxDate     DATE NOT NULL,
    NumAtCard   VARCHAR,
    DocCur      VARCHAR NOT NULL,
    VatSum      DECIMAL(19, 2) NOT NULL,
    DocTotal    DECIMAL(19, 2) NOT NULL,
    PaidToDate  DECIMAL(19, 2) NOT NULL,  -- as of today: applied to invoices
    SlpCode     INTEGER,
    GroupNum    INTEGER NOT NULL,
    Comments    VARCHAR            -- where people type the invoice reference(s) on a standalone credit
);

CREATE TABLE RIN1 (             -- A/R credit note lines; based on an invoice line, or standalone
    DocEntry    INTEGER NOT NULL,
    LineNum     INTEGER NOT NULL,
    ItemCode    VARCHAR NOT NULL,
    Dscription  VARCHAR NOT NULL,
    Quantity    DECIMAL(19, 6) NOT NULL,
    Price       DECIMAL(19, 2) NOT NULL,
    LineTotal   DECIMAL(19, 2) NOT NULL,
    VatPrcnt    DECIMAL(19, 2) NOT NULL,
    BaseType    INTEGER NOT NULL,  -- 13 based on an A/R invoice; -1 standalone (no base document)
    BaseEntry   INTEGER,           -- OINV.DocEntry when based
    BaseLine    INTEGER,           -- INV1.LineNum when based
    PRIMARY KEY (DocEntry, LineNum)
);

CREATE TABLE ORCT (             -- incoming payments
    DocEntry    INTEGER PRIMARY KEY,
    DocNum      INTEGER NOT NULL,
    DocType     VARCHAR NOT NULL,  -- C: from a customer
    Canceled    VARCHAR NOT NULL,
    CardCode    VARCHAR NOT NULL,
    CardName    VARCHAR NOT NULL,
    DocDate     DATE NOT NULL,
    TaxDate     DATE NOT NULL,
    DocTotal    DECIMAL(19, 2) NOT NULL,
    TrsfrSum    DECIMAL(19, 2) NOT NULL,  -- bank transfer: every generated payment
    NoDocSum    DECIMAL(19, 2) NOT NULL,  -- received on account, not allocated to any document
    JrnlMemo    VARCHAR,
    Comments    VARCHAR            -- remittance reference that came with a payment on account
);

CREATE TABLE RCT2 (             -- how each incoming payment is allocated to documents
    DocNum      INTEGER NOT NULL,  -- ORCT.DocEntry
    InvoiceId   INTEGER NOT NULL,  -- line number within the payment
    DocEntry    INTEGER NOT NULL,  -- the document paid
    InvType     VARCHAR NOT NULL,  -- 13: an A/R invoice
    SumApplied  DECIMAL(19, 2) NOT NULL,
    PRIMARY KEY (DocNum, InvoiceId)
);

CREATE TABLE OITR (             -- internal reconciliations: matching open credits and payments to invoices
    ReconNum    INTEGER PRIMARY KEY,
    ReconDate   DATE NOT NULL,     -- the date the documents settle each other
    IsCard      VARCHAR NOT NULL,  -- C a business partner reconciliation
    ReconType   VARCHAR NOT NULL,  -- M manual (check the codes your B1 version uses)
    Total       DECIMAL(19, 2) NOT NULL,
    Canceled    VARCHAR NOT NULL,
    IsSystem    VARCHAR NOT NULL,  -- N made by a person
    CreateDate  DATE NOT NULL
);

CREATE TABLE ITR1 (             -- reconciliation lines: each document and the amount it contributes
    ReconNum    INTEGER NOT NULL,
    LineSeq     INTEGER NOT NULL,
    ShortName   VARCHAR NOT NULL,  -- the business partner
    SrcObjTyp   VARCHAR NOT NULL,  -- 13 A/R invoice, 14 A/R credit note, 24 incoming payment
    SrcObjAbs   INTEGER NOT NULL,  -- that document's DocEntry
    ReconSum    DECIMAL(19, 2) NOT NULL,
    IsCredit    VARCHAR NOT NULL,  -- C credit side (credit note, payment), D debit side (invoice)
    PRIMARY KEY (ReconNum, LineSeq)
);

CREATE TABLE _gen_match (       -- NOT B1: the true match behind every standalone credit and payment on account
    source_type       VARCHAR NOT NULL,  -- credit note or receipt
    source_entry      INTEGER NOT NULL,  -- ORIN or ORCT DocEntry
    invoice_entry     INTEGER NOT NULL,
    amount            DECIMAL(19, 2) NOT NULL,
    reference_quality VARCHAR NOT NULL,  -- clean, typo, po_only, none: what the Comments field gives away
    recon_date        DATE               -- when it was reconciled; NULL never, or later than today = not yet
);

CREATE TABLE _gen_customer (    -- NOT B1: the hidden behaviour behind each customer
    CardCode     VARCHAR PRIMARY KEY,
    payer_type   VARCHAR NOT NULL,
    annual_spend DECIMAL(19, 2) NOT NULL,
    key_account  BOOLEAN NOT NULL,
    stop_date    DATE              -- bad-debt customers: no payments from this date
);

CREATE TABLE _gen_info (        -- NOT B1: how this dataset was made
    key   VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);
