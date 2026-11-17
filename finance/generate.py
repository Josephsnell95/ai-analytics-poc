"""Generate the fictitious finance dataset: SAP Business One-shaped A/R tables in DuckDB.

    python -m finance.generate                  # writes data/finance.duckdb
    python -m finance.generate --out other.duckdb --seed 7

Everything is driven by finance/generator.yaml and a seed, so the same settings
always give the same data. Amounts are whole pence (int) until they are written,
so every total ties up to the penny.

How the data is built, per customer:
  1. A profile: group, payment terms, annual spend, account owner, CIS status, and
     a hidden payer type (prompt, slow, chronic, batch, partial, disputes, bad debt).
  2. Invoices on working days, with seasonality and growth, lines priced from the
     item master, VAT at the standard rate for UK customers. A few are cancelled,
     each with its B1 cancellation document.
  3. Credit notes. Most are based on the invoice they correct, which settles it at
     once. Some are raised standalone, with the invoice reference typed (or mistyped,
     or left out) in Comments, sometimes covering several invoices or only part of
     one. Those stay open until someone matches them in an internal reconciliation
     (OITR/ITR1), which may be weeks later or not yet.
  4. Payments by payer type. Customers short-pay invoices they expect a credit
     against. Most payments are allocated to invoices on receipt (RCT2); some arrive
     on account with a remittance note and are reconciled later, or not yet.
Nothing is dated after today. The generator also records the true match behind
every standalone credit and on-account payment (_gen_match), so a matching engine
can be scored for precision and recall.

The reconciliation rules in the design doc are tests (finance/tests), not hopes.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import math
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import duckdb
import yaml

from . import names

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = Path(__file__).with_name("generator.yaml")
SCHEMA = Path(__file__).with_name("schema.sql")
VIEWS = Path(__file__).with_name("views.sql")
DEFAULT_OUT = ROOT / "data" / "finance.duckdb"

INVOICE, CREDIT_NOTE, RECEIPT = "13", "14", "24"     # B1 object types


# ------------------------------------------------------------------ dates
def is_workday(d: date) -> bool:
    return d.weekday() < 5


def next_workday(d: date) -> date:
    while not is_workday(d):
        d += timedelta(days=1)
    return d


def add_workdays(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        if is_workday(d):
            n -= 1
    return d


def month_end(d: date) -> date:
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


def last_workday_of_month(d: date) -> date:
    e = month_end(d)
    while not is_workday(e):
        e -= timedelta(days=1)
    return e


def workdays(start: date, end: date):
    d = start
    while d <= end:
        if is_workday(d):
            yield d
        d += timedelta(days=1)


# ------------------------------------------------------------------ records
@dataclass
class Customer:
    code: str
    name: str
    group: str
    terms: str
    slp: int
    country: str
    created: date
    spend_pence: int
    key_account: bool
    payer: str
    cis_contractor: bool
    stop: date | None = None


@dataclass
class Line:
    num: int
    item: str
    desc: str
    qty: int
    price: int          # pence
    vat_pct: int        # 20 or 0

    @property
    def net(self) -> int:
        return self.qty * self.price


def vat_of(lines) -> int:
    """VAT on a document: per line, rounded half up to the penny, then summed. Because
    rounding is per line, a credit note's VAT is exactly the sum of its parts."""
    return sum((ln.qty * ln.price * ln.vat_pct + 50) // 100 for ln in lines)


def gross_of(lines) -> int:
    return sum(ln.net for ln in lines) + vat_of(lines)


@dataclass
class Invoice:
    cust: Customer
    doc_date: date
    due: date
    lines: list[Line]
    ref: str
    entry: int = 0
    num: int = 0
    cancelled: bool = False
    disputed_open: bool = False      # in dispute today, with no credit note yet: held, unpaid
    based_credits: list["CreditNote"] = field(default_factory=list)      # settle the invoice on their own date
    expected_credits: list[tuple["CreditNote", int]] = field(default_factory=list)  # standalone: settle when reconciled
    settlements: list[tuple[date, int]] = field(default_factory=list)    # payments and reconciliations, by date

    @property
    def gross(self) -> int:
        return gross_of(self.lines)

    @property
    def net(self) -> int:
        return sum(ln.net for ln in self.lines)

    @property
    def vat(self) -> int:
        return vat_of(self.lines)

    def to_pay(self) -> int:
        """What the customer will pay in cash: the total less every credit they know is coming."""
        return (self.gross - sum(c.gross for c in self.based_credits)
                - sum(a for _, a in self.expected_credits))

    def credits_settled_by(self) -> date | None:
        dates = [c.doc_date for c in self.based_credits] + [c.doc_date for c, _ in self.expected_credits]
        return max(dates, default=None)


@dataclass
class CreditNote:
    kind: str                         # based, standalone, cancellation
    cust: Customer
    doc_date: date
    lines: list[tuple[Invoice, Line, int]]   # (invoice, invoice line, quantity credited)
    targets: list[tuple[Invoice, int]] = field(default_factory=list)   # standalone: how it should be matched
    reference: str = "none"           # standalone: clean, typo, po_only, none
    recon_date: date | None = None    # standalone: when it was matched (None: never)
    entry: int = 0
    num: int = 0
    comment: str = ""

    @property
    def doc_lines(self) -> list[Line]:
        return [Line(0, ln.item, ln.desc, q, ln.price, ln.vat_pct) for _, ln, q in self.lines]

    @property
    def gross(self) -> int:
        return gross_of(self.doc_lines)

    @property
    def net(self) -> int:
        return sum(ln.net for ln in self.doc_lines)

    @property
    def vat(self) -> int:
        return vat_of(self.doc_lines)


@dataclass
class Payment:
    cust: Customer
    doc_date: date
    allocations: list[tuple[Invoice, int]]   # what it pays
    on_account: bool = False          # received without allocation (NoDocSum); matched later, or never
    unidentified: int = 0             # cash on account that will never be matched to anything
    reference: str = "none"
    recon_date: date | None = None
    entry: int = 0
    num: int = 0
    comment: str = ""

    @property
    def total(self) -> int:
        return sum(a for _, a in self.allocations) + self.unidentified


# ------------------------------------------------------------------ generator
class Generator:
    def __init__(self, settings: dict, seed: int | None = None):
        self.s = settings
        self.seed = settings["seed"] if seed is None else seed
        self.rng = random.Random(self.seed)
        self.today: date = settings["today"]
        self.start: date = settings["history_start"]
        self.customers: list[Customer] = []
        self.invoices: list[Invoice] = []
        self.credits: list[CreditNote] = []
        self.payments: list[Payment] = []

    # ---- helpers
    def pick(self, weights: dict):
        keys = list(weights)
        return self.rng.choices(keys, weights=[w["share"] if isinstance(w, dict) else w for w in weights.values()])[0]

    def poisson(self, lam: float) -> int:
        if lam <= 0:
            return 0
        if lam < 1:
            return 1 if self.rng.random() < lam else 0
        L, k, p = math.exp(-lam), 0, 1.0     # Knuth; lam is a few a day at most
        while True:
            p *= self.rng.random()
            if p <= L:
                return k
            k += 1

    def due_date(self, doc_date: date, terms: str) -> date:
        t = self.s["payment_terms"][terms]
        base = month_end(doc_date) if t["eom"] else doc_date
        return base + timedelta(days=t["days"])

    def later_reconciliation(self, after: date, spec: dict, quality: str) -> date | None:
        """When an open credit or payment on account gets matched: some weeks later, or never.
        The worse the reference, the longer it takes and the likelier it is never matched."""
        if self.rng.random() < spec["never_reconciled"][quality]:
            return None
        lo, hi = spec["reconcile_after_days"]
        return add_workdays(after, round(self.rng.randint(lo, hi) * spec["lag_multiplier"][quality]))

    # ---- master data
    def build_items(self):
        self.items = []    # (code, name, group, list price pence)
        for g_idx, (group, spec) in enumerate(self.s["item_groups"].items()):
            for i in range(spec["items"]):
                price = max(50, round(spec["price_gbp"] * 100 * math.exp(self.rng.gauss(0, 0.6))))
                code = f"{group[:3].upper()}-{i + 1:04d}"
                self.items.append((code, names.item_name(self.rng, group, i), g_idx + 1, price))

    def build_customers(self):
        groups = self.s["customer_groups"]
        spend = self.s["annual_spend"]
        cis = self.s["cis"]
        n = self.s["customers"]
        key_idx = set(self.rng.sample(range(n), self.s["key_accounts"]["count"]))
        used_names = set()
        span = (self.today - self.start).days
        for i in range(n):
            group = self.pick(groups)
            terms = self.pick(groups[group]["terms"])
            payer = self.pick(self.s["payer_types"])
            annual = spend["median_gbp"] * math.exp(self.rng.gauss(0, spend["sigma"]))
            annual = min(max(annual, spend["min_gbp"]), spend["max_gbp"])
            key = i in key_idx
            if key:
                annual *= self.s["key_accounts"]["multiplier"]
                payer = self.rng.choice(["prompt", "slow", "batch"])   # a key account is never a bad debt
            created = self.start
            if not key and self.rng.random() < self.s["late_joiners"]:
                created = next_workday(self.start + timedelta(days=self.rng.randint(30, span - 120)))
            stop = None
            if payer == "bad_debt":
                lo, hi = self.s["payer_types"]["bad_debt"]["stops_between"]
                stop = lo + timedelta(days=self.rng.randint(0, (hi - lo).days))
                created = min(created, stop - timedelta(days=200))
            name = names.company_name(self.rng, group, used_names)
            self.customers.append(Customer(
                code=f"C{i + 1:05d}", name=name, group=group, terms=terms,
                slp=self.rng.randint(1, len(self.s["sales_employees"])),
                country="IE" if self.rng.random() < self.s["non_uk_share"] else "GB",
                created=created, spend_pence=round(annual * 100), key_account=key, payer=payer,
                cis_contractor=group in cis["contractor_groups"] and self.rng.random() < cis["contractor_share"],
                stop=stop))

    # ---- invoices
    def season(self, d: date) -> float:
        s = self.s["seasonality"]
        if d.month == 8:
            return s["aug"]
        if d.month == 12 and d.day >= 18:
            return s["late_dec"]
        if d.month == 1 and d.day <= 9:
            return s["early_jan"]
        return 1.0

    def build_invoices(self):
        item_by_group = defaultdict(list)
        for it in self.items:
            item_by_group[it[2]].append(it)
        groups = list(item_by_group)
        growth = self.s["growth_per_year"]
        vat_pct = round(self.s["vat_rate"] * 100)
        for c in self.customers:
            avg_inv = self.s["customer_groups"][c.group]["invoice_gbp"] * 100
            per_day = c.spend_pence / avg_inv / 252
            last_day = self.today
            if c.stop:
                last_day = min(self.today, c.stop + timedelta(days=45))   # keeps ordering for a while after it stops paying
            favourite = self.rng.sample(groups, 3)     # each customer favours a few item groups
            discount = self.rng.uniform(0, 0.12)
            ref_no = self.rng.randint(1000, 9000)
            for d in workdays(c.created, last_day):
                years = (d - self.start).days / 365.25
                lam = per_day * self.season(d) * (1 + growth) ** years
                for _ in range(self.poisson(lam)):
                    target = avg_inv * math.exp(self.rng.gauss(0, 0.6))
                    n_lines = min(8, 1 + int(self.rng.expovariate(0.6)))
                    lines, used = [], set()
                    for _ in range(n_lines):
                        g = self.rng.choice(favourite) if self.rng.random() < 0.8 else self.rng.choice(groups)
                        code, iname, _, list_price = self.rng.choice(item_by_group[g])
                        if code in used:
                            continue
                        used.add(code)
                        price = max(1, round(list_price * (1 - discount)))
                        qty = max(1, round(target / n_lines / price))
                        lines.append(Line(len(lines), code, iname, qty, price, vat_pct if c.country == "GB" else 0))
                    ref_no += self.rng.randint(1, 4)
                    self.invoices.append(Invoice(c, d, self.due_date(d, c.terms), lines, f"PO{ref_no}"))

    def build_cancellations(self):
        spec = self.s["cancellations"]
        for inv in self.invoices:
            if self.rng.random() < spec["rate"]:
                d = add_workdays(inv.doc_date, self.rng.randint(*spec["days_after"]))
                if d <= self.today:
                    inv.cancelled = True
                    self.credits.append(CreditNote("cancellation", inv.cust, d, [(inv, ln, ln.qty) for ln in inv.lines]))

    # ---- credit notes
    def build_credit_notes(self):
        cn = self.s["credit_notes"]
        sa = self.s["standalone_credits"]
        disputed = self.s["payer_types"]["disputes"]["disputed_share"]
        by_cust = defaultdict(list)
        for inv in self.invoices:
            if not inv.cancelled:
                by_cust[inv.cust.code].append(inv)
        credited = set()        # invoices already carrying a credit: one credit event per invoice keeps it simple
        for inv in self.invoices:
            if inv.cancelled or id(inv) in credited:
                continue
            rate = disputed if inv.cust.payer == "disputes" else cn["base_rate"]
            if self.rng.random() >= rate:
                continue
            d = add_workdays(inv.doc_date, self.rng.randint(*cn["days_after"]))
            if d > self.today:
                inv.disputed_open = inv.cust.payer == "disputes"
                continue
            if self.rng.random() >= sa["share_of_credits"]:
                # based on the invoice: B1 settles it on the credit note's date
                lines = self.credit_lines(inv, full=self.rng.random() < cn["full_credit_share"])
                note = CreditNote("based", inv.cust, d, lines)
                inv.based_credits.append(note)
                credited.add(id(inv))
                self.credits.append(note)
                continue
            # standalone: typed reference, matched later in an internal reconciliation
            group = [inv]
            if self.rng.random() < sa["multi_invoice_share"]:
                nearby = [o for o in by_cust[inv.cust.code] if o is not inv and id(o) not in credited
                          and abs((o.doc_date - inv.doc_date).days) <= 60 and o.doc_date <= d]
                extra = self.rng.randint(sa["invoices_per_credit"][0], sa["invoices_per_credit"][1]) - 1
                group += self.rng.sample(nearby, min(extra, len(nearby)))
            lines, targets = [], []
            for target in group:
                part = self.credit_lines(target, full=False)
                lines += part
                targets.append((target, gross_of([Line(0, "", "", q, ln.price, ln.vat_pct) for _, ln, q in part])))
                credited.add(id(target))
            if self.rng.random() < sa["leftover_share"]:
                # some of the credit is never assigned: the £100 credit split 40/30/10 with £20 left on account
                t, amt = targets[-1]
                targets[-1] = (t, amt - round(amt * self.rng.uniform(0.2, 0.6)))
            quality = self.pick(sa["reference_quality"])
            note = CreditNote("standalone", inv.cust, d, lines, targets, reference=quality,
                              recon_date=self.later_reconciliation(d, sa, quality))
            for t, amt in targets:
                t.expected_credits.append((note, amt))
            self.credits.append(note)

    def credit_lines(self, inv: Invoice, full: bool) -> list[tuple[Invoice, Line, int]]:
        if full:
            return [(inv, ln, ln.qty) for ln in inv.lines]
        chosen = self.rng.sample(inv.lines, min(len(inv.lines), self.rng.randint(1, 2)))
        return [(inv, ln, self.rng.randint(1, ln.qty)) for ln in chosen]

    # ---- payments
    def plan_payment(self, inv: Invoice) -> list[tuple[date, int]]:
        """(date, pence) instalments for one invoice, before grouping and before today's cut-off."""
        c, p = inv.cust, self.s["payer_types"]
        remaining = inv.to_pay()
        if remaining <= 0 or inv.disputed_open:
            return []
        after_credit = inv.credits_settled_by()

        def not_before_credit(d: date) -> date:
            if after_credit and d <= after_credit:
                d = add_workdays(after_credit, self.rng.randint(3, 10))
            return max(d, inv.doc_date + timedelta(days=1))

        if c.payer == "partial":
            k = self.rng.randint(*p["partial"]["instalments"])
            first = inv.due + timedelta(days=self.rng.randint(*p["partial"]["days_late"]))
            parts, left = [], remaining
            for i in range(k):
                amt = left if i == k - 1 else round(remaining / k)
                left -= amt
                parts.append((not_before_credit(first + timedelta(days=i * p["partial"]["gap_days"])), amt))
            return parts
        if c.payer == "disputes":
            if after_credit:
                return [(add_workdays(after_credit, self.rng.randint(5, 20)), remaining)]
            lo, hi = p["disputes"]["days_late"]
        elif c.payer in ("prompt", "slow", "chronic"):
            lo, hi = p[c.payer]["days_late"]
        elif c.payer == "bad_debt":
            lo, hi = p["slow"]["days_late"]
        else:
            raise ValueError(c.payer)
        d = not_before_credit(inv.due + timedelta(days=self.rng.randint(lo, hi)))
        if self.rng.random() < self.s["month_end_bunching"]:
            d = last_workday_of_month(d)
        return [(d, remaining)]

    def build_payments(self):
        by_cust = defaultdict(list)
        for inv in self.invoices:
            if not inv.cancelled:
                by_cust[inv.cust.code].append(inv)
        ua = self.s["unallocated_receipts"]
        for c in self.customers:
            invs = by_cust[c.code]
            day_allocs = defaultdict(list)    # date -> [(invoice, pence)]
            if c.payer == "batch":
                self.batch_runs(c, invs, day_allocs)
            else:
                for inv in invs:
                    for d, amt in self.plan_payment(inv):
                        d = next_workday(d)
                        if d > self.today or (c.stop and d >= c.stop):
                            break     # unpaid as at today; later instalments wait on earlier ones
                        day_allocs[d].append((inv, amt))
            for d in sorted(day_allocs):
                pay = Payment(c, d, day_allocs[d])
                if self.rng.random() < ua["share_of_payments"]:
                    pay.on_account = True
                    pay.reference = self.pick(ua["reference_quality"])
                    pay.recon_date = self.later_reconciliation(d, ua, pay.reference)
                self.payments.append(pay)
        # cash that arrives with nothing to match it to
        payers = [c for c in self.customers if c.payer != "bad_debt"]
        for c in self.rng.sample(payers, self.s["unidentified_cash"]):
            span = (self.today - c.created).days - 5
            if span < 60:
                continue
            d = next_workday(c.created + timedelta(days=self.rng.randint(60, span)))
            if d <= self.today:
                self.payments.append(Payment(c, d, [], on_account=True,
                                             unidentified=self.rng.choice([250, 500, 750, 1000, 1500, 2000]) * 100))

    def batch_runs(self, c: Customer, invs: list[Invoice], day_allocs):
        """One remittance on the last working day of each month, paying everything due by then."""
        pending = sorted((inv for inv in invs if inv.to_pay() > 0 and not inv.disputed_open), key=lambda i: i.due)
        run = last_workday_of_month(c.created)
        while run <= self.today and pending:
            ready = [inv for inv in pending if inv.due <= run
                     and (inv.credits_settled_by() is None or inv.credits_settled_by() < run)]
            for inv in ready:
                day_allocs[run].append((inv, inv.to_pay()))
                pending.remove(inv)
            run = last_workday_of_month(month_end(run) + timedelta(days=1))

    # ---- settlement dates, numbering, comments
    def settle(self):
        """Record, on each invoice, when each amount actually settled it."""
        for pay in self.payments:
            when = pay.recon_date if pay.on_account else pay.doc_date
            if when and when <= self.today:
                for inv, amt in pay.allocations:
                    inv.settlements.append((when, amt))
        for cn in self.credits:
            if cn.kind == "standalone" and cn.recon_date and cn.recon_date <= self.today:
                for inv, amt in cn.targets:
                    inv.settlements.append((cn.recon_date, amt))

    def number(self):
        self.invoices.sort(key=lambda i: (i.doc_date, i.cust.code))
        for n, inv in enumerate(self.invoices, start=1):
            inv.entry, inv.num = n, 100000 + n
        self.credits.sort(key=lambda c: (c.doc_date, c.cust.code, c.lines[0][0].entry))
        for n, cn in enumerate(self.credits, start=1):
            cn.entry, cn.num = n, 500000 + n
        self.payments.sort(key=lambda p: (p.doc_date, p.cust.code))
        for n, p in enumerate(self.payments, start=1):
            p.entry, p.num = n, 700000 + n

    def write_comments(self):
        for cn in self.credits:
            if cn.kind == "cancellation":
                cn.comment = f"Cancellation of invoice {cn.lines[0][0].num}"
            elif cn.kind == "based":
                cn.comment = f"Based on A/R Invoice {cn.lines[0][0].num}."
            else:
                partial = any(amt < t.gross for t, amt in cn.targets)
                cn.comment = names.credit_comment(self.rng, cn.reference, [t.num for t, _ in cn.targets],
                                                  [t.ref for t, _ in cn.targets], partial)
        for p in self.payments:
            if p.on_account and p.allocations:
                p.comment = names.remittance_comment(self.rng, p.reference, [i.num for i, _ in p.allocations],
                                                     [i.ref for i, _ in p.allocations], p.cust.name)
            elif p.unidentified:
                p.comment = names.remittance_comment(self.rng, "none", [], [], p.cust.name)

    def run(self):
        self.build_items()
        self.build_customers()
        self.build_invoices()
        self.build_cancellations()
        self.build_credit_notes()
        self.build_payments()
        self.settle()
        self.number()
        self.write_comments()
        return self

    # ---- output rows
    def tables(self) -> dict[str, list[tuple]]:
        s = self.s
        t: dict[str, list[tuple]] = {}
        groups = list(s["customer_groups"])
        sup_groups = list(s["supplier_groups"])
        terms = list(s["payment_terms"])
        wt = s["cis"]["wt_codes"]
        t["OCRG"] = ([(100 + i, g, "C") for i, g in enumerate(groups)]
                     + [(200 + i, g, "S") for i, g in enumerate(sup_groups)])
        t["OCTG"] = [(i + 1, name, 1 if spec["eom"] else 0, spec["days"]) for i, (name, spec) in enumerate(s["payment_terms"].items())]
        t["OSLP"] = [(i + 1, n) for i, n in enumerate(s["sales_employees"])]
        t["OWHT"] = [(code, spec["name"], spec["rate"]) for code, spec in wt.items()]
        rows = []
        for c in self.customers:
            credit_line = max(200000, round(c.spend_pence / 12 * 1.2 * 2 / 100000) * 100000)
            frozen = "Y" if c.stop and c.stop <= self.today else "N"
            rows.append((c.code, c.name, "C", 100 + groups.index(c.group), terms.index(c.terms) + 1, money(credit_line),
                         c.slp, c.country, "GBP", c.created, "Y", frozen, "N", None, "Y" if c.cis_contractor else "N"))
        used = {c.name for c in self.customers}
        for i in range(s["suppliers"]):
            group = self.pick(s["supplier_groups"])
            liable = group in s["cis"]["subcontractor_groups"]
            code = self.pick(s["cis"]["subcontractor_status"]) if liable else None
            rows.append((f"S{i + 1:05d}", names.supplier_name(self.rng, used), "S", 200 + sup_groups.index(group),
                         self.rng.choice([2, 3, 4]), money(0), None, "GB", "GBP", self.start, "Y", "N",
                         "Y" if liable else "N", code, "N"))
        t["OCRD"] = rows
        t["OITB"] = [(i + 1, g) for i, g in enumerate(s["item_groups"])]
        t["OITM"] = [(code, name, grp) for code, name, grp, _ in self.items]

        t["OINV"], t["INV1"] = [], []
        for inv in self.invoices:
            c = inv.cust
            if inv.cancelled:
                canceled, status, paid = "Y", "C", inv.gross
            else:
                paid = (sum(cn.gross for cn in inv.based_credits)
                        + sum(a for d, a in inv.settlements if d <= self.today))
                canceled, status = "N", "C" if paid == inv.gross else "O"
            t["OINV"].append((inv.entry, inv.num, canceled, status, c.code, c.name, inv.doc_date, inv.due, inv.doc_date,
                              inv.ref, "GBP", money(inv.vat), money(inv.gross), money(paid), c.slp,
                              terms.index(c.terms) + 1, None))
            for ln in inv.lines:
                t["INV1"].append((inv.entry, ln.num, ln.item, ln.desc, ln.qty, money(ln.price), money(ln.net), ln.vat_pct))

        t["ORIN"], t["RIN1"], t["OITR"], t["ITR1"], t["_gen_match"] = [], [], [], [], []
        recons = []    # (date, customer, [(objtype, entry, amount, is_credit)])
        for cn in self.credits:
            c = cn.cust
            if cn.kind == "cancellation":
                canceled, status, paid = "C", "C", cn.gross
            elif cn.kind == "based":
                canceled, status, paid = "N", "C", cn.gross
            else:
                matched = sum(a for _, a in cn.targets) if cn.recon_date and cn.recon_date <= self.today else 0
                canceled, status, paid = "N", "C" if matched == cn.gross else "O", matched
            t["ORIN"].append((cn.entry, cn.num, canceled, status, c.code, c.name, cn.doc_date, cn.doc_date, cn.doc_date,
                              None, "GBP", money(cn.vat), money(cn.gross), money(paid), c.slp,
                              terms.index(c.terms) + 1, cn.comment))
            for i, (inv, ln, q) in enumerate(cn.lines):
                based = cn.kind != "standalone"
                t["RIN1"].append((cn.entry, i, ln.item, ln.desc, q, money(ln.price), money(ln.price * q), ln.vat_pct,
                                  13 if based else -1, inv.entry if based else None, ln.num if based else None))
            if cn.kind == "standalone":
                for inv, amt in cn.targets:
                    t["_gen_match"].append(("credit note", cn.entry, inv.entry, money(amt), cn.reference, cn.recon_date))
                if cn.recon_date and cn.recon_date <= self.today:
                    recons.append((cn.recon_date, c, [(CREDIT_NOTE, cn.entry, sum(a for _, a in cn.targets), "C")]
                                   + [(INVOICE, inv.entry, amt, "D") for inv, amt in cn.targets]))

        t["ORCT"], t["RCT2"] = [], []
        for p in self.payments:
            c = p.cust
            no_doc = p.total if p.on_account else 0
            t["ORCT"].append((p.entry, p.num, "C", "N", c.code, c.name, p.doc_date, p.doc_date, money(p.total),
                              money(p.total), money(no_doc), f"Incoming Payments - {c.code}", p.comment or None))
            if not p.on_account:
                for i, (inv, amt) in enumerate(p.allocations):
                    t["RCT2"].append((p.entry, i, inv.entry, INVOICE, money(amt)))
            elif p.allocations:
                for inv, amt in p.allocations:
                    t["_gen_match"].append(("receipt", p.entry, inv.entry, money(amt), p.reference, p.recon_date))
                if p.recon_date and p.recon_date <= self.today:
                    recons.append((p.recon_date, c, [(RECEIPT, p.entry, p.total, "C")]
                                   + [(INVOICE, inv.entry, amt, "D") for inv, amt in p.allocations]))

        recons.sort(key=lambda r: (r[0], r[1].code))
        for n, (d, c, lines) in enumerate(recons, start=1):
            t["OITR"].append((n, d, "C", "M", money(sum(a for _, _, a, ic in lines if ic == "C")), "N", "N", d))
            for seq, (obj, entry, amt, is_credit) in enumerate(lines):
                t["ITR1"].append((n, seq, c.code, obj, entry, money(amt), is_credit))

        t["_gen_customer"] = [(c.code, c.payer, money(c.spend_pence), c.key_account, c.stop) for c in self.customers]
        t["_gen_info"] = [("company", s["company"]), ("today", str(self.today)), ("history_start", str(self.start)),
                          ("seed", str(self.seed))]
        return t


def money(pence: int) -> str:
    """Whole pence as an exact decimal string for DuckDB."""
    sign = "-" if pence < 0 else ""
    p = abs(pence)
    return f"{sign}{p // 100}.{p % 100:02d}"


def load_settings(path: Path = SETTINGS) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def generate(out: Path = DEFAULT_OUT, seed: int | None = None, settings: dict | None = None) -> dict[str, int]:
    """Build the dataset and write it to a fresh DuckDB file. Returns row counts per table."""
    settings = settings or load_settings()
    tables = Generator(settings, seed).run().tables()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = duckdb.connect(str(out))
    try:
        con.execute(SCHEMA.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            for name, rows in tables.items():
                if not rows:
                    continue
                f = Path(tmp) / f"{name}.csv"
                with f.open("w", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerows(rows)
                con.execute(f"COPY {name} FROM '{f.as_posix()}' (HEADER false, NULLSTR '')")
        con.execute(VIEWS.read_text(encoding="utf-8"))
    finally:
        con.close()
    return {name: len(rows) for name, rows in tables.items()}


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m finance.generate")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, help="override the seed in generator.yaml")
    args = parser.parse_args()
    counts = generate(Path(args.out), args.seed)
    for name, n in counts.items():
        print(f"{name:14} {n:>8,}")
    print(f"Written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
