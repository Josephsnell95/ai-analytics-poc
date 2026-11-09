"""Score one agent answer against a question's expected result.

Expected answers are stored as reference semantic queries, checked by a person,
and turned into values by the compiler. The agent is scored on the values its
own query returned, not on how it wrote the query, because several different
queries can be correct. Columns are compared by name, which works because both
sides use the semantic model's field names (e.g. "revenue", "order_date__month").

Match modes:
  exact     The agent grouped by the same dimensions and got the same rows (any
            order). Extra measures are fine. Default for breakdowns.
  contains  Every expected row appears in the agent's result. Default for single
            totals, so grouping "orders in Germany" by country, with Germany
            among the rows, still counts.
  ordered   Top-N: the first N rows' measure values come in the right order and
            every row is genuinely in the full ranking. Ties at the cut-off are
            accepted, and so is returning more than N rows if they're sorted.
            Used automatically when the reference has a sort and a limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from agent.loop import AgentResult
from semantic import SemanticEngine

# Outcomes, from best to worst. "correct_flagged" is a right result whose written
# answer quoted a figure the checker couldn't find, so the user saw a warning.
CORRECT, CORRECT_FLAGGED, DECLINED_OK = "correct", "correct_flagged", "declined_correctly"
WRONG, WRONGLY_DECLINED, ANSWERED_OUT_OF_SCOPE = "wrong", "wrongly_declined", "answered_out_of_scope"
NO_RESULT, ERROR = "no_result", "error"
PASSING = {CORRECT, CORRECT_FLAGGED, DECLINED_OK}


@dataclass
class Score:
    outcome: str
    reason: str = ""
    review: bool = False   # a person should read the stated assumption
    details: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.outcome in PASSING


def score(item: dict, result: AgentResult, engine: SemanticEngine) -> Score:
    review = bool(item.get("review"))
    if result.status == "error":
        return Score(ERROR, result.error or "provider error", review)

    if item.get("expect") == "decline":
        if result.status == "declined":
            return Score(DECLINED_OK, f"declined by {result.declined_by}", review)
        return Score(ANSWERED_OUT_OF_SCOPE, "should have declined but answered", review)

    if result.status == "declined":
        return Score(WRONGLY_DECLINED, f"declined by {result.declined_by}", review)
    if not result.semantic_query or result.status == "no_query":
        return Score(NO_RESULT, f"no successful query (status {result.status})", review)

    references = [item["reference"], *item.get("alternatives", [])]
    reasons = []
    for ref in references:
        ok, reason = match(ref, item.get("match"), result, engine)
        if ok:
            outcome = CORRECT_FLAGGED if result.unverified else CORRECT
            note = f"figures not in the data: {', '.join(result.unverified)}" if result.unverified else ""
            return Score(outcome, note, review)
        reasons.append(reason)
    return Score(WRONG, reasons[0] if len(reasons) == 1 else " / ".join(dict.fromkeys(reasons)), review)


def match(reference: dict, mode: str | None, result: AgentResult, engine: SemanticEngine) -> tuple[bool, str]:
    expected = engine.run(reference)
    exp_cols = [c.name for c in expected.columns]
    exp_dims = {c.name for c in expected.columns if c.role == "dimension"}
    exp_rows = expected.records()

    missing = [c for c in exp_cols if c not in result.columns]
    missing_dims = [c for c in missing if c in exp_dims]
    if missing_dims:
        return False, f"different breakdown: not grouped by {', '.join(missing_dims)}"
    if missing:
        return False, f"missing measure(s) {', '.join(missing)}"

    got_dims = {c for c in result.columns if c in _dimension_names(result, engine)}
    ordered = bool(reference.get("sort") and reference.get("limit"))
    mode = mode or ("ordered" if ordered else "contains" if not exp_dims else "exact")
    got = [{c: row[c] for c in exp_cols} for row in result.rows]

    if mode == "contains":
        for row in exp_rows:
            if not any(_rows_equal(row, g) for g in got):
                return False, f"expected row not found: {_short(row)}"
        return True, ""

    if got_dims != exp_dims:
        return False, f"different breakdown: grouped by {sorted(got_dims) or 'nothing'}, expected {sorted(exp_dims) or 'nothing'}"

    if mode == "ordered":
        return _match_ordered(reference, expected, got, engine)

    if len(got) != len(exp_rows):
        return False, f"{len(got)} rows, expected {len(exp_rows)}"
    remaining = list(got)
    for row in exp_rows:
        hit = next((g for g in remaining if _rows_equal(row, g)), None)
        if hit is None:
            return False, f"values differ, e.g. expected {_short(row)}"
        remaining.remove(hit)
    return True, ""


def _match_ordered(reference: dict, expected, got: list[dict], engine) -> tuple[bool, str]:
    """Right number of rows, sort measure in the expected order, and every row real.
    Comparing the sort values rather than the labels means tied rows can swap."""
    exp_rows = expected.records()
    if len(got) < len(exp_rows):
        return False, f"{len(got)} rows, expected top {len(exp_rows)}"
    # More rows than asked for is fine if the first N are the right N, in order: the
    # answer then contains the top N. An unsorted result fails the ranking check below.
    got = got[:len(exp_rows)]
    key = reference["sort"][0]["field"]
    key_col = next((c.name for c in expected.columns if c.name == key or c.name.startswith(key + "__")), key)
    want = [r[key_col] for r in exp_rows]
    have = [g[key_col] for g in got]
    if not all(_values_equal(a, b) for a, b in zip(want, have)):
        return False, f"wrong ranking: {key_col} values {_short(have)}, expected {_short(want)}"
    full = engine.run({k: v for k, v in reference.items() if k != "limit"} | {"limit": 1000}).records()
    for g in got:
        if not any(_rows_equal(g, f) for f in full):
            return False, f"row not in the data: {_short(g)}"
    return True, ""


def _dimension_names(result: AgentResult, engine) -> set[str]:
    names = set()
    for c in result.columns:
        base = c.partition("__")[0]
        if base in engine.model.dimensions:
            names.add(c)
    return names


def _rows_equal(a: dict, b: dict) -> bool:
    return all(_values_equal(a[k], b.get(k)) for k in a)


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=0.005)
    return a == b


def _short(x: Any, limit: int = 120) -> str:
    s = str(x)
    return s if len(s) <= limit else s[: limit - 3] + "..."
