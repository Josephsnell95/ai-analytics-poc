"""Check the figures in a written answer against the query result.

Small models misread their own results: Llama reported Bikes gross margin as
$11,506,197 when the query returned $11,505,796.50. A prompt can't reliably stop
that, so every number in the answer is checked here, and any that doesn't
match a value in the result rows is flagged to the user.

A figure matches when it equals some result value at the precision it was
written, rounded or truncated: "$15.36 million" matches 15,359,502.36, "41.4%"
matches 0.4137, and "$668,241" matches 668,241.53. A flagged figure isn't necessarily wrong (it
could be a correct sum the model worked out), but it isn't in the data, and the
prompt tells the model not to do its own arithmetic.
"""

from __future__ import annotations

import re
from decimal import Decimal

# A number as written in prose: optional $, digits with optional thousands
# separators and decimals, optional scale word or %.
NUMBER_RE = re.compile(
    r"(?P<dollar>\$)?(?<![\w.])(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<scale>million|billion|thousand|mn|bn|[kKmMbB])\b)?(?P<pct>\s?%)?"
)
SCALES = {"thousand": 3, "k": 3, "million": 6, "mn": 6, "m": 6, "billion": 9, "bn": 9, "b": 9}
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
QUARTER_RE = re.compile(r"\b(?:\d{4}[- ]?)?Q[1-4]\b", re.IGNORECASE)
LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
           r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
# "December 31", "31st December", "1 January": days of the month, not figures.
DAY_OF_MONTH_RE = re.compile(rf"\b(?:{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS})\b")


def unverified_figures(answer: str, rows: list[dict], formats: dict[str, str | None]) -> list[str]:
    """Figures in the answer that don't match any value in the result rows,
    as written in the answer (e.g. "$11,506,197")."""
    if not answer or not rows:
        return []
    text = _strip_non_figures(answer, rows)
    values = _result_values(rows, formats)
    flagged = []
    for m in NUMBER_RE.finditer(text):
        figure = m.group(0).strip()
        written, half_unit = _parse(m)
        if written is None or _is_incidental(m, written):
            continue
        # Rounded (within half a unit either way) or truncated (up to a whole unit
        # below the true value): "$668,241" for 668,241.53 is cutting off the cents.
        if not any(-half_unit <= abs(v) - abs(written) < 2 * half_unit for v in values):
            if figure not in flagged:
                flagged.append(figure)
    return flagged


def _strip_non_figures(answer: str, rows: list[dict]) -> str:
    """Remove text whose digits aren't figures: dimension values such as
    "Mountain-200 Black, 46", dates, quarters and numbered-list markers."""
    labels = {str(v) for row in rows for v in row.values() if isinstance(v, str)}
    for label in sorted(labels, key=len, reverse=True):   # longest first, so "Road-150" before "150"
        answer = answer.replace(label, " ")
    for pattern in (ISO_DATE_RE, DAY_OF_MONTH_RE, QUARTER_RE, LIST_MARKER_RE):
        answer = pattern.sub(" ", answer)
    return answer


def _result_values(rows: list[dict], formats: dict[str, str | None]) -> list[Decimal]:
    values = []
    for row in rows:
        for col, v in row.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            d = Decimal(str(v))
            values.append(d)
            if formats.get(col) == "percent":
                values.append(d * 100)   # 0.4137 is written as 41.4%
    return values


def _parse(m: re.Match) -> tuple[Decimal | None, Decimal]:
    """The written value, and half a unit in its last written digit."""
    num = m.group("num").replace(",", "")
    try:
        value = Decimal(num)
    except ArithmeticError:
        return None, Decimal(0)
    decimals = len(num.split(".")[1]) if "." in num else 0
    exponent = SCALES.get((m.group("scale") or "").lower(), 0)
    if m.group("scale") in ("m", "b") and not m.group("dollar"):
        exponent = 0   # a bare "5 m" is too ambiguous to scale
    tolerance = Decimal(5) * Decimal(10) ** (exponent - decimals - 1)
    return value * Decimal(10) ** exponent, tolerance


def _is_incidental(m: re.Match, value: Decimal) -> bool:
    """Numbers that are part of the phrasing rather than results: years, and small
    counts like "top 5" or "3 product models". Money and percentages always count."""
    if m.group("dollar") or m.group("pct") or m.group("scale"):
        return False
    if "." in m.group("num") or "," in m.group("num"):
        return False
    return value < 13 or 1990 <= value <= 2100
