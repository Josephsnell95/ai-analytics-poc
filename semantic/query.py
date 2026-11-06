"""Parse and validate a semantic query (the JSON the agent emits) against the model.

Every message here is written for the agent: say what was wrong and what would be
valid instead. All problems are collected and raised together.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any

from .errors import QueryValidationError
from .model import SemanticModel

QUERY_KEYS = {"measures", "dimensions", "filters", "sort", "limit", "assumptions"}

# Which operators make sense for which kind of field.
COMPARISON_OPS = {"=", "!=", ">", ">=", "<", "<="}
OPS_BY_KIND = {
    "string": {"=", "!=", "in", "not_in", "contains", "is_null", "is_not_null"},
    "number": COMPARISON_OPS | {"between", "in", "not_in", "is_null", "is_not_null"},
    "time": COMPARISON_OPS | {"between", "in", "is_null", "is_not_null"},
    "boolean": {"=", "!=", "is_null", "is_not_null"},
    "measure": COMPARISON_OPS | {"between"},
}
NO_VALUE_OPS = {"is_null", "is_not_null"}
LIST_OPS = {"in", "not_in"}
# Filter ops that pin a field to specific values. Two of these on the same field
# are ANDed, so they can only both match if they overlap: almost always a mistake.
PINNING_OPS = {"=", "between", "in"}

# Date periods accepted wherever a date is: "2013", "2013-03", "2013-Q2".
YEAR_RE = re.compile(r"^(\d{4})$")
MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
QUARTER_RE = re.compile(r"^(\d{4})-?[Qq]([1-4])$")
DATE_HINT = "dates must be like '2013-01-31', or a whole period like '2013', '2013-03' or '2013-Q2'"


@dataclass(frozen=True)
class DimensionRef:
    name: str
    grain: str | None = None

    @property
    def output_name(self) -> str:
        return f"{self.name}__{self.grain}" if self.grain else self.name


@dataclass(frozen=True)
class Filter:
    field: str
    op: str
    value: Any = None
    is_measure: bool = False


@dataclass(frozen=True)
class Period:
    """A whole year, quarter or month, as its first and last day."""
    start: dt.date
    last: dt.date


@dataclass(frozen=True)
class Sort:
    field: str      # output column name
    dir: str = "asc"


@dataclass
class SemanticQuery:
    measures: list[str] = field(default_factory=list)
    dimensions: list[DimensionRef] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    sort: list[Sort] = field(default_factory=list)
    limit: int = 100
    assumptions: list[str] = field(default_factory=list)


def parse_query(
    raw: dict | str,
    model: SemanticModel,
    *,
    default_limit: int = 100,
    max_limit: int = 1000,
) -> SemanticQuery:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise QueryValidationError([f"Query is not valid JSON: {e}"]) from e
    if not isinstance(raw, dict):
        raise QueryValidationError(["Query must be a JSON object"])

    errors: list[str] = []
    for key in raw:
        if key not in QUERY_KEYS:
            hint = _suggest(key, QUERY_KEYS)
            errors.append(f"Unknown query key '{key}'.{hint} Valid keys: {sorted(QUERY_KEYS)}")

    q = SemanticQuery(limit=default_limit)
    q.measures = _parse_measures(raw.get("measures", []), model, errors)
    q.dimensions = _parse_dimensions(raw.get("dimensions", []), model, errors)
    q.filters = _parse_filters(raw.get("filters", []), model, errors)
    q.sort = _parse_sort(raw.get("sort", []), q, errors)
    q.limit = _parse_limit(raw.get("limit", default_limit), max_limit, errors)
    q.assumptions = _parse_assumptions(raw.get("assumptions", []), errors)

    if not q.measures and not q.dimensions and not errors:
        errors.append("Query needs at least one measure or dimension")
    if errors:
        raise QueryValidationError(errors)
    return q


# ---------------------------------------------------------------- sections
def _parse_measures(raw: Any, model: SemanticModel, errors: list[str]) -> list[str]:
    if not _is_list(raw, "measures", errors):
        return []
    out = []
    for m in raw:
        if not isinstance(m, str):
            errors.append(f"measures must be a list of measure names; got {m!r}")
        elif m not in model.measures:
            errors.append(_unknown_field(m, "measure", model))
        elif m in out:
            errors.append(f"Measure '{m}' is listed twice")
        else:
            out.append(m)
    return out


def _parse_dimensions(raw: Any, model: SemanticModel, errors: list[str]) -> list[DimensionRef]:
    if not _is_list(raw, "dimensions", errors):
        return []
    out: list[DimensionRef] = []
    for d in raw:
        if isinstance(d, str):
            # "order_date__month" is shorthand for {"field": "order_date", "grain": "month"},
            # and matches the result column name, so the same string works in sort and charts.
            name, _, grain = d.partition("__")
            grain = grain or None
        elif isinstance(d, dict):
            extra = set(d) - {"field", "grain"}
            if extra:
                errors.append(f"Dimension {d!r} has unknown keys {sorted(extra)}; use 'field' and 'grain'")
                continue
            name, grain = d.get("field"), d.get("grain")
            if not isinstance(name, str):
                errors.append(f"Dimension {d!r} needs a 'field' naming the dimension")
                continue
        else:
            errors.append(f"Each dimension must be a name or {{'field', 'grain'}}; got {d!r}")
            continue

        if name not in model.dimensions:
            errors.append(_unknown_field(name, "dimension", model))
            continue
        dim = model.dimensions[name]
        if dim.type == "time":
            grain = grain or "day"
            if grain not in dim.grains:
                errors.append(f"Dimension '{name}' has no grain '{grain}'. Valid grains: {list(dim.grains)}")
                continue
        elif grain is not None:
            errors.append(f"Dimension '{name}' is not a time dimension, so it cannot take a grain")
            continue

        ref = DimensionRef(name, grain)
        if any(r.output_name == ref.output_name for r in out):
            errors.append(f"Dimension '{ref.output_name}' is listed twice")
        else:
            out.append(ref)
    return out


def _parse_filters(raw: Any, model: SemanticModel, errors: list[str]) -> list[Filter]:
    if not _is_list(raw, "filters", errors):
        return []
    out = []
    pinned: dict[str, int] = {}
    for f in raw:
        if not isinstance(f, dict):
            errors.append(f"Each filter must be an object with field, op and value; got {f!r}")
            continue
        extra = set(f) - {"field", "op", "value"}
        if extra:
            errors.append(f"Filter {f!r} has unknown keys {sorted(extra)}; use 'field', 'op' and 'value'")
            continue
        name, op = f.get("field"), f.get("op")

        if name in model.measures:
            kind, is_measure = "measure", True
        elif name in model.dimensions:
            kind, is_measure = model.dimensions[name].type, False
        else:
            errors.append(_unknown_field(str(name), "filter field", model))
            continue

        allowed = OPS_BY_KIND[kind]
        if op not in allowed:
            label = "measure" if is_measure else f"{kind} dimension"
            errors.append(f"Filter on '{name}': op {op!r} is not valid for a {label}. Valid ops: {sorted(allowed)}")
            continue

        value, problem = _check_value(kind, op, f.get("value"), "value" in f)
        if not problem and kind == "time":
            op_used, value, problem = _expand_periods(op, value)
        else:
            op_used = op
        if problem:
            errors.append(f"Filter on '{name}' ({op}): {problem}")
            continue
        out.append(Filter(name, op_used, value, is_measure))
        if not is_measure and op in PINNING_OPS:
            pinned[name] = pinned.get(name, 0) + 1

    for name, n in pinned.items():
        if n > 1:
            hint = ("one 'between' covering the whole span, e.g. ['2012', '2013'], and group by a "
                    "grain to compare periods" if model.dimensions[name].type == "time"
                    else "one 'in' filter listing every value")
            errors.append(
                f"'{name}' has {n} filters using =, between or in. Filters are combined with AND, "
                f"so these can never all match. Use {hint}"
            )
    return out


def _parse_sort(raw: Any, q: SemanticQuery, errors: list[str]) -> list[Sort]:
    if not _is_list(raw, "sort", errors):
        return []
    # A sort can name a measure, a dimension, or a time dimension's grained output.
    # Plain "order_date" resolves to whatever grain was selected for it.
    selectable = {m: m for m in q.measures}
    for d in q.dimensions:
        selectable[d.output_name] = d.output_name
        selectable.setdefault(d.name, d.output_name)

    out = []
    for s in raw:
        if not isinstance(s, dict) or "field" not in s:
            errors.append(f"Each sort must be an object with 'field' and 'dir'; got {s!r}")
            continue
        extra = set(s) - {"field", "dir"}
        if extra:
            errors.append(f"Sort {s!r} has unknown keys {sorted(extra)}; use 'field' and 'dir'")
            continue
        direction = str(s.get("dir", "asc")).lower()
        if direction not in ("asc", "desc"):
            errors.append(f"Sort on '{s['field']}': dir must be 'asc' or 'desc'")
            continue
        if s["field"] not in selectable:
            errors.append(
                f"Cannot sort by '{s['field']}' because it is not in the query's measures or "
                f"dimensions. Add it, or sort by one of {sorted(selectable)}"
            )
            continue
        out.append(Sort(selectable[s["field"]], direction))
    return out


def _parse_limit(raw: Any, max_limit: int, errors: list[str]) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        errors.append(f"limit must be a positive integer; got {raw!r}")
        return 0
    if raw > max_limit:
        errors.append(f"limit {raw} exceeds the maximum of {max_limit}. Aggregate further or lower the limit")
        return 0
    return raw


def _parse_assumptions(raw: Any, errors: list[str]) -> list[str]:
    if not _is_list(raw, "assumptions", errors):
        return []
    if not all(isinstance(a, str) for a in raw):
        errors.append("assumptions must be a list of strings")
        return []
    return list(raw)


# ----------------------------------------------------------------- helpers
def _check_value(kind: str, op: str, value: Any, present: bool) -> tuple[Any, str | None]:
    """Return (converted value, problem message or None)."""
    if op in NO_VALUE_OPS:
        return (None, "takes no value") if present and value is not None else (None, None)
    if not present:
        return None, "needs a value"
    if op == "between":
        if not isinstance(value, list) or len(value) != 2:
            return None, "value must be a list of two items [low, high]"
        items = [_convert_scalar(kind, v) for v in value]
        problems = [p for _, p in items if p]
        if problems:
            return None, problems[0]
        return [v for v, _ in items], None
    if op in LIST_OPS:
        if not isinstance(value, list) or not value:
            return None, "value must be a non-empty list"
        items = [_convert_scalar(kind, v) for v in value]
        problems = [p for _, p in items if p]
        if problems:
            return None, problems[0]
        return [v for v, _ in items], None
    return _convert_scalar(kind, value)


def _convert_scalar(kind: str, v: Any) -> tuple[Any, str | None]:
    if kind == "time":
        if isinstance(v, int) and not isinstance(v, bool):
            v = str(v)   # 2013 sent as a number
        if not isinstance(v, str):
            return None, f"{DATE_HINT}; got {v!r}"
        v = v.strip()
        period = _parse_period(v)
        if period:
            return period, None
        try:
            return dt.date.fromisoformat(v), None
        except ValueError:
            return None, f"{DATE_HINT}; got {v!r}"
    if kind in ("number", "measure"):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, f"value must be a number; got {v!r}"
        return v, None
    if kind == "boolean":
        if not isinstance(v, bool):
            return None, f"value must be true or false; got {v!r}"
        return v, None
    if not isinstance(v, str):
        return None, f"value must be a string; got {v!r}"
    return v, None


def _parse_period(v: str) -> Period | None:
    if m := YEAR_RE.match(v):
        y = int(m.group(1))
        return Period(dt.date(y, 1, 1), dt.date(y, 12, 31))
    if m := QUARTER_RE.match(v):
        y, q = int(m.group(1)), int(m.group(2))
        return Period(dt.date(y, 3 * q - 2, 1), _month_end(y, 3 * q))
    if m := MONTH_RE.match(v):
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return Period(dt.date(y, mo, 1), _month_end(y, mo))
    return None


def _month_end(y: int, m: int) -> dt.date:
    first_of_next = dt.date(y + (m == 12), m % 12 + 1, 1)
    return first_of_next - dt.timedelta(days=1)


def _expand_periods(op: str, value: Any) -> tuple[str, Any, str | None]:
    """Turn period values into plain date comparisons, so "= 2013" means the whole
    year rather than 1 January. Returns (op, value, problem)."""
    if op in NO_VALUE_OPS:
        return op, value, None
    if op == "between":
        low, high = value
        return op, [low.start if isinstance(low, Period) else low,
                    high.last if isinstance(high, Period) else high], None
    if op in LIST_OPS:
        if any(isinstance(v, Period) for v in value):
            return op, value, "lists can't hold periods like '2013'; use one 'between' filter instead"
        return op, value, None
    if not isinstance(value, Period):
        return op, value, None
    p = value
    return {
        "=": ("between", [p.start, p.last], None),
        "!=": ("not_between", [p.start, p.last], None),
        ">": (">", p.last, None),
        ">=": (">=", p.start, None),
        "<": ("<", p.start, None),
        "<=": ("<=", p.last, None),
    }[op]


def _unknown_field(name: str, role: str, model: SemanticModel) -> str:
    # Most useful hint first: a dimension used as a measure (or vice versa).
    if role == "measure" and name in model.dimensions:
        return f"'{name}' is a dimension, not a measure. Put it in 'dimensions'"
    if role == "dimension" and name in model.measures:
        return f"'{name}' is a measure, not a dimension. Put it in 'measures'"

    target = model.synonym_target(name)
    if target:
        return f"Unknown {role} '{name}'. It is a synonym of '{target}'; use '{target}'"

    if role == "measure":
        candidates = list(model.measures)
    elif role == "dimension":
        candidates = list(model.dimensions)
    else:
        candidates = [*model.dimensions, *model.measures]
    hint = _suggest(name, candidates)
    return f"Unknown {role} '{name}'.{hint} Call get_model to see all valid {role}s"


def _suggest(name: str, candidates) -> str:
    close = get_close_matches(name.lower(), list(candidates), n=3, cutoff=0.5)
    return f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""


def _is_list(raw: Any, key: str, errors: list[str]) -> bool:
    if not isinstance(raw, list):
        errors.append(f"'{key}' must be a list; got {type(raw).__name__}")
        return False
    return True
