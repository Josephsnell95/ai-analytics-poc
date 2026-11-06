"""The agent's three tools, and the state they share within one question.

Every tool returns a string for the model to read. Failures come back as text
the model can act on, never as exceptions, so the loop can let it retry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from semantic import QueryResult, QueryValidationError, SemanticEngine

from .charts import CHART_TYPES, ChartError, build_chart

# ------------------------------------------------------------ tool schemas
# Dimensions are plain strings, with time grains written "order_date__month".
# The compiler also accepts {"field", "grain"} objects, but offering Llama an object
# here led it to stuff filters into dimensions ({"field", "op", "value"}).

TOOLS = [
    {
        "name": "get_model",
        "description": (
            "Look up measures and dimensions in the semantic model. The full list is already in "
            "your instructions; use this to search by keyword when you're unsure which field fits."
        ),
        "parameters": {
            "type": "object",
            "properties": {"search": {"type": "string", "description": "Keyword, e.g. 'margin' or 'country'"}},
        },
    },
    {
        "name": "run_query",
        "description": (
            "Run a semantic query and get the result rows. Use only measure and dimension names "
            "from the model. If the query is invalid you get an error explaining how to fix it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "measures": {"type": "array", "items": {"type": "string"}},
                "dimensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only for breakdowns ('by', 'each', 'per', trends). Leave out for a "
                                   "single total. Time grains: order_date__year, order_date__quarter, "
                                   "order_date__month, order_date__week, order_date__day",
                },
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "op": {
                                "type": "string",
                                "enum": ["=", "!=", ">", ">=", "<", "<=", "between", "in",
                                         "not_in", "contains", "is_null", "is_not_null"],
                            },
                            "value": {"description": "Value; a list for in/not_in; [low, high] for between. "
                                                     "Dates: '2013-01-31', or a whole period '2013', '2013-Q2', '2013-03'"},
                        },
                        "required": ["field", "op"],
                    },
                },
                "sort": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "dir": {"type": "string", "enum": ["asc", "desc"]},
                        },
                        "required": ["field", "dir"],
                    },
                },
                "limit": {"type": "integer", "description": "Only for top-N questions. Leave out otherwise"},
                "assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "How you interpreted anything ambiguous in the question",
                },
            },
            "required": ["measures"],
        },
    },
    {
        "name": "make_chart",
        "description": "Chart the result of the last successful run_query, using its column names.",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {"type": "string", "enum": CHART_TYPES},
                "x": {"type": "string", "description": "Column for the x axis (or pie labels)"},
                "y": {"type": "array", "items": {"type": "string"}, "description": "Measure column(s)"},
                "series": {"type": "string", "description": "Optional dimension column to split into coloured series"},
                "title": {"type": "string"},
            },
            "required": ["chart_type", "x", "y"],
        },
    },
]
TOOL_NAMES = {t["name"] for t in TOOLS}
LIST_KEYS = ("measures", "dimensions", "filters", "sort", "assumptions")


@dataclass
class QueryAttempt:
    query: dict
    sql: str | None = None
    error: str | None = None
    row_count: int | None = None


@dataclass
class ToolState:
    """What the tools produced during one question; the loop returns it to the caller."""
    attempts: list[QueryAttempt] = field(default_factory=list)
    last_result: QueryResult | None = None
    last_query: dict | None = None
    chart: dict | None = None


class Tools:
    def __init__(self, engine: SemanticEngine, rows_to_llm: int):
        self.engine = engine
        self.rows_to_llm = rows_to_llm
        self.state = ToolState()

    def reset(self) -> None:
        self.state = ToolState()

    def call(self, name: str, args: dict[str, Any]) -> str:
        if name == "get_model":
            return describe_model(self.engine.model, args.get("search"))
        if name == "run_query":
            return self.run_query(args)
        if name == "make_chart":
            return self.make_chart(args)
        return f"Unknown tool '{name}'. Available tools: {sorted(TOOL_NAMES)}"

    # --------------------------------------------------------------- run_query
    def run_query(self, args: dict) -> str:
        query = normalise_query_args(args)
        attempt = QueryAttempt(query)
        self.state.attempts.append(attempt)
        try:
            result = self.engine.run(query)
        except QueryValidationError as e:
            attempt.error = str(e)
            return f"{e}\nFix the query and call run_query again."

        attempt.sql = result.compiled.display_sql()
        attempt.row_count = len(result.rows)
        self.state.last_result, self.state.last_query = result, query
        self.state.chart = None   # a chart always describes the latest result

        records = [_round_record(r, result) for r in result.records()]
        shown = records[: self.rows_to_llm]
        payload = {
            "columns": [c.name for c in result.columns],
            "row_count": len(records),
            "rows": shown,
        }
        if len(records) > len(shown):
            payload["note"] = f"Showing the first {len(shown)} of {len(records)} rows"
        if result.compiled.query.limit == len(records):
            payload["note_limit"] = f"Result hit the limit of {len(records)} rows; there may be more"
        if not records:
            payload["note"] = (
                "No rows matched. Orders only exist from 2010-12-29 to 2014-01-28, so check date "
                "filters fall in that range. For a text filter, run a query grouped by that dimension "
                "to see its exact values."
            )
        return json.dumps(payload, default=str)

    # -------------------------------------------------------------- make_chart
    def make_chart(self, args: dict) -> str:
        y = args.get("y")
        if isinstance(y, str):
            y = _maybe_json_list(y)
        try:
            spec = build_chart(self.state.last_result, args.get("chart_type", ""), args.get("x", ""),
                               y, args.get("series") or None, args.get("title", ""))
        except ChartError as e:
            return str(e)
        self.state.chart = spec
        return "Chart created. Now give your written answer."


# --------------------------------------------------------- model description
def describe_model(model, search: str | None = None) -> str:
    """Compact text description of the model. Used in the system prompt and by get_model."""
    term = (search or "").lower().strip()

    def matches(f) -> bool:
        haystack = " ".join([f.name, f.description, *f.synonyms]).lower()
        return not term or term in haystack

    lines = ["MEASURES (numbers to aggregate):"]
    for m in model.measures.values():
        if matches(m):
            syn = f" Also called: {', '.join(m.synonyms)}." if m.synonyms else ""
            lines.append(f"- {m.name}: {m.description}{syn}")
    lines.append("")
    lines.append("DIMENSIONS (things to group or filter by):")
    for d in model.dimensions.values():
        if matches(d):
            syn = f" Also called: {', '.join(d.synonyms)}." if d.synonyms else ""
            grains = f" Grains: {', '.join(d.grains)}." if d.grains else ""
            lines.append(f"- {d.name}: {d.description}{grains}{syn}")
    if term and len(lines) == 3:
        return f"No fields match '{search}'. Call get_model with no search to see everything."
    if not term:
        if model.not_available:
            lines += ["", "NOT AVAILABLE (decline these):"]
            lines += [f"- {na.topic}: {na.reason}" for na in model.not_available]
        lines += ["", "CONTEXT:", model.ai_context]
    return "\n".join(lines)


# --------------------------------------------------------------- helpers
def normalise_query_args(args: dict) -> dict:
    """Undo common small-model tool-calling slips before strict validation.

    Llama often wraps the query in a "query" key, or sends a list as a JSON string
    ('["revenue"]') or a bare string ('revenue'). These are formatting slips, not
    wrong answers, so fix them here. Anything else is left for the compiler to reject.
    """
    if set(args) == {"query"}:
        inner = args["query"]
        args = _maybe_json(inner) if isinstance(inner, str) else inner
        if not isinstance(args, dict):
            return {"query": inner}
    out = dict(args)
    for key in LIST_KEYS:
        if isinstance(out.get(key), str):
            out[key] = _maybe_json_list(out[key])
        if isinstance(out.get(key), dict):
            out[key] = [out[key]]
    limit = out.get("limit")
    if isinstance(limit, str):
        limit = int(limit) if limit.strip().isdigit() else None   # "null", "none", "all"
    if limit == 0 or isinstance(limit, bool):
        limit = None   # Llama sends 0 to mean "no limit"
    out["limit"] = limit
    if isinstance(out.get("dimensions"), list):
        out["dimensions"] = [_normalise_dimension(d) for d in out["dimensions"]]
    # Optional keys sent as null or empty mean "not given".
    return {k: v for k, v in out.items() if v not in (None, "", [])} or out


def _normalise_dimension(d: Any) -> Any:
    """{"field": "x", "grain": null} -> "x"; {"field": "x", "grain": "month"} -> "x__month".
    Objects with other keys (e.g. a filter put in the wrong place) are left for the
    compiler to reject, because guessing what was meant could change the answer.
    Keys sent as null mean "not given", so they are dropped first."""
    if isinstance(d, dict):
        d = {k: v for k, v in d.items() if v is not None}
    if not isinstance(d, dict) or not isinstance(d.get("field"), str) or set(d) - {"field", "grain"}:
        return d
    grain = d.get("grain")
    if grain in (None, "", "none", "null"):
        return d["field"]
    return f"{d['field']}__{grain}" if isinstance(grain, str) else d


def _maybe_json(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def _maybe_json_list(s: str) -> list:
    v = _maybe_json(s)
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        return [v]
    return [p.strip() for p in s.split(",") if p.strip()] if isinstance(v, str) else [v]


def _round_record(record: dict, result: QueryResult) -> dict:
    # Fewer digits means fewer tokens and less for a small model to misread.
    out = {}
    for c in result.columns:
        v = record[c.name]
        if isinstance(v, float):
            v = round(v, 4) if c.format == "percent" else round(v, 2)
        out[c.name] = v
    return out
