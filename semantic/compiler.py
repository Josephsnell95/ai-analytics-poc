"""Compile a validated SemanticQuery into DuckDB SQL.

No LLM is involved here. Given the same model and query this always produces the
same SQL, which is what makes the agent's answers checkable.

Rules (see the handoff doc):
- Only tables needed by the requested fields are joined, walking many-to-one
  join paths out from the fact table. LEFT JOINs, so a missing dimension row can
  never silently drop sales from a total.
- Filters on dimensions go in WHERE; filters on measures go in HAVING.
- Time grains compile to DATE_TRUNC on the time dimension.
- Derived measures expand into their base measures' expressions.
- Filter values are bound as parameters, never pasted into the SQL.
- A row limit is always applied, and ordering is always fully specified so a
  limited result is deterministic.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .model import Join, SemanticModel
from .query import Filter, SemanticQuery, parse_query

SQL_OPS = {"=": "=", "!=": "<>", ">": ">", ">=": ">=", "<": "<", "<=": "<="}


@dataclass
class Column:
    name: str
    role: str                 # "dimension" or "measure"
    type: str                 # dimension type, or "number" for measures
    format: str | None = None
    grain: str | None = None


@dataclass
class CompiledQuery:
    sql: str
    params: list[Any]
    columns: list[Column]
    query: SemanticQuery = field(repr=False)

    def display_sql(self) -> str:
        """SQL with parameters inlined as literals, for showing to people only.
        Always execute .sql with .params."""
        parts = self.sql.split("?")
        out = [parts[0]]
        for value, rest in zip(self.params, parts[1:]):
            out.append(_literal(value))
            out.append(rest)
        return "".join(out)


def compile_query(model: SemanticModel, query: SemanticQuery) -> CompiledQuery:
    params: list[Any] = []
    select, columns = [], []

    for d in query.dimensions:
        dim = model.dimensions[d.name]
        select.append(f'{_grained(model.dimension_sql(d.name), d.grain)} AS "{d.output_name}"')
        columns.append(Column(d.output_name, "dimension", dim.type, grain=d.grain))
    for m in query.measures:
        select.append(f'{model.measure_sql(m)} AS "{m}"')
        columns.append(Column(m, "measure", "number", format=model.measures[m].format))

    where = [_condition(model, f, params) for f in query.filters if not f.is_measure]
    having = [_condition(model, f, params) for f in query.filters if f.is_measure]

    lines = ["SELECT", "  " + ",\n  ".join(select)]
    lines.append(f"FROM {model.tables[model.fact].source} AS {model.fact}")
    lines += _joins(model, _tables_needed(model, query))
    if where:
        lines.append("WHERE " + "\n  AND ".join(where))
    if query.dimensions:
        # With no measures this lists the distinct values that occur in sales,
        # and it still lets a measure filter go in HAVING.
        lines.append("GROUP BY " + ", ".join(str(i + 1) for i in range(len(query.dimensions))))
    if having:
        lines.append("HAVING " + "\n  AND ".join(having))
    order = _order_by(query)
    if order:
        lines.append("ORDER BY " + ", ".join(order))
    lines.append(f"LIMIT {int(query.limit)}")

    return CompiledQuery("\n".join(lines), params, columns, query)


def compile(model: SemanticModel, raw: dict | str, *, default_limit: int = 100,
            max_limit: int = 1000) -> CompiledQuery:
    """Validate a raw semantic query and compile it. Raises QueryValidationError."""
    query = parse_query(raw, model, default_limit=default_limit, max_limit=max_limit)
    return compile_query(model, query)


# ------------------------------------------------------------------ pieces
def _grained(expr: str, grain: str | None) -> str:
    if grain is None:
        return expr
    if grain == "day":
        return f"CAST({expr} AS DATE)"
    return f"CAST(DATE_TRUNC('{grain}', {expr}) AS DATE)"


def _tables_needed(model: SemanticModel, query: SemanticQuery) -> set[str]:
    tables = {model.dimensions[d.name].table for d in query.dimensions}
    for m in query.measures:
        tables |= model.measure_tables(m)
    for f in query.filters:
        if f.is_measure:
            tables |= model.measure_tables(f.field)
        else:
            tables.add(model.dimensions[f.field].table)
    return tables


def _joins(model: SemanticModel, tables: set[str]) -> list[str]:
    # Union of the join paths to every needed table, in breadth-first order from
    # the fact table so each join's parent is always already in the FROM clause.
    steps: dict[str, tuple[str, Join]] = {}
    for t in tables:
        for parent, join in model.join_paths[t]:
            steps[join.to] = (parent, join)
    ordered = sorted(steps.items(), key=lambda kv: (len(model.join_paths[kv[0]]), kv[0]))
    return [
        f"LEFT JOIN {model.tables[to].source} AS {to} "
        f"ON {parent}.{join.from_key} = {to}.{join.to_key}"
        for to, (parent, join) in ordered
    ]


def _condition(model: SemanticModel, f: Filter, params: list[Any]) -> str:
    expr = model.measure_sql(f.field) if f.is_measure else model.dimension_sql(f.field)
    if f.op == "is_null":
        return f"{expr} IS NULL"
    if f.op == "is_not_null":
        return f"{expr} IS NOT NULL"
    if f.op == "between":
        params.extend(f.value)
        return f"{expr} BETWEEN ? AND ?"
    if f.op == "not_between":   # only produced by expanding "!= 2013"-style periods
        params.extend(f.value)
        return f"{expr} NOT BETWEEN ? AND ?"
    if f.op in ("in", "not_in"):
        params.extend(f.value)
        keyword = "IN" if f.op == "in" else "NOT IN"
        return f"{expr} {keyword} ({', '.join('?' * len(f.value))})"
    if f.op == "contains":
        params.append(f.value)
        return f"contains(lower({expr}), lower(?))"
    params.append(f.value)
    return f"{expr} {SQL_OPS[f.op]} ?"


def _order_by(query: SemanticQuery) -> list[str]:
    # Explicit sorts first, then every remaining dimension ascending as a
    # tie-breaker, so LIMIT always cuts at the same rows.
    order = [f'"{s.field}" {s.dir.upper()}' for s in query.sort]
    sorted_fields = {s.field for s in query.sort}
    order += [f'"{d.output_name}" ASC' for d in query.dimensions if d.output_name not in sorted_fields]
    return order


def _literal(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float, Decimal)):
        return repr(v) if not isinstance(v, Decimal) else str(v)
    if isinstance(v, dt.date):
        return f"DATE '{v.isoformat()}'"
    return "'" + str(v).replace("'", "''") + "'"
