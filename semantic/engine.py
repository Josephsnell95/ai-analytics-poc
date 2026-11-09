"""Run compiled queries against DuckDB. The only place the semantic layer touches data."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import duckdb
import yaml

from .compiler import Column, CompiledQuery, compile
from .errors import QueryValidationError
from .model import SemanticModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "adventureworks.duckdb"
# Text dimensions with at most this many distinct values have their filter values
# checked. Above it (customer names, cities) a typo just matches nothing.
MAX_CHECKED_VALUES = 300
VALUE_OPS = {"=", "!=", "in", "not_in"}


@dataclass
class QueryResult:
    columns: list[Column]
    rows: list[tuple]
    compiled: CompiledQuery

    def records(self) -> list[dict[str, Any]]:
        """Rows as dicts with JSON-friendly values (Decimal -> float, dates -> ISO)."""
        return [
            {c.name: _plain(v) for c, v in zip(self.columns, row)}
            for row in self.rows
        ]


class SemanticEngine:
    """Model + database + limits in one object: validate, compile, run."""

    def __init__(self, model: SemanticModel, db_path: str | Path = DEFAULT_DB_PATH,
                 default_limit: int = 100, max_limit: int = 1000):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"{db_path} not found. Build it with: python data/load.py")
        self.model = model
        self.con = duckdb.connect(str(db_path), read_only=True)
        self.default_limit = default_limit
        self.max_limit = max_limit
        self._values: dict[str, set[str] | None] = {}   # dimension -> values in sales, None if too many

    @classmethod
    def from_config(cls, config_path: str | Path = REPO_ROOT / "config.yaml") -> "SemanticEngine":
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        root = Path(config_path).resolve().parent
        return cls(
            SemanticModel.load(root / cfg["paths"]["model"]),
            root / cfg["paths"]["database"],
            default_limit=cfg["compiler"]["default_limit"],
            max_limit=cfg["compiler"]["max_limit"],
        )

    def compile(self, raw: dict | str) -> CompiledQuery:
        return compile(self.model, raw, default_limit=self.default_limit, max_limit=self.max_limit)

    def run(self, raw: dict | str) -> QueryResult:
        compiled = self.compile(raw)
        self._check_filter_values(compiled)
        rows = self.con.execute(compiled.sql, compiled.params).fetchall()
        return QueryResult(compiled.columns, rows, compiled)

    def dimension_values(self, name: str) -> set[str] | None:
        """Distinct values of a text dimension, or None when there are more than
        MAX_CHECKED_VALUES. Read from the dimension table itself rather than from sales,
        so a real value with no sales (the Components category) is valid and gives zero.
        Loaded on first use and cached."""
        if name not in self._values:
            dim = self.model.dimensions[name]
            if dim.type != "string":
                self._values[name] = None
            else:
                table = self.model.tables[dim.table]
                sql = (f"SELECT DISTINCT {self.model.dimension_sql(name)} FROM {table.source} AS {dim.table} "
                       f"LIMIT {MAX_CHECKED_VALUES + 1}")
                vals = {r[0] for r in self.con.execute(sql).fetchall() if r[0] is not None}
                self._values[name] = vals if len(vals) <= MAX_CHECKED_VALUES else None
        return self._values[name]

    def _check_filter_values(self, compiled: CompiledQuery) -> None:
        """Reject text filter values that never occur in the data, saying where the value
        does occur. Otherwise "customer_country = Europe" quietly returns no rows, and a
        small model invents an answer from nothing (seen in the first eval run)."""
        errors = []
        for f in compiled.query.filters:
            if f.is_measure or f.op not in VALUE_OPS:
                continue
            known = self.dimension_values(f.field)
            if known is None:
                continue
            for value in (f.value if isinstance(f.value, list) else [f.value]):
                if value not in known:
                    errors.append(self._unknown_value_message(f.field, value, known))
        if errors:
            raise QueryValidationError(errors)

    def _unknown_value_message(self, field: str, value: str, known: set[str]) -> str:
        msg = f"'{value}' is not a value of {field}."
        exact_case = [k for k in known if k.lower() == str(value).lower()]
        if exact_case:
            return f"{msg} Did you mean '{exact_case[0]}'? Values are case-sensitive"
        elsewhere = [d for d in self.model.dimensions
                     if d != field and self.dimension_values(d)
                     and any(v.lower() == str(value).lower() for v in self.dimension_values(d))]
        if elsewhere:
            return f"{msg} It is a value of {', '.join(elsewhere)}; filter on that instead"
        close = get_close_matches(str(value), sorted(known), n=3, cutoff=0.6)
        if close:
            return f"{msg} Did you mean {', '.join(repr(c) for c in close)}?"
        sample = sorted(known)[:12]
        more = f" (and {len(known) - 12} more)" if len(known) > 12 else ""
        return f"{msg} Valid values: {', '.join(repr(v) for v in sample)}{more}"

    def close(self) -> None:
        self.con.close()


def _plain(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v
