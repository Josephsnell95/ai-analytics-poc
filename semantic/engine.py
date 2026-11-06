"""Run compiled queries against DuckDB. The only place the semantic layer touches data."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import yaml

from .compiler import Column, CompiledQuery, compile
from .model import SemanticModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "adventureworks.duckdb"


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
        rows = self.con.execute(compiled.sql, compiled.params).fetchall()
        return QueryResult(compiled.columns, rows, compiled)

    def close(self) -> None:
        self.con.close()


def _plain(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v
