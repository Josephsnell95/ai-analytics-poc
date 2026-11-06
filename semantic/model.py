"""Load and validate the YAML semantic model.

Validation happens once, at load time, so the compiler can trust the model and
only has to validate queries. A broken model fails loudly with every problem listed.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import ModelError

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")          # model names: lower_snake_case
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")  # a bare source column
TOKEN_RE = re.compile(r"\b[a-z_][a-z0-9_]*\b")      # measure references in derived sql

AGGREGATIONS = {
    "sum": "SUM({})",
    "avg": "AVG({})",
    "min": "MIN({})",
    "max": "MAX({})",
    "count": "COUNT({})",
    "count_distinct": "COUNT(DISTINCT {})",
}
TIME_GRAINS = ["day", "week", "month", "quarter", "year"]
DIMENSION_TYPES = {"string", "number", "time", "boolean"}
FORMATS = {None, "currency", "percent", "number"}

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "semantic_model.yaml"


@dataclass(frozen=True)
class Join:
    to: str
    from_key: str
    to_key: str
    relationship: str


@dataclass(frozen=True)
class Table:
    name: str
    source: str
    joins: tuple[Join, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class Dimension:
    name: str
    table: str
    sql: str
    type: str = "string"
    grains: tuple[str, ...] = ()
    description: str = ""
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Measure:
    name: str
    sql: str
    agg: str | None = None       # None for derived measures
    table: str | None = None     # None for derived measures
    derived: bool = False
    format: str | None = None
    description: str = ""
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class NotAvailable:
    """Something people will ask about that this model can't answer, and why.
    Questions mentioning any of its terms are declined before an LLM sees them."""
    topic: str
    terms: tuple[str, ...]
    reason: str
    pattern: re.Pattern = field(compare=False, repr=False, default=None)


@dataclass
class SemanticModel:
    fact: str
    tables: dict[str, Table]
    dimensions: dict[str, Dimension]
    measures: dict[str, Measure]
    ai_context: str = ""
    not_available: tuple[NotAvailable, ...] = ()
    # Filled by _resolve_join_paths: table -> joins walked from the fact to reach it.
    join_paths: dict[str, list[tuple[str, Join]]] = field(default_factory=dict)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "SemanticModel":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticModel":
        errors: list[str] = []
        try:
            tables = {
                name: Table(
                    name=name,
                    source=t["source"],
                    joins=tuple(Join(**j) for j in t.get("joins", []) or []),
                    description=t.get("description", ""),
                )
                for name, t in (d.get("tables") or {}).items()
            }
            dimensions = {
                name: Dimension(
                    name=name,
                    table=x["table"],
                    sql=str(x["sql"]).strip(),
                    type=x.get("type", "string"),
                    grains=tuple(x.get("grains", []) or []),
                    description=str(x.get("description", "")).strip(),
                    synonyms=tuple(x.get("synonyms", []) or []),
                )
                for name, x in (d.get("dimensions") or {}).items()
            }
            measures = {
                name: Measure(
                    name=name,
                    sql=str(x["sql"]).strip(),
                    agg=x.get("agg"),
                    table=x.get("table"),
                    derived=x.get("type") == "derived",
                    format=x.get("format"),
                    description=str(x.get("description", "")).strip(),
                    synonyms=tuple(x.get("synonyms", []) or []),
                )
                for name, x in (d.get("measures") or {}).items()
            }
            not_available = tuple(
                NotAvailable(
                    topic=x["topic"],
                    terms=tuple(str(t).lower().strip() for t in x["terms"]),
                    reason=str(x["reason"]).strip(),
                    pattern=re.compile(
                        r"\b(" + "|".join(re.escape(str(t).lower().strip()) for t in x["terms"]) + r")\b",
                        re.IGNORECASE,
                    ),
                )
                for x in (d.get("not_available") or [])
            )
        except (KeyError, TypeError) as e:
            raise ModelError([f"Malformed model entry: {e!r}"]) from e

        model = cls(
            fact=d.get("fact", ""),
            tables=tables,
            dimensions=dimensions,
            measures=measures,
            ai_context=(d.get("ai_context") or "").strip(),
            not_available=not_available,
        )
        errors += model._resolve_join_paths()
        errors += model._validate_fields()
        if errors:
            raise ModelError(errors)
        return model

    # ------------------------------------------------------------- validation
    def _resolve_join_paths(self) -> list[str]:
        """Breadth-first walk from the fact table. Every table must be reached exactly
        once via many_to_one joins, i.e. the join graph is a tree rooted at the fact."""
        errors = []
        for name in self.tables:
            if not NAME_RE.match(name):
                errors.append(f"Table name '{name}' must be lower_snake_case")
        if self.fact not in self.tables:
            return errors + [f"Fact table '{self.fact}' is not defined under tables"]

        paths: dict[str, list[tuple[str, Join]]] = {self.fact: []}
        queue = deque([self.fact])
        while queue:
            parent = queue.popleft()
            for j in self.tables[parent].joins:
                if j.relationship != "many_to_one":
                    errors.append(
                        f"Join {parent} -> {j.to} is '{j.relationship}'; only many_to_one "
                        "is supported, so the compiler can never fan out"
                    )
                    continue
                if j.to not in self.tables:
                    errors.append(f"Join {parent} -> {j.to}: table '{j.to}' is not defined")
                    continue
                if not (IDENT_RE.match(j.from_key) and IDENT_RE.match(j.to_key)):
                    errors.append(f"Join {parent} -> {j.to}: keys must be plain column names")
                    continue
                if j.to in paths:
                    errors.append(
                        f"Table '{j.to}' is reachable by more than one join path (via '{parent}'); "
                        "join paths must be unambiguous"
                    )
                    continue
                paths[j.to] = paths[parent] + [(parent, j)]
                queue.append(j.to)

        for name in self.tables:
            if name not in paths:
                errors.append(f"Table '{name}' cannot be reached from the fact table '{self.fact}'")
        self.join_paths = paths
        return errors

    def _validate_fields(self) -> list[str]:
        errors = []
        overlap = set(self.dimensions) & set(self.measures)
        for name in sorted(overlap):
            errors.append(f"'{name}' is defined as both a dimension and a measure")

        for d in self.dimensions.values():
            where = f"Dimension '{d.name}'"
            if not NAME_RE.match(d.name):
                errors.append(f"{where}: name must be lower_snake_case")
            if d.table not in self.tables:
                errors.append(f"{where}: table '{d.table}' is not defined")
            errors += _check_column_sql(where, d.sql)
            if d.type not in DIMENSION_TYPES:
                errors.append(f"{where}: type '{d.type}' must be one of {sorted(DIMENSION_TYPES)}")
            if d.type == "time":
                bad = [g for g in d.grains if g not in TIME_GRAINS]
                if not d.grains or bad:
                    errors.append(f"{where}: time dimensions need grains from {TIME_GRAINS}")
            elif d.grains:
                errors.append(f"{where}: only time dimensions can have grains")

        for m in self.measures.values():
            where = f"Measure '{m.name}'"
            if not NAME_RE.match(m.name):
                errors.append(f"{where}: name must be lower_snake_case")
            if m.format not in FORMATS:
                errors.append(f"{where}: format '{m.format}' must be one of currency, percent, number")
            if "?" in m.sql:
                errors.append(f"{where}: sql must not contain '?' (reserved for query parameters)")
            if m.derived:
                if m.agg or m.table:
                    errors.append(f"{where}: derived measures take neither agg nor table")
                refs = TOKEN_RE.findall(m.sql)
                unknown = [r for r in refs if r not in self.measures]
                if unknown:
                    errors.append(
                        f"{where}: references unknown measures {unknown}. Derived sql may only "
                        "refer to other measures; write SQL functions and keywords in UPPERCASE"
                    )
                if not refs:
                    errors.append(f"{where}: derived sql must reference at least one measure")
                continue
            if m.agg not in AGGREGATIONS:
                errors.append(f"{where}: agg '{m.agg}' must be one of {sorted(AGGREGATIONS)}")
            if m.table not in self.tables:
                errors.append(f"{where}: table '{m.table}' is not defined")
            elif m.table != self.fact and m.agg != "count_distinct":
                errors.append(
                    f"{where}: '{m.agg}' over dimension table '{m.table}' would be repeated once per "
                    "fact row and give wrong totals. Put the measure on the fact table, or use "
                    "count_distinct"
                )
            if m.sql != "*" or m.agg != "count":
                errors += _check_column_sql(where, m.sql)

        errors += self._check_derived_cycles()
        errors += self._check_synonyms()
        errors += self._check_not_available()
        return errors

    def _check_not_available(self) -> list[str]:
        # A blocked term that is also a field name or synonym would decline questions
        # the model can answer, e.g. blocking "sales" would block every revenue question.
        answerable = set(self.dimensions) | set(self.measures)
        answerable |= {s.lower() for f in self.all_fields() for s in f.synonyms}
        answerable |= {n.replace("_", " ") for n in answerable}
        errors = []
        for na in self.not_available:
            if not na.terms or any(not t for t in na.terms):
                errors.append(f"not_available '{na.topic}': terms must be a non-empty list of words")
            for t in na.terms:
                if t in answerable:
                    errors.append(
                        f"not_available '{na.topic}': term '{t}' is a field or synonym the model "
                        "can answer, so blocking it would decline valid questions"
                    )
        return errors

    def _check_derived_cycles(self) -> list[str]:
        errors = []

        def visit(name: str, stack: list[str]) -> None:
            if name in stack:
                errors.append(f"Derived measures form a cycle: {' -> '.join(stack + [name])}")
                return
            m = self.measures.get(name)
            if m and m.derived:
                for ref in set(TOKEN_RE.findall(m.sql)):
                    visit(ref, stack + [name])

        for name, m in self.measures.items():
            if m.derived:
                visit(name, [])
        return sorted(set(errors))

    def _check_synonyms(self) -> list[str]:
        errors = []
        seen: dict[str, str] = {}
        for f in self.all_fields():
            for s in f.synonyms:
                key = s.lower()
                if key in self.dimensions or key in self.measures:
                    if key != f.name:
                        errors.append(f"Synonym '{s}' on '{f.name}' is already a field name")
                elif key in seen and seen[key] != f.name:
                    errors.append(f"Synonym '{s}' is used by both '{seen[key]}' and '{f.name}'")
                seen[key] = f.name
        return errors

    # ---------------------------------------------------------------- lookups
    def all_fields(self) -> list[Dimension | Measure]:
        return [*self.dimensions.values(), *self.measures.values()]

    def unavailable_topic(self, question: str) -> NotAvailable | None:
        """The first not_available entry the question mentions, if any."""
        for na in self.not_available:
            if na.pattern.search(question):
                return na
        return None

    def synonym_target(self, word: str) -> str | None:
        word = word.lower().strip()
        for f in self.all_fields():
            if word in (s.lower() for s in f.synonyms):
                return f.name
        return None

    # ------------------------------------------------------- SQL expressions
    def dimension_sql(self, name: str) -> str:
        d = self.dimensions[name]
        return _qualify(d.sql, d.table)

    def measure_sql(self, name: str) -> str:
        """Aggregate expression for a measure. Derived measures are expanded
        recursively into their base measures' aggregate expressions."""
        m = self.measures[name]
        if not m.derived:
            inner = "*" if m.sql == "*" else _qualify(m.sql, m.table)
            return AGGREGATIONS[m.agg].format(inner)
        # Base measures are a single aggregate call, so only derived ones need brackets.
        def expand(t):
            ref = t.group(0)
            sql = self.measure_sql(ref)
            return f"({sql})" if self.measures[ref].derived else sql

        return TOKEN_RE.sub(expand, m.sql)

    def measure_tables(self, name: str) -> set[str]:
        """Tables a measure needs, following derived measures down to their bases."""
        m = self.measures[name]
        if not m.derived:
            return {m.table}
        return set().union(*(self.measure_tables(r) for r in TOKEN_RE.findall(m.sql)))


def _qualify(sql: str, table: str) -> str:
    if IDENT_RE.match(sql):
        return f"{table}.{sql}"
    return sql.replace("{TABLE}", table)


def _check_column_sql(where: str, sql: str) -> list[str]:
    if "?" in sql:
        return [f"{where}: sql must not contain '?' (reserved for query parameters)"]
    if IDENT_RE.match(sql) or "{TABLE}" in sql:
        return []
    return [f"{where}: sql must be a bare column name or an expression using {{TABLE}}"]
