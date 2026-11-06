"""Run a semantic query by hand, no LLM involved.

    python -m semantic '{"measures": ["revenue"], "dimensions": ["product_category"]}'
    python -m semantic path/to/query.json
    python -m semantic --sql-only '{"measures": ["gross_margin_pct"]}'
"""

import argparse
import json
import sys
from pathlib import Path

from . import QueryValidationError, SemanticEngine


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m semantic")
    parser.add_argument("query", help="semantic query as JSON text or a path to a .json file")
    parser.add_argument("--sql-only", action="store_true", help="print the SQL without running it")
    args = parser.parse_args()

    raw = Path(args.query).read_text() if args.query.endswith(".json") else args.query
    engine = SemanticEngine.from_config()
    try:
        if args.sql_only:
            print(engine.compile(raw).display_sql())
            return 0
        result = engine.run(raw)
    except QueryValidationError as e:
        print(e, file=sys.stderr)
        return 1

    print(result.compiled.display_sql(), end="\n\n")
    names = [c.name for c in result.columns]
    widths = [max(len(n), *(len(_fmt(r[i])) for r in result.rows)) if result.rows else len(n)
              for i, n in enumerate(names)]
    print("  ".join(n.ljust(w) for n, w in zip(names, widths)))
    print("  ".join("-" * w for w in widths))
    for row in result.rows:
        print("  ".join(_fmt(v).ljust(w) for v, w in zip(row, widths)))
    print(f"\n{len(result.rows)} rows")
    return 0


def _fmt(v) -> str:
    if v is None:
        return ""
    if hasattr(v, "as_tuple") or isinstance(v, float):
        return f"{float(v):,.4f}".rstrip("0").rstrip(".")
    return str(v)


if __name__ == "__main__":
    sys.exit(main())
