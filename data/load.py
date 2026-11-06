"""Build data/adventureworks.duckdb from Microsoft's public AdventureWorksDW sample.

    python data/load.py            # download (if needed) and rebuild from scratch
    python data/load.py --verify   # rebuild, then check row counts against source files

Column names and types come from the CREATE TABLE statements in Microsoft's own
install script (instawdbdw.sql), so nothing here is hand-typed per table. Binary
image columns are dropped: the app never needs them and they are most of the file size.

Only the Internet Sales star is loaded. Reseller Sales is out of scope for v1
(see the handoff doc), so its fact table and DimReseller/DimEmployee are skipped.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import duckdb

SOURCE_URL = (
    "https://github.com/microsoft/sql-server-samples/releases/download/"
    "adventureworks/AdventureWorksDW-data-warehouse-install-script.zip"
)

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"          # gitignored
DB_PATH = DATA_DIR / "adventureworks.duckdb"  # gitignored

TABLES = [
    "FactInternetSales",
    "DimDate",
    "DimProduct",
    "DimProductSubcategory",
    "DimProductCategory",
    "DimCustomer",
    "DimGeography",
    "DimPromotion",
    "DimSalesTerritory",
    "DimCurrency",
]

# SQL Server -> DuckDB. tinyint is unsigned 0-255 in SQL Server, so it needs SMALLINT here.
TYPE_MAP = {
    "int": "INTEGER",
    "smallint": "SMALLINT",
    "tinyint": "SMALLINT",
    "bigint": "BIGINT",
    "bit": "BOOLEAN",
    "money": "DECIMAL(19,4)",
    "float": "DOUBLE",
    "date": "DATE",
    "datetime": "TIMESTAMP",
    "nvarchar": "VARCHAR",
    "nchar": "VARCHAR",
    "varchar": "VARCHAR",
    "char": "VARCHAR",
}
DROPPED_TYPES = {"varbinary"}


def download() -> None:
    RAW_DIR.mkdir(exist_ok=True)
    if (RAW_DIR / "instawdbdw.sql").exists():
        return
    zip_path = RAW_DIR / "awdw.zip"
    print(f"Downloading {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)


def parse_columns(ddl_script: str, table: str) -> list[tuple[str, str | None]]:
    """Return [(column, duckdb_type or None if dropped)] from the CREATE TABLE statement."""
    m = re.search(rf"CREATE TABLE \[dbo\]\.\[{table}\]\((.*?)\) ON", ddl_script, re.S)
    if not m:
        raise RuntimeError(f"No CREATE TABLE for {table} in install script")
    cols = []
    for line in m.group(1).splitlines():
        cm = re.match(r"\s*\[(\w+)\]\s*\[(\w+)\]", line)
        if not cm:
            continue
        name, sql_type = cm.groups()
        if sql_type in DROPPED_TYPES:
            cols.append((name, None))
        elif sql_type in TYPE_MAP:
            cols.append((name, TYPE_MAP[sql_type]))
        else:
            raise RuntimeError(f"Unmapped type {sql_type} for {table}.{name}")
    return cols


def read_rows(path: Path, n_cols: int) -> list[list[str | None]]:
    """Pipe-delimited, no quoting. Some files use CRLF and some contain NUL bytes
    standing in for empty strings, so clean those before splitting."""
    text = path.read_bytes().decode("utf-8").replace("\x00", "").replace("\r\n", "\n")
    rows = []
    for i, line in enumerate(text.split("\n")):
        if line == "":
            continue
        fields = line.split("|")
        if len(fields) != n_cols:
            raise RuntimeError(f"{path.name} line {i + 1}: {len(fields)} fields, expected {n_cols}")
        rows.append([f if f != "" else None for f in fields])
    return rows


def build() -> dict[str, int]:
    """Rebuild the database from scratch. Returns source row counts per table."""
    download()
    ddl = (RAW_DIR / "instawdbdw.sql").read_text(encoding="utf-8", errors="replace")

    if DB_PATH.exists():
        DB_PATH.unlink()
    con = duckdb.connect(str(DB_PATH))
    source_counts = {}

    for table in TABLES:
        cols = parse_columns(ddl, table)
        rows = read_rows(RAW_DIR / f"{table}.csv", len(cols))
        source_counts[table] = len(rows)

        keep = [i for i, (_, t) in enumerate(cols) if t is not None]
        kept_cols = [cols[i] for i in keep]

        # Write the cleaned rows out as a tab-separated file and let DuckDB read it
        # all as VARCHAR (row-by-row inserts take minutes; this takes seconds), then
        # cast in one typed CREATE TABLE AS so a bad value fails naming its column.
        clean_path = RAW_DIR / f"{table}.clean.tsv"
        with clean_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write("\t".join("" if r[i] is None else r[i] for i in keep) + "\n")
        col_spec = "{" + ", ".join(f"'{c}': 'VARCHAR'" for c, _ in kept_cols) + "}"
        casts = ", ".join(f"CAST(NULLIF({c}, '') AS {t}) AS {c}" for c, t in kept_cols)
        con.execute(
            f"CREATE TABLE {table} AS SELECT {casts} FROM read_csv(?, delim='\\t', "
            f"quote='', escape='', header=false, all_varchar=true, columns={col_spec})",
            [str(clean_path)],
        )
        clean_path.unlink()
        print(f"  {table:<24} {len(rows):>7} rows")

    con.close()
    return source_counts


def verify(source_counts: dict[str, int]) -> bool:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    ok = True
    for table, expected in source_counts.items():
        actual = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        status = "ok" if actual == expected else "MISMATCH"
        ok &= actual == expected
        print(f"  {table:<24} source {expected:>7}  db {actual:>7}  {status}")

    # Referential checks: every fact row must reach each dimension, otherwise
    # the semantic layer's inner joins would silently drop sales.
    orphans = {
        "product": "SELECT COUNT(*) FROM FactInternetSales f LEFT JOIN DimProduct p USING (ProductKey) WHERE p.ProductKey IS NULL",
        "order date": "SELECT COUNT(*) FROM FactInternetSales f LEFT JOIN DimDate d ON f.OrderDateKey = d.DateKey WHERE d.DateKey IS NULL",
        "customer": "SELECT COUNT(*) FROM FactInternetSales f LEFT JOIN DimCustomer c USING (CustomerKey) WHERE c.CustomerKey IS NULL",
        "territory": "SELECT COUNT(*) FROM FactInternetSales f LEFT JOIN DimSalesTerritory t USING (SalesTerritoryKey) WHERE t.SalesTerritoryKey IS NULL",
    }
    for name, sql in orphans.items():
        n = con.execute(sql).fetchone()[0]
        ok &= n == 0
        print(f"  fact rows with no {name:<11} {n:>7}  {'ok' if n == 0 else 'FAIL'}")
    con.close()
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="check row counts after building")
    args = parser.parse_args()

    print(f"Building {DB_PATH}")
    counts = build()
    if args.verify:
        print("Verifying")
        sys.exit(0 if verify(counts) else 1)
