"""Build a Plotly figure spec from a query result.

The agent names a chart type and which result columns go where; this module turns
that into plain Plotly JSON ({"data": [...], "layout": {...}}). No plotting code is
generated or run, so a bad request is a validation error, never a crash. The spec
renders the same way in Streamlit (st.plotly_chart) or a static HTML page.
"""

from __future__ import annotations

import html
import json
from typing import Any

from semantic import QueryResult

CHART_TYPES = ["bar", "line", "scatter", "pie"]
TICK_FORMATS = {"currency": "$,.0f", "percent": ".1%", "number": ",.0f"}


class ChartError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid chart:\n- " + "\n- ".join(errors))


def build_chart(result: QueryResult | None, chart_type: str, x: str, y: Any,
                series: str | None = None, title: str = "") -> dict:
    if result is None:
        raise ChartError(["There is no query result to chart yet. Call run_query first"])

    cols = {c.name: c for c in result.columns}
    dims = [c.name for c in result.columns if c.role == "dimension"]
    measures = [c.name for c in result.columns if c.role == "measure"]
    ys = [y] if isinstance(y, str) else list(y or [])

    errors = []
    if chart_type not in CHART_TYPES:
        errors.append(f"chart_type '{chart_type}' must be one of {CHART_TYPES}")
    if x not in cols:
        errors.append(f"x '{x}' is not a column of the last result. Columns: {list(cols)}")
    if not ys:
        errors.append(f"y needs at least one measure column. Measures in the last result: {measures}")
    for col in ys:
        if col not in measures:
            errors.append(f"y '{col}' is not a measure column of the last result. Measures: {measures}")
    if series is not None:
        if series not in dims or series == x:
            errors.append(f"series '{series}' must be a dimension column other than x. Dimensions: {dims}")
        elif len(ys) > 1:
            errors.append("Use either several y columns or a series, not both")
    if chart_type == "pie" and (len(ys) != 1 or series):
        errors.append("A pie chart takes exactly one y column and no series")
    if errors:
        raise ChartError(errors)

    records = result.records()
    if cols[x].type == "time":
        records = sorted(records, key=lambda r: (r[x] is None, r[x]))

    if chart_type == "pie":
        data = [{"type": "pie", "labels": [r[x] for r in records], "values": [r[ys[0]] for r in records]}]
    elif series:
        groups: dict[Any, list[dict]] = {}
        for r in records:
            groups.setdefault(r[series], []).append(r)
        data = [_trace(chart_type, str(name), rows, x, ys[0]) for name, rows in groups.items()]
    else:
        data = [_trace(chart_type, _label(col), records, x, col) for col in ys]

    layout: dict[str, Any] = {
        "title": {"text": title or _default_title(ys, x, series)},
        "showlegend": len(data) > 1 or chart_type == "pie",
        "margin": {"t": 60, "r": 20, "b": 60, "l": 70},
        "meta": {"chart_type": chart_type},   # Plotly draws lines as "scatter" traces
    }
    if chart_type != "pie":
        layout["xaxis"] = {"title": {"text": _label(x)}, "automargin": True}
        layout["yaxis"] = {"title": {"text": " / ".join(_label(c) for c in ys)}, "automargin": True}
        fmt = TICK_FORMATS.get(cols[ys[0]].format)
        if fmt and len({cols[c].format for c in ys}) == 1:
            layout["yaxis"]["tickformat"] = fmt
        if chart_type == "bar" and len(data) > 1:
            layout["barmode"] = "group"   # clustered columns, side by side
    return {"data": data, "layout": layout}


def _trace(chart_type: str, name: str, rows: list[dict], x: str, y: str) -> dict:
    trace = {"type": "bar" if chart_type == "bar" else "scatter", "name": name,
             "x": [r[x] for r in rows], "y": [r[y] for r in rows]}
    if chart_type == "line":
        trace["mode"] = "lines+markers"
    elif chart_type == "scatter":
        trace["mode"] = "markers"
    return trace


def _label(col: str) -> str:
    name, _, grain = col.partition("__")
    label = name.replace("_", " ").capitalize()
    return f"{label} ({grain})" if grain else label


def _default_title(ys: list[str], x: str, series: str | None) -> str:
    names = [_label(c) for c in ys]
    names = names[:1] + [n.lower() for n in names[1:]]   # "Revenue and gross margin", not "...Gross margin"
    t = f"{' and '.join(names)} by {_label(x).lower()}"
    return t + (f" and {_label(series).lower()}" if series else "")


def to_html(spec: dict, heading: str = "") -> str:
    """Standalone page for viewing a chart outside the app, e.g. from the CLI."""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(heading or 'Chart')}</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>body{{font-family:system-ui,sans-serif;margin:24px;}} #chart{{height:520px;}}</style>
</head><body>
<p>{html.escape(heading)}</p>
<div id="chart"></div>
<script>const spec = {json.dumps(spec)};
Plotly.newPlot("chart", spec.data, spec.layout, {{responsive: true}});</script>
</body></html>
"""


def default_chart(result: QueryResult | None) -> dict | None:
    """A sensible chart for a result the agent didn't chart itself: a line over time
    if there's a time column, otherwise a bar. None when a chart wouldn't help
    (a single number, or nothing to plot). Small models often forget make_chart,
    so this keeps charts from depending on the model remembering."""
    if result is None or len(result.rows) < 2:
        return None
    dims = [c for c in result.columns if c.role == "dimension"]
    measures = [c.name for c in result.columns if c.role == "measure"]
    if not dims or not measures or len(dims) > 2:
        return None
    time = next((d for d in dims if d.type == "time"), None)
    x = time or dims[0]
    other = next((d for d in dims if d is not x), None)
    if other and len({r[other.name] for r in result.records()}) > 10:
        return None   # too many series to read
    if other:
        ys = [measures[0]]   # one measure, split into a series per value of the other dimension
    else:
        # Every measure in the same unit as the first, side by side (revenue and margin
        # together; not revenue next to a margin %, which would need a second axis).
        fmt = {c.name: c.format for c in result.columns}
        ys = [m for m in measures if fmt[m] == fmt[measures[0]]][:4]
    try:
        spec = build_chart(result, "line" if time else "bar", x.name, ys,
                           series=other.name if other else None)
    except ChartError:
        return None
    spec["layout"]["meta"]["auto"] = True
    return spec
