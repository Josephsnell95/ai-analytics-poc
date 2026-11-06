"""Rendering helpers for the Streamlit app. Kept apart from main.py so the page
script stays a readable top-to-bottom description of the screen."""

from __future__ import annotations

import copy
import datetime as dt
import json

import requests
import streamlit as st

from agent.config import Settings, env
from agent.loop import AgentResult

PROVIDER_LABELS = {
    "ollama": "Llama 3.1 8B (local, free)",
    "anthropic": "Claude Haiku (API, paid)",
}

EXAMPLE_QUESTIONS = [
    "What was total revenue in 2013?",
    "Plot monthly revenue for 2013",
    "Which 5 product models had the highest revenue?",
    "Average order value by territory group",
]


# ------------------------------------------------------------ availability
def provider_status(name: str, settings: Settings) -> tuple[bool, str]:
    """Whether a provider can be used right now, and a one-line reason."""
    cfg = settings.provider(name)
    if name == "anthropic":
        if env("ANTHROPIC_API_KEY"):
            return True, f"API key found · {cfg['model']}"
        return False, "No ANTHROPIC_API_KEY in .env"
    host = env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        r = requests.get(f"{host}/api/tags", timeout=1.5)
        models = {m.get("name") for m in r.json().get("models", [])}
    except (requests.RequestException, ValueError):
        return False, f"Ollama isn't running at {host}. Start the Ollama app"
    if cfg["model"] not in models:
        return False, f"Run: ollama pull {cfg['model']}"
    return True, f"Ollama running · {cfg['model']}"


# ------------------------------------------------------------------ result
def render_result(r: AgentResult, key: str) -> None:
    """One assistant answer: the answer itself first, then evidence, then cost."""
    if r.status == "error":
        st.error(r.answer)
    elif r.status in ("declined", "capped"):
        st.warning(r.answer)
    else:
        if r.assumptions:
            st.info("**Assumed:** " + "; ".join(r.assumptions), icon=":material/lightbulb:")
        st.markdown(_escape(r.answer))
        if r.unverified:
            figures = ", ".join(_escape(f) for f in r.unverified)
            st.warning(f"**Check these figures:** {figures} {'doesn' if len(r.unverified) == 1 else 'don'}'t "
                       "match the query results. The data table below has the exact values.",
                       icon=":material/fact_check:")
        if r.status == "no_query":
            st.caption(":material/warning: This answer wasn't backed by a query. Treat any numbers with caution.")

    if r.chart:
        st.plotly_chart(_static_chart(r.chart), key=f"chart-{key}", config={"displayModeBar": False})
        if r.chart_auto:
            st.caption("Default chart: the model didn't ask for one.")

    if r.rows:
        with st.expander(f"Data · {len(r.rows)} row{'s' * (len(r.rows) != 1)}"):
            render_table(r)

    if r.semantic_query or r.attempts:
        with st.expander("How this was answered"):
            render_working(r)

    st.caption(cost_line(r))


def render_table(r: AgentResult) -> None:
    formats = r.column_formats
    rows = [{c: _display_value(c, row[c], formats.get(c)) for c in r.columns} for row in r.rows]
    config = {}
    for c in r.columns:
        label = column_label(c)
        # Percent values are scaled to 0-100 in _display_value so "%.1f%%" gives one decimal
        # everywhere; Streamlit's "percent" preset varies the decimals row by row.
        fmt = {"currency": "dollar", "percent": "%.1f%%", "number": "localized"}.get(formats.get(c))
        config[c] = st.column_config.NumberColumn(label, format=fmt) if fmt else st.column_config.TextColumn(label)
    st.dataframe(rows, column_config=config, hide_index=True)


def render_working(r: AgentResult) -> None:
    rejected = [a for a in r.attempts if a.error]
    if rejected:
        st.markdown(f"**{len(rejected)} rejected quer{'ies' if len(rejected) > 1 else 'y'}** "
                    "(the semantic layer explained the problem and the model tried again):")
        for a in rejected:
            st.code(_compact_json(a.query), language="json", wrap_lines=True)
            st.caption(a.error.replace("Invalid semantic query:\n", "").replace("\n", "  \n"))
    if r.semantic_query:
        st.markdown("**Semantic query** (what the model asked for)")
        st.code(_compact_json(r.semantic_query), language="json", wrap_lines=True)
        st.markdown("**SQL** (what the compiler generated from it)")
        st.code(r.sql, language="sql", wrap_lines=True)


def cost_line(r: AgentResult) -> str:
    u = r.usage
    if r.declined_by == "guard":
        return f"Declined by the semantic model's known-gaps list · no model call · $0 · {u.latency_s:.1f}s"
    rejected = sum(1 for a in r.attempts if a.error)
    parts = [
        r.model,
        f"{u.llm_calls} model call{'s' * (u.llm_calls != 1)}",
        f"{u.tool_calls} tool call{'s' * (u.tool_calls != 1)}" + (f" ({rejected} rejected)" if rejected else ""),
        f"{u.input_tokens + u.output_tokens:,} tokens",
        "free" if u.cost_usd == 0 else f"${u.cost_usd:.4f}",
        f"{u.latency_s:.1f}s",
    ]
    return " · ".join(parts)


# ----------------------------------------------------------------- helpers
def column_label(col: str) -> str:
    name, _, grain = col.partition("__")
    label = name.removesuffix("_pct").replace("_", " ").capitalize() + (" %" if name.endswith("_pct") else "")
    return f"{label} ({grain})" if grain else label


def _display_value(col: str, v, fmt: str | None = None):
    """Time columns come back as the first day of the period; show the period.
    Percent measures are scaled to 0-100 for display."""
    if fmt == "percent" and isinstance(v, (int, float)):
        return v * 100
    grain = col.partition("__")[2]
    if not grain or not isinstance(v, str):
        return v
    try:
        d = dt.date.fromisoformat(v)
    except ValueError:
        return v
    return {
        "year": f"{d.year}",
        "quarter": f"{d.year} Q{(d.month - 1) // 3 + 1}",
        "month": d.strftime("%b %Y"),
        "week": f"w/c {d.isoformat()}",
    }.get(grain, d.isoformat())


def _static_chart(spec: dict) -> dict:
    """No zoom or pan: dragging a chart in a chat by accident is more annoying than useful."""
    spec = copy.deepcopy(spec)
    layout = spec["layout"]
    for axis in ("xaxis", "yaxis"):
        if axis in layout:
            layout[axis]["fixedrange"] = True
    layout.pop("meta", None)
    return spec


def _compact_json(obj: dict) -> str:
    """One line per top-level key: readable without the vertical sprawl of indent=2."""
    lines = [f'  "{k}": {json.dumps(v)}' for k, v in obj.items()]
    return "{\n" + ",\n".join(lines) + "\n}"


def _escape(text: str) -> str:
    # Streamlit markdown treats $...$ as LaTeX, which mangles "$1,234 and $5,678".
    return text.replace("$", r"\$")
