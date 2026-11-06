"""Streamlit chat over the semantic model.

    streamlit run app/main.py

Follow-ups work: the agent sees the last couple of questions in the chat and
the query behind each answer (history_turns in config.yaml).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # so "agent", "semantic" and "app" import from the repo

import streamlit as st  # noqa: E402

from agent import providers  # noqa: E402
from agent.config import load_settings  # noqa: E402
from agent.log import QuestionLog  # noqa: E402
from agent.loop import Agent  # noqa: E402
from app.ui import EXAMPLE_QUESTIONS, PROVIDER_LABELS, provider_status, render_result  # noqa: E402
from semantic import SemanticEngine  # noqa: E402

st.set_page_config(page_title="AI Analytics", page_icon=":material/insights:", layout="centered")


# ------------------------------------------------------------------- setup
def setup() -> None:
    """Per-session objects. Each browser session gets its own read-only database
    connection, so nothing is shared between threads."""
    if "settings" in st.session_state:
        return
    settings = load_settings(ROOT)
    try:
        engine = SemanticEngine.from_config(ROOT / "config.yaml")
    except FileNotFoundError:
        st.error("The database hasn't been built yet. In a terminal in the project folder, run:\n\n"
                 "`python data/load.py --verify`")
        st.stop()
    st.session_state.update(
        settings=settings,
        engine=engine,
        log=QuestionLog(settings.path("app_state")),
        agents={},        # provider name -> Agent, built on first use
        messages=[],      # list of (question, AgentResult)
        pending=None,     # an example question clicked on the empty screen
    )


def get_agent(name: str) -> Agent:
    agents = st.session_state.agents
    if name not in agents:
        provider = providers.make_provider(name, st.session_state.settings)
        agents[name] = Agent(provider, st.session_state.engine, st.session_state.settings)
    return agents[name]


setup()
settings = st.session_state.settings

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("AI Analytics")
    st.caption("Questions about AdventureWorks internet sales, answered through a governed "
               "semantic model. The AI never writes SQL.")

    names = list(PROVIDER_LABELS)
    default = names.index(settings.agent["default_provider"])
    provider_name = st.radio("Model", names, index=default, format_func=PROVIDER_LABELS.get)
    available, status_text = provider_status(provider_name, settings)
    if available:
        st.caption(f":material/check_circle: {status_text}")
    else:
        st.caption(f":material/error: {status_text}")

    results = [r for _, r in st.session_state.messages]
    if results:
        st.divider()
        total_cost = sum(r.usage.cost_usd for r in results)
        avg_time = sum(r.usage.latency_s for r in results) / len(results)
        left, right = st.columns(2)
        left.metric("Questions", len(results))
        right.metric("Cost", "free" if total_cost == 0 else f"${total_cost:.4f}")
        st.caption(f"Average {avg_time:.0f}s per answer")
        if st.button("New chat", icon=":material/refresh:", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    with st.expander("How it works"):
        st.markdown(
            "1. The model turns your question into a **semantic query**: which defined measures "
            "and dimensions to use.\n"
            "2. The semantic layer checks it and **compiles the SQL**. A query that doesn't fit "
            "the model is rejected with an explanation, and the model tries again.\n"
            "3. DuckDB runs the SQL; the model writes the answer.\n\n"
            "Open **How this was answered** under any answer to see the query and SQL. "
            "Follow-ups like \"now by month\" build on the previous answer's query; "
            "use **New chat** to start fresh."
        )

# -------------------------------------------------------------------- chat
for i, (question, result) in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(question.replace("$", r"\$"))
    with st.chat_message("assistant"):
        render_result(result, key=str(i))

if not st.session_state.messages:
    st.markdown("### Ask about internet sales")
    st.markdown("Revenue, orders, customers, margin and more, by product, customer, territory, "
                "promotion and date, from Dec 2010 to Jan 2014.")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"example-{i}", width="stretch", disabled=not available):
            st.session_state.pending = q
            st.rerun()

typed = st.chat_input("Ask a question about internet sales", disabled=not available)
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    with st.chat_message("user"):
        st.markdown(question.replace("$", r"\$"))
    with st.chat_message("assistant"):
        slow = provider_name == "ollama"
        label = "Thinking (the local model can take a minute or two)..." if slow else "Thinking..."
        with st.status(label, expanded=False) as status:
            try:
                agent = get_agent(provider_name)
                history = [r for _, r in st.session_state.messages]
                result = agent.ask(question, on_event=lambda msg: status.update(label=f"{msg}..."),
                                   history=history)
            except providers.ProviderError as e:
                status.update(label="Couldn't reach the model", state="error")
                st.error(str(e))
                st.stop()
            st.session_state.log.write(result)
            done = {"error": "error"}.get(result.status, "complete")
            status.update(label=f"Done in {result.usage.latency_s:.0f}s", state=done)
        st.session_state.messages.append((question, result))
        st.rerun()   # redraw from history so this answer renders like every earlier one
