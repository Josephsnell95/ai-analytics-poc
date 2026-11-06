"""The agent loop: ask the model, run any tools it calls, repeat until it answers
in text or hits the tool-call cap. One question in, one AgentResult out."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from semantic import SemanticEngine

from .charts import default_chart
from .config import Settings
from .prompt import CANNOT_ANSWER, build_system_prompt
from .providers import Provider, ProviderError, ToolCall
from .tools import TOOLS, QueryAttempt, Tools
from .verify import unverified_figures

# How a question ended. "answered" means the final text was backed by at least one
# successful query; it does not mean the answer is right (that's the eval's job).
ANSWERED, DECLINED, NO_QUERY, CAPPED, ERROR = "answered", "declined", "no_query", "capped", "error"
TOOL_LABELS = {"run_query": "Running a semantic query", "make_chart": "Drawing a chart",
               "get_model": "Looking up the model"}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0


@dataclass
class AgentResult:
    question: str
    provider: str
    model: str
    status: str
    answer: str
    assumptions: list[str] = field(default_factory=list)
    semantic_query: dict | None = None
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    column_formats: dict[str, str | None] = field(default_factory=dict)   # currency, percent, number
    rows: list[dict] = field(default_factory=list)
    chart: dict | None = None
    chart_auto: bool = False   # True when the code drew the chart because the agent didn't
    attempts: list[QueryAttempt] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    declined_by: str | None = None   # "guard" (not_available list, no LLM call) or "model"
    unverified: list[str] = field(default_factory=list)   # figures in the answer not found in the result
    transcript: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Agent:
    def __init__(self, provider: Provider, engine: SemanticEngine, settings: Settings):
        self.provider = provider
        self.engine = engine
        self.settings = settings
        self.max_tool_calls = settings.agent["max_tool_calls"]
        self.history_turns = settings.agent.get("history_turns", 0)
        self.tools = Tools(engine, settings.agent["rows_to_llm"])
        self.system = build_system_prompt(engine.model)

    def ask(self, question: str, on_event: Callable[[str], None] | None = None,
            history: list[AgentResult] | None = None) -> AgentResult:
        """Answer one question.

        on_event, if given, is called with short progress messages ("Running a
        semantic query") so a UI can show what a slow local model is doing.
        history, if given, is the earlier results in this chat, so a follow-up like
        "now by month" can build on the previous query.
        """
        notify = on_event or (lambda _msg: None)
        self.tools.reset()
        messages = self._history_messages(history) + [{"role": "user", "content": question}]
        usage = Usage()
        started = time.perf_counter()
        status, answer, error = ERROR, "", None

        # Known gaps are declined in code, before the model can answer them with
        # something that looks plausible (e.g. internet revenue as "reseller sales").
        gap = self.engine.model.unavailable_topic(question)
        if gap:
            answer = f"{CANNOT_ANSWER}. {gap.reason}"
            messages.append({"role": "assistant", "content": answer})
            usage.latency_s = round(time.perf_counter() - started, 2)
            result = self._result(question, DECLINED, answer, None, usage, messages)
            result.declined_by = "guard"
            return result

        try:
            while True:
                notify("Thinking" if usage.llm_calls == 0 else "Reading the results")
                response = self.provider.chat(self.system, messages, TOOLS)
                usage.llm_calls += 1
                usage.input_tokens += response.input_tokens
                usage.output_tokens += response.output_tokens

                if not response.tool_calls:
                    answer = response.text.strip()
                    messages.append({"role": "assistant", "content": answer})
                    status = self._final_status(answer)
                    break

                messages.append({"role": "assistant", "content": response.text,
                                 "tool_calls": response.tool_calls})
                capped = False
                for call in response.tool_calls:
                    if usage.tool_calls >= self.max_tool_calls:
                        capped = True
                        break
                    usage.tool_calls += 1
                    notify(TOOL_LABELS.get(call.name, f"Calling {call.name}"))
                    rejected_before = sum(1 for a in self.tools.state.attempts if a.error)
                    messages.append(self._run_tool(call))
                    if sum(1 for a in self.tools.state.attempts if a.error) > rejected_before:
                        notify("Query rejected by the semantic layer; the model is fixing it")
                if capped:
                    status = CAPPED
                    answer = (f"I couldn't answer this within {self.max_tool_calls} tool calls. "
                              "Try rephrasing the question more specifically.")
                    break
        except ProviderError as e:
            status, error = ERROR, str(e)
            answer = f"The model provider failed: {e}"

        usage.latency_s = round(time.perf_counter() - started, 2)
        usage.cost_usd = self.settings.cost(self.provider.model, usage.input_tokens, usage.output_tokens)
        result = self._result(question, status, answer, error, usage, messages)
        if status == DECLINED:
            result.declined_by = "model"
        return result

    # ------------------------------------------------------------- pieces
    def _history_messages(self, history: list[AgentResult] | None) -> list[dict]:
        """Earlier turns as plain question/answer pairs, each answer followed by the
        semantic query behind it. No rows or tool messages: the query is enough to
        build on, and the model must re-query rather than reuse old numbers."""
        if not history or self.history_turns <= 0:
            return []
        out = []
        for prev in history[-self.history_turns:]:
            answer = prev.answer
            if prev.semantic_query:
                answer += f"\n\n[Query behind this answer: {json.dumps(prev.semantic_query)}]"
            out += [{"role": "user", "content": prev.question}, {"role": "assistant", "content": answer}]
        return out

    def _run_tool(self, call: ToolCall) -> dict:
        try:
            content = self.tools.call(call.name, call.arguments)
        except Exception as e:  # a tool bug must not kill the loop or lose the log
            content = f"Tool '{call.name}' failed unexpectedly: {e!r}"
        return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}

    def _final_status(self, answer: str) -> str:
        if answer.lower().startswith(CANNOT_ANSWER.lower()[:20]):
            return DECLINED
        return ANSWERED if self.tools.state.last_result is not None else NO_QUERY

    def _result(self, question, status, answer, error, usage, messages) -> AgentResult:
        st = self.tools.state
        result = AgentResult(
            question=question, provider=self.provider.name, model=self.provider.model,
            status=status, answer=answer, error=error, usage=usage, chart=st.chart,
            attempts=st.attempts, transcript=_serialisable(messages),
        )
        if st.last_result is not None:
            result.semantic_query = st.last_query
            result.assumptions = list(st.last_result.compiled.query.assumptions)
            result.sql = st.last_result.compiled.display_sql()
            result.columns = [c.name for c in st.last_result.columns]
            result.column_formats = {c.name: c.format for c in st.last_result.columns}
            result.rows = st.last_result.records()
            if status == ANSWERED:
                result.unverified = unverified_figures(answer, result.rows, result.column_formats)
                if result.chart is None:
                    result.chart = default_chart(st.last_result)
                    result.chart_auto = result.chart is not None
        return result


def _serialisable(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in m["tool_calls"]]
        out.append(m)
    return out
