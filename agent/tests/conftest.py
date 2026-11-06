from pathlib import Path

import pytest

from agent.config import load_settings
from agent.providers import LLMResponse, ToolCall
from semantic import SemanticEngine

REPO = Path(__file__).resolve().parents[2]


class ScriptedProvider:
    """Plays back a fixed list of responses and records what it was sent."""

    name = "fake"
    model = "llama3.1:8b"   # priced at zero in prices.yaml

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, system, messages, tools):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        if not self.responses:
            raise AssertionError("ScriptedProvider ran out of responses")
        return self.responses.pop(0)


def tool(name, **arguments):
    return LLMResponse(text="", tool_calls=[ToolCall(f"id_{name}", name, arguments)],
                       input_tokens=1000, output_tokens=50)


def say(text):
    return LLMResponse(text=text, input_tokens=1200, output_tokens=80)


@pytest.fixture(scope="session")
def settings():
    return load_settings(REPO)


@pytest.fixture(scope="session")
def engine(settings):
    if not (REPO / "data" / "adventureworks.duckdb").exists():
        pytest.skip("Database not built; run python data/load.py")
    e = SemanticEngine.from_config(REPO / "config.yaml")
    yield e
    e.close()
