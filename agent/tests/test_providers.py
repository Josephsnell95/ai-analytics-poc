"""Wire-format conversion for both providers, with HTTP mocked out. The response
shapes are taken from the Ollama /api/chat and Anthropic Messages API docs."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from agent.providers import AnthropicProvider, OllamaProvider, ProviderError, ToolCall
from agent.tools import TOOLS, TOOL_NAMES

CONVERSATION = [
    {"role": "user", "content": "Revenue by category?"},
    {"role": "assistant", "content": "", "tool_calls": [
        ToolCall("c1", "run_query", {"measures": ["revenue"], "dimensions": ["product_category"]}),
        ToolCall("c2", "get_model", {"search": "category"}),
    ]},
    {"role": "tool", "tool_call_id": "c1", "name": "run_query", "content": '{"rows": []}'},
    {"role": "tool", "tool_call_id": "c2", "name": "get_model", "content": "MEASURES..."},
]


def ok(json_body):
    r = MagicMock(ok=True, status_code=200)
    r.json.return_value = json_body
    return r


# -------------------------------------------------------------------- Ollama
def ollama():
    return OllamaProvider("llama3.1:8b", "http://localhost:11434/", 8192, "30m", 0, 300)


def test_ollama_request_shape():
    with patch("agent.providers.requests.post", return_value=ok({"message": {"content": "hi"}})) as post:
        ollama().chat("SYSTEM", CONVERSATION, TOOLS)
    url, body = post.call_args.args[0], post.call_args.kwargs["json"]
    assert url == "http://localhost:11434/api/chat"
    assert body["options"] == {"temperature": 0, "num_ctx": 8192} and body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert body["messages"][2]["tool_calls"][0]["function"]["name"] == "run_query"
    assert body["messages"][3] == {"role": "tool", "content": '{"rows": []}', "tool_name": "run_query"}
    assert body["tools"][0]["type"] == "function"


def test_ollama_parses_tool_calls_and_tokens():
    data = {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "run_query", "arguments": {"measures": ["revenue"]}}}]},
        "prompt_eval_count": 2100, "eval_count": 40}
    r = OllamaProvider._from_wire(data, TOOL_NAMES)
    assert r.tool_calls[0].name == "run_query" and r.tool_calls[0].arguments == {"measures": ["revenue"]}
    assert r.tool_calls[0].id.startswith("call_")
    assert (r.input_tokens, r.output_tokens) == (2100, 40)


def test_ollama_string_arguments_are_parsed():
    data = {"message": {"tool_calls": [{"function": {"name": "run_query", "arguments": '{"measures": ["revenue"]}'}}]}}
    assert OllamaProvider._from_wire(data, TOOL_NAMES).tool_calls[0].arguments == {"measures": ["revenue"]}


@pytest.mark.parametrize("content", [
    '{"name": "run_query", "parameters": {"measures": ["revenue"]}}',
    '```json\n{"name": "run_query", "arguments": {"measures": ["revenue"]}}\n```',
])
def test_ollama_recovers_tool_call_written_as_text(content):
    r = OllamaProvider._from_wire({"message": {"content": content}}, TOOL_NAMES)
    assert r.text == "" and r.tool_calls[0].arguments == {"measures": ["revenue"]}


def test_ollama_plain_json_answer_not_mistaken_for_tool_call():
    r = OllamaProvider._from_wire({"message": {"content": '{"name": "Bikes"}'}}, TOOL_NAMES)
    assert r.tool_calls == [] and r.text == '{"name": "Bikes"}'


def test_ollama_not_running():
    with patch("agent.providers.requests.post", side_effect=requests.ConnectionError()):
        with pytest.raises(ProviderError, match="Is it running"):
            ollama().chat("s", [], TOOLS)


def test_ollama_model_not_pulled():
    resp = MagicMock(ok=False, status_code=404, text='{"error":"model \\"llama3.1:8b\\" not found"}')
    with patch("agent.providers.requests.post", return_value=resp):
        with pytest.raises(ProviderError, match="ollama pull llama3.1:8b"):
            ollama().chat("s", [], TOOLS)


# ----------------------------------------------------------------- Anthropic
def anthropic():
    return AnthropicProvider("claude-haiku-5-5", "sk-test", 1024, 60)


def test_anthropic_requires_key():
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider("claude-haiku-5-5", "", 1024, 60)


def test_anthropic_request_shape():
    reply = {"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 1, "output_tokens": 1}}
    with patch("agent.providers.requests.post", return_value=ok(reply)) as post:
        anthropic().chat("SYSTEM", CONVERSATION, TOOLS)
    body, headers = post.call_args.kwargs["json"], post.call_args.kwargs["headers"]
    assert headers["x-api-key"] == "sk-test" and headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "SYSTEM" and body["model"] == "claude-haiku-5-5"
    assert "input_schema" in body["tools"][0]
    assert "temperature" not in body   # current Claude models reject it with a 400

    msgs = body["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [b["type"] for b in msgs[1]["content"]] == ["tool_use", "tool_use"]
    # Both tool results answer one assistant turn, so they share one user message.
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["c1", "c2"]


def test_anthropic_parses_text_and_tool_use():
    data = {"content": [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_1", "name": "run_query", "input": {"measures": ["revenue"]}},
    ], "usage": {"input_tokens": 2500, "output_tokens": 60}, "stop_reason": "tool_use"}
    r = AnthropicProvider._from_wire(data)
    assert r.text == "Let me check."
    assert r.tool_calls == [ToolCall("toolu_1", "run_query", {"measures": ["revenue"]})]
    assert (r.input_tokens, r.output_tokens) == (2500, 60)


def test_anthropic_bad_key():
    with patch("agent.providers.requests.post", return_value=MagicMock(ok=False, status_code=401, text="")):
        with pytest.raises(ProviderError, match="rejected the API key"):
            anthropic().chat("s", [], TOOLS)
