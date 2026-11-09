"""One interface over Ollama (local) and the Anthropic API.

The agent loop speaks a small neutral message format, and each provider converts
to and from its own wire format. Plain `requests` rather than SDKs or LiteLLM, so
the whole translation is visible in this one file.

Neutral messages:
    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Neutral tools:
    {"name": str, "description": str, "parameters": <JSON schema>}
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from .config import Settings, env


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderError(Exception):
    """The provider could not be reached or returned an error. Message is for humans."""


class Provider(Protocol):
    name: str
    model: str

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


# --------------------------------------------------------------------- Ollama
class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, host: str, num_ctx: int, keep_alive: str,
                 temperature: float, timeout: float):
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + [self._to_wire(m) for m in messages],
            "tools": [{"type": "function", "function": t} for t in tools],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise ProviderError(
                f"Can't reach Ollama at {self.host}. Is it running? (Start the Ollama app, "
                "or run 'ollama serve')"
            ) from e
        except requests.Timeout as e:
            raise ProviderError(f"Ollama took longer than {self.timeout}s to respond") from e
        if r.status_code == 404 and "not found" in r.text:
            raise ProviderError(f"Model '{self.model}' isn't pulled. Run: ollama pull {self.model}")
        if not r.ok:
            raise ProviderError(f"Ollama returned {r.status_code}: {r.text[:300]}")
        return self._from_wire(r.json(), {t["name"] for t in tools})

    @staticmethod
    def _to_wire(m: dict) -> dict:
        if m["role"] == "assistant":
            out = {"role": "assistant", "content": m.get("content", "")}
            if m.get("tool_calls"):
                out["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in m["tool_calls"]
                ]
            return out
        if m["role"] == "tool":
            return {"role": "tool", "content": m["content"], "tool_name": m["name"]}
        return {"role": m["role"], "content": m["content"]}

    @staticmethod
    def _from_wire(data: dict, tool_names: set[str]) -> LLMResponse:
        msg = data.get("message", {})
        text = msg.get("content", "") or ""
        calls = []
        for c in msg.get("tool_calls", []) or []:
            fn = c.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = _loads_or_empty(args)
            calls.append(ToolCall(c.get("id") or _new_id(), fn.get("name", ""), args))

        # Llama 3.1 sometimes writes the tool call as JSON in its reply instead of
        # using the tool-call channel. Recover it if it names a real tool.
        if not calls:
            recovered = _tool_call_from_text(text, tool_names)
            if recovered:
                calls, text = [recovered], ""

        return LLMResponse(
            text=text,
            tool_calls=calls,
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
        )


# ------------------------------------------------------------------ Anthropic
class AnthropicProvider:
    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, model: str, api_key: str, max_tokens: int, timeout: float):
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set. Add it to the .env file in the repo root")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._to_wire(messages),
            "tools": [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }
        try:
            r = requests.post(self.URL, json=body, headers=headers, timeout=self.timeout)
        except requests.RequestException as e:
            raise ProviderError(f"Can't reach the Anthropic API: {e}") from e
        if r.status_code == 401:
            raise ProviderError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env")
        if not r.ok:
            raise ProviderError(f"Anthropic returned {r.status_code}: {r.text[:300]}")
        return self._from_wire(r.json())

    @staticmethod
    def _to_wire(messages: list[dict]) -> list[dict]:
        """Anthropic puts tool calls in the assistant's content blocks, and all tool
        results that answer one assistant turn into a single user message."""
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in m.get("tool_calls", []) or []:
                    blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    @staticmethod
    def _from_wire(data: dict) -> LLMResponse:
        text_parts, calls = [], []
        for block in data.get("content", []):
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                calls.append(ToolCall(block["id"], block["name"], block.get("input") or {}))
        usage = data.get("usage", {})
        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )


# ------------------------------------------------------------------ factory
def make_provider(name: str, settings: Settings) -> Provider:
    cfg = settings.provider(name)
    if name == "ollama":
        return OllamaProvider(
            model=cfg["model"],
            host=env("OLLAMA_HOST", "http://localhost:11434"),
            num_ctx=cfg["num_ctx"],
            keep_alive=cfg["keep_alive"],
            temperature=cfg.get("temperature", 0),
            timeout=cfg["timeout_seconds"],
        )
    if name == "anthropic":
        return AnthropicProvider(
            model=cfg["model"],
            api_key=env("ANTHROPIC_API_KEY", ""),
            max_tokens=cfg["max_tokens"],
            timeout=cfg["timeout_seconds"],
        )
    raise ValueError(f"No provider implementation for '{name}'")


# ------------------------------------------------------------------ helpers
RUN_QUERY_KEYS = {"measures", "dimensions", "filters", "sort", "limit", "assumptions"}


def _new_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


def _loads_or_empty(s: str) -> dict:
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def _tool_call_from_text(text: str, tool_names: set[str]) -> ToolCall | None:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").removeprefix("json").strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    name = obj.get("name") or obj.get("function")
    args = obj.get("parameters", obj.get("arguments", {}))
    if name in tool_names and isinstance(args, dict):
        return ToolCall(_new_id(), name, args)
    # A bare semantic query as the reply: {"measures": [...], ...} with nothing else.
    # Only run_query takes these keys, so there's no doubt about which tool was meant.
    if "run_query" in tool_names and "measures" in obj and set(obj) <= RUN_QUERY_KEYS:
        return ToolCall(_new_id(), "run_query", obj)
    return None


# --------------------------------------------------------------- availability
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
