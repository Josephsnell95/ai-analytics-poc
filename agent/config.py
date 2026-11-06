"""Load config.yaml, agent/prices.yaml and .env in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    raw: dict
    prices: dict
    root: Path

    @property
    def agent(self) -> dict:
        return self.raw["agent"]

    def provider(self, name: str) -> dict:
        if name not in self.raw["providers"]:
            raise ValueError(f"Unknown provider '{name}'. Options: {sorted(self.raw['providers'])}")
        return self.raw["providers"][name]

    def path(self, key: str) -> Path:
        return self.root / self.raw["paths"][key]

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if model not in self.prices:
            raise KeyError(f"No prices for '{model}' in agent/prices.yaml. Add it (use 0 for local models)")
        p = self.prices[model]
        long = input_tokens > p.get("long_context_threshold", float("inf"))
        rate_in = p["long_context_input"] if long else p["input"]
        rate_out = p["long_context_output"] if long else p["output"]
        return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


def load_settings(root: Path = REPO_ROOT) -> Settings:
    load_dotenv(root / ".env")   # does not override variables already set in the shell
    with open(root / "config.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    with open(root / "agent" / "prices.yaml", encoding="utf-8") as f:
        prices = yaml.safe_load(f)
    return Settings(raw, prices, root)


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value.strip() if isinstance(value, str) else value
