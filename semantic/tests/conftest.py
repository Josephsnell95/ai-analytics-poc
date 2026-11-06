import copy
from pathlib import Path

import pytest
import yaml

from semantic import SemanticEngine, SemanticModel

REPO = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO / "model" / "semantic_model.yaml"
DB_PATH = REPO / "data" / "adventureworks.duckdb"


@pytest.fixture(scope="session")
def model_dict():
    with open(MODEL_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def mutable_model_dict(model_dict):
    """A fresh copy of the real model to break in a specific way."""
    return copy.deepcopy(model_dict)


@pytest.fixture(scope="session")
def model():
    return SemanticModel.load(MODEL_PATH)


@pytest.fixture(scope="session")
def engine(model):
    if not DB_PATH.exists():
        pytest.skip("Database not built; run python data/load.py")
    e = SemanticEngine(model, DB_PATH, default_limit=100, max_limit=1000)
    yield e
    e.close()
