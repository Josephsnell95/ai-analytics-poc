"""The question files themselves: valid, and consistent with the guard."""

from pathlib import Path

import pytest

from evals.runner import check_questions, load_questions
from semantic import SemanticEngine

CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


@pytest.fixture(scope="module")
def engine():
    if not (CONFIG.parent / "data" / "adventureworks.duckdb").exists():
        pytest.skip("Database not built")
    return SemanticEngine.from_config(CONFIG)


@pytest.mark.parametrize("set_name", ["tuning", "heldout"])
def test_question_files_valid(engine, set_name):
    assert check_questions(load_questions(set_name), engine) == []


def test_out_of_scope_categories_match_the_guard(engine):
    for q in load_questions("tuning"):
        guarded = engine.model.unavailable_topic(q["question"]) is not None
        if q["category"] == "out_of_scope_listed":
            assert guarded, f"{q['id']} should be caught by not_available"
        else:
            assert not guarded, f"{q['id']} is blocked by the guard but is meant to be {q['category']}"


def test_every_category_covered():
    cats = {q["category"] for q in load_questions("tuning")}
    assert cats >= {"lookup", "breakdown", "ranking", "time", "synonym", "follow_up", "ambiguous",
                    "out_of_scope_listed", "out_of_scope_unlisted"}


def test_check_reports_problems(engine):
    bad = [{"id": "a", "category": "lookup", "question": "x", "reference": {"measures": ["nope"]}, "set": "t"},
           {"id": "a", "category": "lookup", "question": "y", "expect": "decline", "set": "t"},
           {"id": "b", "category": "follow_up", "question": "z", "follow_up_of": "later", "set": "t",
            "reference": {"measures": ["revenue"]}}]
    problems = " | ".join(check_questions(bad, engine))
    assert "reference is invalid" in problems and "duplicate id" in problems and "must be an earlier question" in problems
