"""Shared fixtures: both sample curricula, keyword-only indexes (no embedding model needed) and
scripted stand-ins for the LLM, so every test is deterministic and runs without Ollama."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant.generate import LLMResult  # noqa: E402

from curriculum_ingest import ingest_curriculum  # noqa: E402
from model_provider import ModelProvider  # noqa: E402
from retrieval import CurriculumIndex  # noqa: E402

ALGEBRA = ROOT / "examples" / "sample_curricula" / "algebra_unit"
CONTROLS = ROOT / "examples" / "sample_curricula" / "controls_unit"


class ScriptedLLM:
    """Returns the given replies in order (the last one repeats); records every prompt it receives."""

    name = "test/scripted"

    def __init__(self, *replies):
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.prompts: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> LLMResult:
        self.prompts.append((system, user))
        text = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        return LLMResult(text=text, model=self.name)


def scripted(*replies) -> ModelProvider:
    return ModelProvider(ScriptedLLM(*replies))


@pytest.fixture(scope="session")
def controls_cur():
    return ingest_curriculum([CONTROLS])


@pytest.fixture(scope="session")
def algebra_cur():
    return ingest_curriculum([ALGEBRA])


@pytest.fixture(scope="session")
def controls_index(controls_cur):
    return CurriculumIndex(controls_cur, None, None)


@pytest.fixture(scope="session")
def algebra_index(algebra_cur):
    return CurriculumIndex(algebra_cur, None, None)
