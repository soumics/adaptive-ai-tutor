"""LLM access for the tutor: a local model by default, an API model as an option.

The two back-ends are Project 3's (``rag_assistant.generate``), imported unchanged:

* ``ollama`` (default): a local model served by Ollama, e.g. ``qwen3.5:4b``. Free, private, runs
  without any account or key; weaker at long reasoning and strict formats than large API models.
* ``anthropic`` (optional): Claude through the official SDK. Better output, but it needs an
  ``ANTHROPIC_API_KEY`` (set in the environment, never in the code) and every call costs money.

On top of plain text generation the tutor needs **structured output** (questions, rubrics,
per-point grading decisions) as JSON. With Ollama the JSON schema is passed to the server, which
constrains the model's output to it. With other back-ends the schema is put in the prompt and the
reply parsed. Either way, the result is validated by the caller before it is trusted.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from rag_assistant.generate import LLM, LLMResult, OllamaLLM, make_llm

DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"


class StructuredOutputError(ValueError):
    pass


def extract_json(text: str) -> Any:
    """The first JSON object in a reply (tolerates code fences and text around it)."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise StructuredOutputError("no JSON object in the reply")


@dataclass
class ModelProvider:
    """Wraps a Project 3 LLM; adds JSON generation and a per-call temperature for Ollama."""

    llm: LLM

    @property
    def name(self) -> str:
        return self.llm.name

    @property
    def is_local(self) -> bool:
        return isinstance(self.llm, OllamaLLM)

    def text(self, system: str, user: str) -> LLMResult:
        return self.llm.generate(system, user)

    def json(self, system: str, user: str, schema: dict, temperature: float | None = None, seed: int | None = None,
             retries: int = 1) -> dict:
        """Generate a JSON object matching ``schema`` (checked for the schema's required keys)."""
        last = None
        for attempt in range(retries + 1):
            try:
                if isinstance(self.llm, OllamaLLM):
                    raw = self._ollama_json(system, user, schema, temperature, None if seed is None else seed + attempt)
                else:
                    instruction = ("\n\nReply with a single JSON object and nothing else. It must match this JSON "
                                   f"schema:\n{json.dumps(schema)}")
                    raw = self.llm.generate(system + instruction, user).text
                data = extract_json(raw)
                missing = [k for k in schema.get("required", []) if k not in data]
                if missing:
                    raise StructuredOutputError(f"missing keys: {missing}")
                return data
            except StructuredOutputError as e:
                last = e
        raise last

    def _ollama_json(self, system: str, user: str, schema: dict, temperature: float | None, seed: int | None) -> str:
        import requests

        llm = self.llm
        options = {"temperature": llm.temperature if temperature is None else temperature, "num_ctx": llm.num_ctx}
        if seed is not None:
            options["seed"] = seed
        t0 = time.perf_counter()
        r = requests.post(f"{llm.base_url}/api/chat", timeout=llm.timeout_s, json={
            "model": llm.model, "stream": False, "think": False, "format": schema, "options": options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        self.last_seconds = time.perf_counter() - t0
        return r.json()["message"]["content"]


def make_provider(kind: str = "ollama", model: str | None = None, base_url: str = "http://localhost:11434",
                  temperature: float = 0.2) -> ModelProvider:
    """``ollama`` (default, local) or ``anthropic`` (optional; reads ANTHROPIC_API_KEY from the environment)."""
    if kind == "ollama":
        return ModelProvider(OllamaLLM(model=model or DEFAULT_OLLAMA_MODEL, base_url=base_url, temperature=temperature))
    if kind == "anthropic":
        return ModelProvider(make_llm("anthropic", model=model))
    raise ValueError(f"unknown provider {kind!r} (use 'ollama' or 'anthropic')")
