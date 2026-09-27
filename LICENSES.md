# Licences of the sample material and dependencies

The project's own code is MIT-licensed ([LICENSE](LICENSE)).

## Sample curricula

| Material | Source | Licence |
|---|---|---|
| `examples/sample_curricula/algebra_unit/Elementary_Algebra_Ch2_Linear_Equations.pdf` | Pages 205-256 (Chapter 2, sections 2.1-2.4) of **Elementary Algebra** (1st edition), Lynn Marecek and MaryAnne Anthony-Smith, OpenStax, Rice University, 2017, https://openstax.org/details/books/elementary-algebra | **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). Pages unchanged; extracted with `scripts/make_algebra_sample.py` (see `algebra_unit/SOURCE.md`). Not affiliated with or endorsed by OpenStax or Rice University. The 1st edition is used because *Elementary Algebra 2e* is CC BY-NC-SA. |
| `examples/sample_curricula/controls_unit/feedback_control_basics.md` | Written for this repository | MIT (part of this repository) |

Uploaded curricula stay on your machine (in `data/`). You are responsible for having the right to use them.

## Tools and models (installed separately)

| Tool | Licence | Use |
|---|---|---|
| Ollama | MIT | local model server on `localhost:11434` |
| `qwen3.5:4b` | Apache-2.0 | default LLM |
| `nomic-embed-text` | Apache-2.0 | embeddings |
| Anthropic API (optional) | commercial terms | only with `--provider anthropic` and your own key |

## Python packages

| Package | Licence |
|---|---|
| rag-assistant (Project 3, github.com/soumics/llm-rag-assistant) | MIT |
| pypdf | BSD-3-Clause |
| numpy | BSD-3-Clause |
| faiss-cpu | MIT |
| requests | Apache-2.0 |
| PyYAML | MIT |
| anthropic | MIT |
| gradio | Apache-2.0 |
| pillow | MIT-CMU (HPND) |
| pytest | MIT |
