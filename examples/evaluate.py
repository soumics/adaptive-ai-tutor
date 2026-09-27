"""Measure the tutor on both sample curricula; writes examples/eval_results.json.

    python examples/evaluate.py                  # local model (Ollama)

1. **Cited answers** (``eval_questions.jsonl``): 10 in-scope questions per subject with the section that
   answers them, and 4 out-of-scope ones (2 from the *other* subject). Reported: retrieval top-1
   section, whether the model's own answer passed the citation check (vs. the quoted-passages
   fallback), whether it cites the right section, and whether out-of-scope questions are declined.
2. **Question bank**: what the checks kept and rejected, and why (from the bank build log).
3. **Diagnostic placement** with simulated learners on the real question banks: every learner knows
   the first k topics in learning order (k = 0..n). Noise-free, and with realistic noise (10 % slips
   on known topics, 25 % lucky guesses on unknown ones), 50 learners per k. Reported: how often the
   recommended starting topic is the first unknown topic, and the number of questions asked.

The learners are simulated: this measures the placement *logic*, not learning outcomes.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagnostic import run_diagnostic  # noqa: E402
from model_provider import make_provider  # noqa: E402
from question_gen import QuestionBank  # noqa: E402
from retrieval import load_or_build  # noqa: E402
from topic_graph import build_topic_graph  # noqa: E402
from tutor_engine import ask  # noqa: E402

SUBJECTS = {"algebra": ROOT / "examples/sample_curricula/algebra_unit",
            "controls": ROOT / "examples/sample_curricula/controls_unit"}


def section_number(index, section_id: str) -> str:
    return index.cur.section(section_id).number


def eval_answers(provider) -> dict:
    items = [json.loads(line) for line in (ROOT / "examples/eval_questions.jsonl").read_text().splitlines() if line]
    out = {}
    for subject, path in SUBJECTS.items():
        index = load_or_build([path], ROOT / "data/indexes" / subject)
        rows = []
        for it in (i for i in items if i["subject"] == subject):
            hits = index.search(it["question"], k=5)
            a = ask(it["question"], index, provider, level="intermediate")
            cited = sorted({section_number(index, c.section_id) for c in a.citations})
            rows.append(dict(question=it["question"], expected=it["section"],
                             top1=section_number(index, hits[0].chunk.section_id) if hits else None,
                             grounded=a.grounded, fallback=a.fallback, abstained=a.abstained, cited_sections=cited,
                             attempts=a.attempts, uncited_sentences=a.uncited_sentences, seconds=round(a.seconds, 2),
                             answer=a.text))
            print(f"[{subject}] {'OUT ' if it['section'] is None else it['section']:4s} "
                  f"g={a.grounded!s:5s} fb={a.fallback!s:5s} ab={a.abstained!s:5s} cited={cited} {it['question'][:60]}")
        ins = [r for r in rows if r["expected"] is not None]
        outs = [r for r in rows if r["expected"] is None]
        out[subject] = dict(
            in_scope=len(ins), out_of_scope=len(outs),
            retrieval_top1_correct=sum(r["top1"] == r["expected"] for r in ins),
            answered_grounded=sum(r["grounded"] for r in ins),
            fallback_to_passages=sum(r["fallback"] for r in ins),
            wrongly_declined=sum(r["abstained"] for r in ins),
            cites_expected_section=sum(r["expected"] in r["cited_sections"] for r in ins),
            answers_with_citation=sum(bool(r["cited_sections"]) for r in rows),
            out_of_scope_declined=sum(r["abstained"] for r in outs),
            median_seconds=round(statistics.median(r["seconds"] for r in rows), 2),
            rows=rows)
    return out


def eval_bank() -> dict:
    out = {}
    for subject in SUBJECTS:
        bank = QuestionBank.load(ROOT / "data/banks" / f"{subject}.json")
        rej = [r for r in bank.log if r[0] == "rejected"]
        out[subject] = dict(
            kept=len(bank.questions),
            kept_mcq=sum(q.kind == "mcq" for q in bank.questions),
            kept_short=sum(q.kind == "short" for q in bank.questions),
            kept_short_exact=sum(bool(q.exact_answer) for q in bank.questions),
            mcq_by_difficulty={d: sum(q.kind == "mcq" and q.difficulty == d for q in bank.questions) for d in (1, 2, 3)},
            rejected=len(rej), rejected_by_reason={k: sum(r[2] == k for r in rej) for k in sorted({r[2] for r in rej})},
            generation_errors=sum(r[0] == "error" for r in bank.log), generator=bank.generator)
    return out


def eval_diagnostic(n_noisy: int = 50) -> dict:
    out = {}
    for subject, path in SUBJECTS.items():
        index = load_or_build([path], ROOT / "data/indexes" / subject)
        graph = build_topic_graph(index.cur)
        bank = QuestionBank.load(ROOT / "data/banks" / f"{subject}.json")
        order = graph.order()
        res = {}
        for noise in (False, True):
            correct, asked = 0, []
            trials = 0
            for k in range(len(order) + 1):
                known = set(order[:k])
                truth = order[k] if k < len(order) else None
                for seed in range(n_noisy if noise else 1):
                    rng = random.Random(f"{subject}:{k}:{seed}")

                    def answer(q, known=known, rng=rng):
                        knows = q.section_id in known
                        if not noise:
                            return knows
                        return (rng.random() > 0.10) if knows else (rng.random() < 0.25)
                    p = run_diagnostic(graph, bank, answer)
                    correct += p.start_topic == truth
                    asked.append(p.questions_asked)
                    trials += 1
            res["noisy" if noise else "noise_free"] = dict(learners=trials, start_topic_correct=correct,
                                                           mean_questions=round(statistics.mean(asked), 2),
                                                           max_questions=max(asked))
        out[subject] = dict(topics=len(order), **res)
    return out


def main() -> None:
    provider = make_provider()
    t0 = time.perf_counter()
    results = dict(model=provider.name, answers=eval_answers(provider), bank=eval_bank(),
                   diagnostic=eval_diagnostic())
    results["seconds"] = round(time.perf_counter() - t0, 1)
    (ROOT / "examples/eval_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    for s in SUBJECTS:
        a, b, d = results["answers"][s], results["bank"][s], results["diagnostic"][s]
        print(f"\n== {s}: answers {a['answered_grounded']}/{a['in_scope']} grounded, "
              f"{a['cites_expected_section']}/{a['in_scope']} cite the right section, "
              f"{a['out_of_scope_declined']}/{a['out_of_scope']} out-of-scope declined, "
              f"{a['answers_with_citation']}/{a['in_scope'] + a['out_of_scope']} with a citation; "
              f"bank {b['kept']} kept / {b['rejected']} rejected; diagnostic noise-free "
              f"{d['noise_free']['start_topic_correct']}/{d['noise_free']['learners']}, noisy "
              f"{d['noisy']['start_topic_correct']}/{d['noisy']['learners']} ({d['noisy']['mean_questions']} questions)")


if __name__ == "__main__":
    main()
