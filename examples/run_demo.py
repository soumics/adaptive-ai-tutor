"""The whole tutoring flow on one curriculum, end to end, from the command line.

    python examples/run_demo.py --subject algebra      # examples/sample_curricula/algebra_unit (textbook PDF)
    python examples/run_demo.py --subject controls     # examples/sample_curricula/controls_unit (Markdown notes)
    python examples/run_demo.py --curriculum path/to/your/files --name "My course"

Steps: ingest the material -> topic graph -> question bank (written and checked by the local LLM,
cached in data/) -> adaptive diagnostic -> cited tutor answers -> a Socratic exchange -> a practice
quiz (multiple choice + free text) -> spaced-repetition reviews over the following days -> progress
dashboard. The learner in the diagnostic, quiz and reviews is **simulated** (a fixed profile of
what it knows, and scripted free-text answers), so the run is repeatable; the tutor's side is real.
A report is written to output/<subject>/report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagnostic import Diagnostic  # noqa: E402
from grading import grade  # noqa: E402
from model_provider import make_provider  # noqa: E402
from progress_tracker import ProgressTracker  # noqa: E402
from question_gen import QuestionBank, build_bank  # noqa: E402
from retrieval import load_or_build  # noqa: E402
from socratic import SocraticSession  # noqa: E402
from topic_graph import build_topic_graph  # noqa: E402
from tutor_engine import ask  # noqa: E402

SAMPLES = ROOT / "examples" / "sample_curricula"
SCENARIOS = {
    "algebra": dict(
        path=SAMPLES / "algebra_unit",
        knows=2,                                   # the simulated learner knows the first two topics
        questions=["How do I check whether a number is a solution of an equation?",
                   "When an equation has variables on both sides, which side should I put the variable on?"],
        socratic=("How do I solve x + 7 = 12?", ["Subtract 7?", "x = 5, and 5 + 7 = 12 checks out."]),
        free_text={"exact": "x = {answer}"}),
    "controls": dict(
        path=SAMPLES / "controls_unit",
        knows=2,
        questions=["Why does a P controller leave a steady-state error, and what removes it?",
                   "What does the second-smallest eigenvalue of the Laplacian tell us?"],
        socratic=("What does the time constant of a first-order system tell me?",
                  ["How fast it responds?", "At t = tau the output is at about 63 percent of the final value."]),
        free_text={"exact": "{answer}"}),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", choices=sorted(SCENARIOS), default="controls")
    ap.add_argument("--curriculum", nargs="+", help="your own curriculum files or folders (instead of --subject)")
    ap.add_argument("--name", help="curriculum name (with --curriculum)")
    ap.add_argument("--provider", default="ollama", choices=["ollama", "anthropic"])
    ap.add_argument("--model")
    ap.add_argument("--rebuild-bank", action="store_true")
    args = ap.parse_args()

    key = args.subject if not args.curriculum else (args.name or Path(args.curriculum[0]).stem).lower().replace(" ", "-")
    sc = SCENARIOS.get(args.subject) if not args.curriculum else dict(path=None, knows=1, questions=[], socratic=None,
                                                                         free_text={"exact": "{answer}"})
    paths = args.curriculum or [sc["path"]]
    out = ROOT / "output" / key
    out.mkdir(parents=True, exist_ok=True)
    report: dict = dict(subject=key, provider=None, timings_s={})
    t = time.perf_counter

    def stage(name, t0):
        report["timings_s"][name] = round(t() - t0, 2)

    provider = make_provider(args.provider, args.model)
    report["provider"] = provider.name

    t0 = t()
    index = load_or_build(paths, ROOT / "data" / "indexes" / key, name=args.name)
    graph = build_topic_graph(index.cur)
    stage("ingest_and_index", t0)
    print(f"\n# {index.cur.name}: {len(index.cur.sections)} topics, {len(index.cur.chunks)} chunks")
    for topic, requires, how in graph.as_rows():
        print(f"  {topic}  <- {requires} ({how})")
    report["topics"] = [dict(topic=a, requires=b, found_by=c) for a, b, c in graph.as_rows()]

    t0 = t()
    bank_path = ROOT / "data" / "banks" / f"{key}.json"
    if bank_path.exists() and not args.rebuild_bank:
        bank = QuestionBank.load(bank_path)
    else:
        bank = build_bank(index, provider, progress=lambda f, m: print(f"  [{f:.0%}] {m}"))
        bank.save(bank_path)
    stage("question_bank", t0)
    kinds = {k: sum(q.kind == k for q in bank.questions) for k in ("mcq", "short")}
    rejected = [r for r in bank.log if r[0] == "rejected"]
    print(f"\n## Question bank: {len(bank.questions)} kept ({kinds}), {len(rejected)} candidates rejected")
    report["bank"] = dict(kept=len(bank.questions), by_kind=kinds, rejected=len(rejected),
                          rejected_reasons={r: sum(x[2] == r for x in rejected) for r in {x[2] for x in rejected}},
                          errors=sum(r[0] == "error" for r in bank.log))

    # ---- diagnostic with a simulated learner who knows the first `knows` topics (all difficulties)
    order = graph.order()
    known = set(order[: sc["knows"]])
    t0 = t()
    diag = Diagnostic(graph, bank)
    tracker = ProgressTracker(out / "progress.sqlite")
    learner, day0 = "demo-learner", datetime(2026, 1, 5, 9, 0)
    tracker.db.execute("DELETE FROM attempts WHERE learner=?", (learner,))
    tracker.db.execute("DELETE FROM cards WHERE learner=?", (learner,))
    while (q := diag.next_question()) is not None:
        choice = q.answer_index if q.section_id in known else (q.answer_index + 1) % 4
        g = grade(q, choice)
        diag.answer(g.correct)
        tracker.record(learner, key, q, g, "diagnostic", day0)
    placement = diag.placement()
    tracker.set_placement(learner, key, {s: r.status for s, r in placement.topics.items()}, day0)
    stage("diagnostic", t0)
    start = placement.start_topic
    print(f"\n## Diagnostic: {placement.questions_asked} questions, overall {placement.overall}, "
          f"start at: {graph.topic(start).label if start else '(all mastered)'}")
    for sid in order:
        r = placement.topics[sid]
        print(f"  {graph.topic(sid).label[:60]:60s} {r.status:10s} {[('✓' if ok else '✗') + str(d) for _, d, ok in r.answers]}")
    report["diagnostic"] = dict(questions=placement.questions_asked, overall=placement.overall,
                                start_topic=graph.topic(start).label if start else None,
                                simulated_learner_knows=[graph.topic(s).label for s in known],
                                topics={graph.topic(s).label: placement.topics[s].status for s in order})

    # ---- cited tutor answers, at the level the diagnostic suggests
    t0 = t()
    report["tutor"] = []
    for question in sc["questions"]:
        a = ask(question, index, provider, level=placement.level_for(start) if start else "advanced")
        print(f"\n## Tutor: {question}\n{a.formatted()}")
        report["tutor"].append(dict(question=question, answer=a.text, citations=[c.citation for c in a.citations],
                                    grounded=a.grounded, fallback=a.fallback, abstained=a.abstained,
                                    attempts=a.attempts, seconds=round(a.seconds, 2)))
    stage("tutor", t0)

    # ---- a Socratic exchange with scripted learner replies
    if sc["socratic"]:
        t0 = t()
        q_text, replies = sc["socratic"]
        s = SocraticSession(q_text, index, provider, level="beginner")
        turns = [("tutor", s.start())]
        for r in replies:
            if s.done:
                break
            turns += [("learner", r), ("tutor", s.reply(r))]
        print(f"\n## Socratic: {q_text}")
        log = []
        for who, x in turns:
            text = x if who == "learner" else x.text
            print(f"  {who.upper()}: {text}")
            log.append(dict(who=who, text=text, cites=[] if who == "learner" else [c.citation for c in x.citations],
                            cite_added=False if who == "learner" else x.cite_added))
        report["socratic"] = dict(question=q_text, turns=log, ended=s.done)
        stage("socratic", t0)

    # ---- practice quiz on the starting topic: 3 multiple choice + 2 free-text, simulated answers
    t0 = t()
    quiz_topic = start or order[-1]
    items = bank.for_topic(quiz_topic, "mcq")[:3] + bank.for_topic(quiz_topic, "short")[:2]
    report["quiz"] = []
    print(f"\n## Practice quiz: {graph.topic(quiz_topic).label}")
    for i, q in enumerate(items):
        if q.kind == "mcq":
            response = q.answer_index if i % 2 == 0 else (q.answer_index + 1) % 4     # alternate right / wrong
        elif q.exact_answer:
            response = sc["free_text"]["exact"].format(answer=q.exact_answer)
        elif not any(r["kind"] == "short" for r in report["quiz"]):
            response = q.model_answer                   # first free-text answer: a correct one
        else:
            response = "I think it has something to do with the size of the numbers."   # then a wrong one
        g = grade(q, response, provider)
        tracker.record(learner, key, q, g, "practice", day0 + timedelta(hours=1))
        print(f"  [{q.kind}] {q.prompt}\n     answer: {response if q.kind != 'mcq' else 'ABCD'[response]}  "
              f"-> score {g.score:.2f}, SM-2 quality {g.quality}")
        report["quiz"].append(dict(kind=q.kind, question=q.prompt, response=str(response), score=g.score,
                                   quality=g.quality, feedback=g.feedback,
                                   points=[dict(point=p.point, met=p.met, method=p.method) for p in g.points]))
    stage("quiz", t0)

    # ---- spaced repetition: review whatever is due, day by day, for two weeks (answers: right if known topic,
    # right on the second try otherwise -- the learner is learning)
    seen: dict[str, int] = {}
    reviews = []
    for day in range(1, 15):
        today = (day0 + timedelta(days=day)).date()
        for qid in tracker.due_items(learner, key, today):
            q = bank.get(qid)
            seen[qid] = seen.get(qid, 0) + 1
            ok = q.section_id in known or seen[qid] >= 2
            g = grade(q, q.answer_index if (q.kind == "mcq" and ok) else ((q.answer_index + 1) % 4 if q.kind == "mcq"
                                                                          else (q.model_answer if ok else "")))
            card, _ = tracker.record(learner, key, q, g, "review", datetime.combine(today, datetime.min.time()))
            reviews.append(dict(day=day, qid=qid, correct=g.correct, next_interval_days=card.interval))
    print(f"\n## Reviews over 14 days: {len(reviews)} reviews; per day: "
          f"{ {d: sum(r['day'] == d for r in reviews) for d in range(1, 15) if any(r['day'] == d for r in reviews)} }")
    report["reviews"] = reviews

    # ---- dashboard
    rows = tracker.dashboard(learner, key, [(s, graph.topic(s).label) for s in order], date(2026, 1, 19))
    print("\n## Progress dashboard (after two weeks)")
    for r in rows:
        print(f"  {r.label[:55]:55s} mastery {r.mastery:.2f}  {r.status:12s} attempts {r.attempts:2d}  "
              f"placement {r.placement:10s} next review {r.next_review}")
    report["dashboard"] = [dict(topic=r.label, mastery=r.mastery, status=r.status, attempts=r.attempts,
                                placement=r.placement, due=r.due_reviews, next_review=str(r.next_review)) for r in rows]
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\ntimings (s): {report['timings_s']}\nreport: {out / 'report.json'}")


if __name__ == "__main__":
    main()
