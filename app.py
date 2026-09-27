"""Adaptive AI Tutor: the web interface (local, in your browser).

    python app.py                        # then open http://127.0.0.1:7860
    python app.py --provider anthropic   # optional API model (needs ANTHROPIC_API_KEY in the environment)

Flow: choose a sample curriculum or upload your own (PDF / Markdown / text) -> the tutor finds its
topics and prerequisites and writes checked practice questions -> diagnostic -> tutoring (cited
explanations, or Socratic mode) -> practice quiz -> spaced-repetition review -> progress dashboard.
The server listens on 127.0.0.1 only and Gradio's analytics are switched off.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"      # before gradio is imported

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from diagnostic import Diagnostic  # noqa: E402
from grading import grade  # noqa: E402
from model_provider import make_provider  # noqa: E402
from progress_tracker import ProgressTracker  # noqa: E402
from question_gen import QuestionBank, build_bank  # noqa: E402
from retrieval import load_or_build  # noqa: E402
from socratic import SocraticSession  # noqa: E402
from topic_graph import build_topic_graph  # noqa: E402
from tutor_engine import ask  # noqa: E402

DATA = ROOT / "data"
SAMPLES = {"Algebra: linear equations (OpenStax textbook PDF)": ROOT / "examples/sample_curricula/algebra_unit",
           "Control systems: feedback basics (Markdown notes)": ROOT / "examples/sample_curricula/controls_unit"}
LETTERS = "ABCD"
# inline $...$ maths as well as $$...$$ (the models write equations in LaTeX)
LATEX = [{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}]
ALL_TOPICS = "All topics"
PROVIDER = None
TRACKER: ProgressTracker | None = None
_COURSES: dict[str, dict] = {}                      # key -> {index, graph, bank}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "course"


def load_course(key: str, paths: list[Path], name: str | None = None) -> dict:
    if key not in _COURSES:
        index = load_or_build(paths, DATA / "indexes" / key, name=name)
        bank_path = DATA / "banks" / f"{key}.json"
        bank = QuestionBank.load(bank_path) if bank_path.exists() else None
        _COURSES[key] = dict(index=index, graph=build_topic_graph(index.cur), bank=bank, bank_path=bank_path)
    return _COURSES[key]


def course_of(state) -> dict:
    if not state.get("course"):
        raise gr.Error("Load a curriculum first (tab 1).")
    return _COURSES[state["course"]]


def today(state) -> date:
    return date.today() + timedelta(days=state.get("day_offset", 0))


def topic_choices(c) -> list[str]:
    return [c["index"].cur.section(s).label for s in c["graph"].order()]


def sid_of(c, label: str) -> str | None:
    return next((s.section_id for s in c["index"].cur.sections if s.label == label), None)


def sources_md(citations) -> str:
    return "\n".join(f"- **[{c.label}]** {c.citation}" for c in citations)


# ------------------------------------------------------------------ 1. curriculum


def load_curriculum(choice, files, name, learner, state, progress=gr.Progress()):
    state = dict(state or {})
    state["learner"] = (learner or "learner").strip() or "learner"
    if files:
        name = (name or "").strip() or Path(files[0]).stem.replace("_", " ")
        key = slug(name)
        folder = DATA / "uploads" / key
        folder.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, folder / Path(f).name)
        paths = [folder]
    else:
        if choice not in SAMPLES:
            raise gr.Error("Choose a sample curriculum or upload your own files.")
        # the samples use the same index/bank names as examples/run_demo.py and examples/evaluate.py
        key, paths, name = SAMPLES[choice].name.replace("_unit", ""), [SAMPLES[choice]], None
    progress(0.1, desc="Reading the material and building the index")
    try:
        c = load_course(key, paths, name)
    except ValueError as e:
        raise gr.Error(str(e)) from None
    state.update(course=key, diag=None, quiz=[], current=None, socratic=None)
    if c["bank"] is None:
        progress(0.3, desc="Writing and checking practice questions (a few minutes, once per curriculum)")
        c["bank"] = build_bank(c["index"], PROVIDER, progress=lambda f, m: progress(0.3 + 0.7 * f, desc=m))
        c["bank"].save(c["bank_path"])
    cur, g, bank = c["index"].cur, c["graph"], c["bank"]
    kinds = {k: sum(q.kind == k for q in bank.questions) for k in ("mcq", "short")}
    summary = (f"### {cur.name}\n{len(cur.sections)} topics · {len(cur.chunks)} passages · "
               f"{len(bank.questions)} checked practice questions ({kinds['mcq']} multiple choice, "
               f"{kinds['short']} short answer)")
    rows = [list(r) for r in g.as_rows()]
    topics = topic_choices(c)
    table, plot = dashboard(state)
    return (state, summary, rows, gr.update(choices=[ALL_TOPICS] + topics, value=ALL_TOPICS),
            gr.update(choices=topics, value=topics[0]), table, plot, [])   # [] clears the previous course's chat


# ------------------------------------------------------------------ 2. diagnostic


def question_view(q, title: str):
    if q.kind == "mcq":
        return (f"**{title}**\n\n{q.prompt}", gr.update(choices=[f"{LETTERS[i]}) {o}" for i, o in enumerate(q.options)],
                                                         value=None, visible=True), gr.update(value="", visible=False))
    return (f"**{title}**\n\n{q.prompt}", gr.update(visible=False, value=None), gr.update(value="", visible=True))


def diag_start(state):
    c = course_of(state)
    state = dict(state)
    state["diag"] = Diagnostic(c["graph"], c["bank"])
    q = state["diag"].next_question()
    if q is None:
        raise gr.Error("The question bank has no multiple-choice questions for this curriculum.")
    return (state, *question_view(q, f"Diagnostic · {c['index'].cur.section(q.section_id).label}"), "")


def diag_answer(choice, state):
    d = state.get("diag")
    if d is None or d.current is None:
        raise gr.Error("Start the diagnostic first.")
    if not choice:
        raise gr.Error("Choose an answer.")
    c = course_of(state)
    q = d.current
    g = grade(q, LETTERS.index(choice[0]))
    TRACKER.record(state["learner"], state["course"], q, g, "diagnostic")
    d.answer(g.correct)
    nxt = d.next_question()
    if nxt is not None:
        return (state, *question_view(nxt, f"Diagnostic · {c['index'].cur.section(nxt.section_id).label}"), "")
    p = d.placement()
    TRACKER.set_placement(state["learner"], state["course"], {s: r.status for s, r in p.topics.items()})
    state["placement"] = p
    g = c["graph"]
    lines = [f"### Placement: {p.overall} ({p.questions_asked} questions)", "",
             "| Topic | Result | Answers |", "|---|---|---|"]
    for sid in g.order():
        r = p.topics[sid]
        marks = " ".join(("✓" if ok else "✗") + f"d{dd}" for _, dd, ok in r.answers)
        lines.append(f"| {g.topic(sid).label} | {r.status}{' (prerequisite gap)' if r.probe else ''} | {marks} |")
    start = g.topic(p.start_topic).label if p.start_topic else None
    lines += ["", f"**Start with:** {start}" if start else "**You have mastered every topic in this unit.**"]
    return (state, "\n".join(lines), gr.update(visible=False), gr.update(visible=False), "")


# ------------------------------------------------------------------ 3. tutor


def tutor_send(message, history, topic, level, mode, state):
    c = course_of(state)
    history = list(history or [])
    if not message or not message.strip():
        return history, "", state
    state = dict(state)
    sid = None if topic in (None, ALL_TOPICS) else sid_of(c, topic)
    topics = [sid] if sid else None                  # a topic from another course counts as "all topics"
    if level == "Auto (from the diagnostic)":
        p = state.get("placement")
        sid = topics[0] if topics else (p.start_topic if p else None)
        level = p.level_for(sid) if (p and sid) else "intermediate"
    history.append({"role": "user", "content": message})
    s = state.get("socratic")
    if mode == "Socratic (I'll ask you questions)":
        if s is None or s.done:
            s = SocraticSession(message, c["index"], PROVIDER, topics=topics, level=level)
            turn = s.start()
        else:
            turn = s.reply(message)
        state["socratic"] = None if turn.done else s
        reply = turn.text + "\n\n" + sources_md(turn.citations)
        if turn.done:
            reply += "\n\n*(Socratic session finished: ask a new question to start another.)*"
    else:
        a = ask(message, c["index"], PROVIDER, level=level, topics=topics)
        note = " *(shown as quoted passages: the model's answer couldn't be verified)*" if a.fallback else ""
        reply = f"{a.text}{note}\n\n{sources_md(a.citations)}"
    history.append({"role": "assistant", "content": reply})
    return history, "", state


# ------------------------------------------------------------------ 4. practice


def quiz_start(topic, state):
    c = course_of(state)
    sid = sid_of(c, topic)
    qs = c["bank"].for_topic(sid, "mcq")
    done = {r[0] for r in TRACKER.db.execute("SELECT qid FROM attempts WHERE learner=? AND curriculum=?",
                                            (state["learner"], state["course"])).fetchall()}
    fresh = [q for q in qs if q.qid not in done] + [q for q in qs if q.qid in done]
    quiz = sorted(fresh[:4], key=lambda q: q.difficulty) + c["bank"].for_topic(sid, "short")[:1]
    if not quiz:
        raise gr.Error("No practice questions for this topic yet.")
    state = dict(state, quiz=[q.qid for q in quiz[1:]], current=quiz[0].qid, quiz_mode="practice")
    return (state, *question_view(quiz[0], f"Practice · question 1 of {len(quiz)}"), "")


def submit(choice, text, state):
    c = course_of(state)
    if not state.get("current"):
        raise gr.Error("Start a quiz or a review first.")
    q = c["bank"].get(state["current"])
    if q.kind == "mcq":
        if not choice:
            raise gr.Error("Choose an answer.")
        response = LETTERS.index(choice[0])
    else:
        if not (text or "").strip():
            raise gr.Error("Type your answer.")
        response = text
    g = grade(q, response, PROVIDER)
    # with the demo's "next day" button, answers are recorded on the pretend date
    now = datetime.combine(today(state), time(12)) if state.get("day_offset") else None
    card, again = TRACKER.record(state["learner"], state["course"], q, g, state.get("quiz_mode", "practice"), now)
    rubric = ""
    if q.kind == "short" and q.rubric and not q.exact_answer:
        rubric = "\n\n**Rubric used:**\n" + "\n".join(f"- {p.point} *(looks for: {', '.join(p.keywords)})*"
                                                     for p in q.rubric)
    when = "later today" if again else f"in {card.interval} day{'s' if card.interval != 1 else ''}"
    feedback = f"{g.feedback}{rubric}\n\n*Next review of this question: {when}.*"
    state = dict(state)
    queue = list(state.get("quiz", []))
    if queue:
        nxt = c["bank"].get(queue.pop(0))
        state.update(quiz=queue, current=nxt.qid)
        return (state, *question_view(nxt, "Next question"), feedback)
    state.update(current=None)
    return (state, "**Done.** See your progress in the last tab.", gr.update(visible=False),
            gr.update(visible=False), feedback)


# ------------------------------------------------------------------ 5. review


def review_start(state):
    c = course_of(state)
    due = TRACKER.due_items(state["learner"], state["course"], today(state))
    if not due:
        nxt = TRACKER.db.execute("SELECT MIN(due) FROM cards WHERE learner=? AND curriculum=?",
                                 (state["learner"], state["course"])).fetchone()[0]
        msg = f"Nothing due on {today(state)}." + (f" Next review: {nxt}." if nxt else " Answer some questions first.")
        return state, msg, gr.update(visible=False), gr.update(visible=False), ""
    q = c["bank"].get(due[0])
    state = dict(state, quiz=due[1:], current=due[0], quiz_mode="review")
    return (state, *question_view(q, f"Review · {len(due)} due on {today(state)}"), "")


def next_day(state):
    state = dict(state or {})
    state["day_offset"] = state.get("day_offset", 0) + 1
    return state, f"Pretending it is **{today(state)}** (demo: see how the review schedule plays out)."


# ------------------------------------------------------------------ 6. progress


def dashboard(state):
    import pandas as pd

    if not state or not state.get("course"):
        return [], pd.DataFrame({"topic": [], "mastery": []})
    c = _COURSES[state["course"]]
    g = c["graph"]
    rows = TRACKER.dashboard(state["learner"], state["course"], [(s, g.topic(s).label) for s in g.order()],
                             today(state))
    table = [[r.label, f"{r.mastery:.2f}", r.status, r.attempts, r.placement, r.due_reviews,
              str(r.next_review or "-")] for r in rows]
    df = pd.DataFrame({"topic": [r.label[:40] for r in rows], "mastery": [r.mastery for r in rows]})
    return table, df


# ------------------------------------------------------------------ layout


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Adaptive AI Tutor") as demo:
        state = gr.State({})
        gr.Markdown("# Adaptive AI Tutor\nLearns your curriculum, finds out what you know, and teaches from your "
                    "material: every explanation cites the page or section it comes from.")
        with gr.Row():
            learner = gr.Textbox(label="Your name", value="learner", scale=1)
            model_info = gr.Markdown(f"Model: `{PROVIDER.name}`" + (" (local, private)" if PROVIDER.is_local else
                                                                    " (API)"))
        with gr.Tabs():
            with gr.Tab("1 · Curriculum"):
                with gr.Row():
                    choice = gr.Radio(list(SAMPLES), label="Sample curriculum", value=list(SAMPLES)[0])
                    with gr.Column():
                        files = gr.File(label="...or upload your own (PDF, Markdown, text)", file_count="multiple",
                                        file_types=[".pdf", ".md", ".txt"], type="filepath")
                        cname = gr.Textbox(label="Course name (for uploads)")
                load = gr.Button("Load curriculum", variant="primary")
                summary = gr.Markdown()
                topics_table = gr.Dataframe(headers=["Topic", "Requires", "Found from"], interactive=False, wrap=True,
                                            column_widths=["42%", "42%", "16%"], label="Topics and prerequisites")
            with gr.Tab("2 · Diagnostic"):
                gr.Markdown("A short adaptive test (about 2 questions per topic) that finds where to start.")
                d_start = gr.Button("Start the diagnostic", variant="primary")
                d_q = gr.Markdown(latex_delimiters=LATEX)
                d_choice = gr.Radio([], label="Your answer", visible=False)
                d_text = gr.Textbox(visible=False)
                d_submit = gr.Button("Submit")
                d_fb = gr.Markdown(latex_delimiters=LATEX)
            with gr.Tab("3 · Tutor"):
                with gr.Row():
                    t_topic = gr.Dropdown([ALL_TOPICS], value=ALL_TOPICS, label="Topic")
                    t_level = gr.Dropdown(["Auto (from the diagnostic)", "beginner", "intermediate", "advanced"],
                                          value="Auto (from the diagnostic)", label="Explanation level")
                    t_mode = gr.Radio(["Explain", "Socratic (I'll ask you questions)"], value="Explain", label="Mode")
                chat = gr.Chatbot(height=460, label="Tutor", latex_delimiters=LATEX)
                t_msg = gr.Textbox(label="Ask a question about the material", placeholder="e.g. Why do we do the "
                                   "same thing to both sides of an equation?")
            with gr.Tab("4 · Practice"):
                p_topic = gr.Dropdown([], label="Topic")
                p_start = gr.Button("Start a practice quiz", variant="primary")
                p_q = gr.Markdown(latex_delimiters=LATEX)
                p_choice = gr.Radio([], label="Your answer", visible=False)
                p_text = gr.Textbox(label="Your answer", lines=3, visible=False)
                p_submit = gr.Button("Submit")
                p_fb = gr.Markdown(latex_delimiters=LATEX)
            with gr.Tab("5 · Review"):
                gr.Markdown("Questions come back on the SM-2 spaced-repetition schedule: soon after a mistake, "
                            "then at growing intervals as you keep getting them right.")
                with gr.Row():
                    r_start = gr.Button("Review what's due", variant="primary")
                    r_day = gr.Button("Pretend it's the next day (demo)")
                r_info = gr.Markdown()
                r_q = gr.Markdown(latex_delimiters=LATEX)
                r_choice = gr.Radio([], label="Your answer", visible=False)
                r_text = gr.Textbox(label="Your answer", lines=3, visible=False)
                r_submit = gr.Button("Submit")
                r_fb = gr.Markdown(latex_delimiters=LATEX)
            with gr.Tab("6 · Progress"):
                refresh = gr.Button("Refresh")
                table = gr.Dataframe(headers=["Topic", "Mastery", "Status", "Attempts", "Diagnostic", "Due now",
                                              "Next review"], interactive=False, wrap=True,
                                     column_widths=["40%", "9%", "12%", "9%", "11%", "8%", "11%"])
                plot = gr.BarPlot(x="topic", y="mastery", y_lim=[0, 1], label="Mastery by topic", height=320)

        load.click(load_curriculum, [choice, files, cname, learner, state],
                   [state, summary, topics_table, t_topic, p_topic, table, plot, chat])
        d_start.click(diag_start, state, [state, d_q, d_choice, d_text, d_fb])
        d_submit.click(diag_answer, [d_choice, state], [state, d_q, d_choice, d_text, d_fb])
        t_msg.submit(tutor_send, [t_msg, chat, t_topic, t_level, t_mode, state], [chat, t_msg, state])
        p_start.click(quiz_start, [p_topic, state], [state, p_q, p_choice, p_text, p_fb])
        p_submit.click(submit, [p_choice, p_text, state], [state, p_q, p_choice, p_text, p_fb])
        r_start.click(review_start, state, [state, r_q, r_choice, r_text, r_fb])
        r_submit.click(submit, [r_choice, r_text, state], [state, r_q, r_choice, r_text, r_fb])
        r_day.click(next_day, state, [state, r_info])
        refresh.click(dashboard, state, [table, plot])
        learner.change(lambda n, s: dict(s or {}, learner=(n or "learner").strip() or "learner"), [learner, state],
                       state)
    return demo


def main() -> None:
    global PROVIDER, TRACKER
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="ollama", choices=["ollama", "anthropic"])
    ap.add_argument("--model", help="model name (default: qwen3.5:4b for Ollama)")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1", help="listen address (default: this computer only; in Docker "
                                                        "with a published port use 0.0.0.0)")
    args = ap.parse_args()
    PROVIDER = make_provider(args.provider, args.model, args.ollama_url)
    TRACKER = ProgressTracker(DATA / "progress.sqlite")
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)       # one LLM request at a time on a single local GPU
    demo.launch(server_name=args.host, server_port=args.port, share=False,
                theme=gr.themes.Soft(font=["ui-sans-serif", "system-ui", "sans-serif"],
                                     font_mono=["ui-monospace", "monospace"]))


if __name__ == "__main__":
    main()
