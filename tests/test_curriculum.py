"""Ingestion and the topic graph on two very different subjects and formats (textbook PDF, Markdown notes),
plus the heading fallback for a PDF-less plain-text curriculum."""

from curriculum_ingest import find_heading, ingest_curriculum, split_number
from topic_graph import build_topic_graph


def test_textbook_pdf_sections_come_from_its_bookmarks(algebra_cur):
    assert [s.number for s in algebra_cur.sections] == ["2.1", "2.2", "2.3", "2.4"]
    assert [(s.first_page, s.last_page) for s in algebra_cur.sections] == [(1, 15), (16, 29), (30, 39), (40, 52)]
    for s in algebra_cur.sections:                                   # every chunk stays inside its section's pages
        for c in algebra_cur.chunks_for(s.section_id):
            assert s.first_page <= c.page <= s.last_page
    assert all(len(algebra_cur.chunks_for(s.section_id)) >= 10 for s in algebra_cur.sections)
    # layout extraction keeps word spacing that plain extraction loses on this PDF
    assert "expressions in an equation remain balanced" in " ".join(c.text for c in algebra_cur.chunks).lower() \
        or not any(len(w) > 30 and w.isalpha() for c in algebra_cur.chunks for w in c.text.split())


def test_markdown_notes_sections_and_explicit_prerequisites(controls_cur):
    labels = [s.label for s in controls_cur.sections]
    assert labels == ["1 Open-Loop and Closed-Loop Control", "2 First-Order Systems and the Time Constant",
                      "3 Stability and Poles", "4 PID Control", "5 Consensus in Multi-Agent Systems"]
    pid = controls_cur.sections[3]
    assert pid.explicit_prereqs == ["1. Open-Loop and Closed-Loop Control", "3. Stability and Poles"]
    assert not any("Prerequisites:" in c.text for c in controls_cur.chunks)   # metadata isn't taught as content
    assert all(c.page == 0 and "§" in c.citation() and "p." not in c.citation() for c in controls_cur.chunks)


def test_topic_graphs_for_both_subjects(algebra_cur, controls_cur):
    alg = build_topic_graph(algebra_cur)
    assert [(alg.topic(e.before).number, alg.topic(e.after).number, e.kind) for e in alg.edges] == \
        [("2.1", "2.2", "sequence"), ("2.2", "2.3", "sequence"), ("2.3", "2.4", "sequence")]
    ctl = build_topic_graph(controls_cur)
    pid = ctl.topics[3].section_id
    assert sorted(ctl.topic(p).number for p in ctl.prerequisites(pid)) == ["1", "3"]
    assert {e.kind for e in ctl.edges} == {"explicit"}
    ids = ctl.ids()
    assert ctl.ready_topics({ids[0], ids[1], ids[2]}) == [ids[3], ids[4]]   # PID and consensus unlock together
    assert ctl.ready_topics({ids[0]}) == [ids[1]]


def test_plain_text_uses_numbered_headings_and_references(tmp_path):
    f = tmp_path / "chemistry.txt"
    f.write_text("Intro to the unit.\n1.1 Atoms and Elements\nAn atom is the smallest unit of an element. " * 3 +
                 "\n1.2 Chemical Bonds\nBonds join atoms. See Section 1.1 for atoms. " * 4 +
                 "\n1.3 Reactions\nReactions rearrange bonds between atoms into new substances. " * 4)
    cur = ingest_curriculum([f], name="chem")
    assert [s.number for s in cur.sections] == ["1.1", "1.2", "1.3"]
    g = build_topic_graph(cur)
    kinds = {(g.topic(e.before).number, g.topic(e.after).number): e.kind for e in g.edges}
    assert kinds == {("1.1", "1.2"): "reference", ("1.2", "1.3"): "sequence"}


def test_heading_helpers():
    assert split_number("2.3. Solve Equations with Variables*") == ("2.3", "Solve Equations with Variables")
    assert split_number("PID Control") == ("", "PID Control")
    text = "Chapter intro text.\n2.1\nSolve Equations Using the\nSubtraction Property\nLearning Objectives"
    assert text[find_heading(text, "2.1", "Solve Equations Using the Subtraction Property"):].startswith("2.1")
