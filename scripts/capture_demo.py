"""Record the demo: drive the running web app in headless Firefox and save a screenshot per step.

    python app.py --port 7861 &                       # the app, with both sample curricula
    python scripts/capture_demo.py --out media/frames  # needs Firefox + geckodriver (no Python packages)

The simulated learner answers diagnostic and quiz questions from the question bank's keys: right on the
topics it "knows", wrong on the others. Everything the tutor says is produced live by the app.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Browser:
    def __init__(self, driver: str, port: int = 4466, width: int = 1280):
        self.proc = subprocess.Popen([driver, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.base, self.width = f"http://127.0.0.1:{port}", width
        for _ in range(50):
            try:
                urllib.request.urlopen(self.base + "/status")
                break
            except OSError:
                time.sleep(0.3)
        caps = {"capabilities": {"alwaysMatch": {"moz:firefoxOptions": {"args": ["-headless"]}}}}
        self.session = self._req("POST", "/session", caps)["sessionId"]

    def _req(self, method, path, body=None):
        r = urllib.request.Request(self.base + path, method=method, headers={"Content-Type": "application/json"},
                                   data=json.dumps(body).encode() if body is not None else None)
        return json.load(urllib.request.urlopen(r, timeout=600))["value"]

    def s(self, method, path, body=None):
        return self._req(method, f"/session/{self.session}{path}", body)

    def js(self, code, *args):
        return self.s("POST", "/execute/sync", {"script": code, "args": list(args)})

    def goto(self, url):
        self.s("POST", "/url", {"url": url})

    def size(self, h):
        self.s("POST", "/window/rect", {"width": self.width, "height": h})

    def shot(self, path: Path, h: int = 900):
        self.size(h)
        time.sleep(0.8)
        path.write_bytes(base64.b64decode(self.s("GET", "/screenshot")))
        print("  saved", path.name)

    def type_into(self, label: str, text: str, enter: bool = False) -> None:
        """Real keyboard input (WebDriver), into the visible textbox with this label."""
        el = self.js("""const l=[...document.querySelectorAll('label, span')].find(x=>x.innerText.trim()===arguments[0]
                        && x.offsetParent!==null); let box=l;
                        while(box && !(box.querySelector && box.querySelector('textarea, input[type=text]')))
                            box=box.parentElement;
                        const el=box && box.querySelector('textarea, input[type=text]'); if(el){el.value='';} return el""",
                     label)
        if not el:
            raise RuntimeError(f"no textbox labelled {label!r}")
        eid = next(iter(el.values()))
        self.s("POST", f"/element/{eid}/clear", {})
        self.s("POST", f"/element/{eid}/value", {"text": text + ("\ue007" if enter else "")})

    def choose(self, label: str, option_prefix: str) -> None:
        """Open a dropdown with a real click and click the option that starts with ``option_prefix``."""
        find = """const l=[...document.querySelectorAll('label, span')].find(x=>x.innerText.trim()===arguments[0] &&
                  x.offsetParent!==null); let box=l; while(box && !(box.querySelector && box.querySelector('input')))
                  box=box.parentElement; return box && box.querySelector('input')"""
        el = self.js(find, label)
        self.s("POST", f"/element/{next(iter(el.values()))}/click", {})
        time.sleep(1)
        opt = self.js("return [...document.querySelectorAll('[role=option]')].find(x=>x.offsetParent!==null && "
                      "x.innerText.replace(/^[\\s✓]+/, '').startsWith(arguments[0]))", option_prefix)
        if not opt:
            raise RuntimeError(f"no option {option_prefix!r} in {label!r}")
        self.s("POST", f"/element/{next(iter(opt.values()))}/click", {})
        time.sleep(1)

    def wait(self, code, timeout=300, *args):
        t0 = time.time()
        while time.time() - t0 < timeout:
            v = self.js(code, *args)
            if v:
                return v
            time.sleep(0.5)
        raise TimeoutError(code[:80])

    def close(self):
        try:
            self.s("DELETE", "")
        finally:
            self.proc.terminate()


CLICK = ("const b=[...document.querySelectorAll('button')].find(x=>x.innerText.trim()===arguments[0] && "
         "x.offsetParent!==null); if(b){b.click(); return true} return false")
TAB = ("const b=[...document.querySelectorAll('[role=tab]')].find(x=>x.innerText.trim()===arguments[0]); "
       "if(b){b.click(); return true} return false")
LABEL = ("const r=[...document.querySelectorAll('label')].find(x=>x.offsetParent!==null && "
         "x.innerText.trim().startsWith(arguments[0])); if(r){r.click(); return true} return false")
FILL = """const l=[...document.querySelectorAll('label, span')].find(x=>x.innerText.trim()===arguments[0] && x.offsetParent!==null);
let box=l; while(box && !(box.querySelector && box.querySelector('textarea, input[type=text]'))) box=box.parentElement;
const el=box && box.querySelector('textarea, input[type=text]'); if(!el) return false;
const proto = el.tagName==='TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
Object.getOwnPropertyDescriptor(proto,'value').set.call(el, arguments[1]);
el.dispatchEvent(new Event('input',{bubbles:true}));
if (arguments[2]) { for (const t of ['keydown','keypress']) el.dispatchEvent(new KeyboardEvent(t,{key:'Enter',code:'Enter',keyCode:13,which:13,bubbles:true})); }
return true"""
VISIBLE_TEXT = "return [...document.querySelectorAll('.prose, .md')].filter(x=>x.offsetParent!==null).map(x=>x.innerText).join('\\n---\\n')"
BUSY = "return !!document.querySelector('.progress-text, .eta-bar, .generating, .pending')"


def idle(b: Browser, timeout=600):
    time.sleep(1.0)
    b.wait("return !document.querySelector('.progress-text, .eta-bar, .generating, .pending')", timeout)
    time.sleep(0.5)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def shown(prompt: str, page_text: str) -> float:
    """Share of the prompt's words (4+ letters, outside $...$ maths) that appear on the page."""
    words = re.findall(r"[a-z]{4,}", re.sub(r"\$[^$]*\$", " ", prompt.lower()))
    page = set(re.findall(r"[a-z]{4,}", page_text.lower()))
    return sum(w in page for w in words) / max(1, len(words))


def answer_mcq(b: Browser, bank: dict, knows: set[str]) -> str:
    """Find the displayed question in the bank; click the right option if its topic is known, else a wrong one."""
    text = b.js(VISIBLE_TEXT)
    # the page shows rendered Markdown and maths, so match on the question's ordinary words
    q = max((q for q in bank["questions"] if q["kind"] == "mcq"), key=lambda q: shown(q["prompt"], text))
    if shown(q["prompt"], text) < 0.8:
        q = None
    if q is None:
        raise RuntimeError("question not found in the bank")
    i = q["answer_index"] if q["section_id"] in knows else (q["answer_index"] + 1) % 4
    assert b.js(LABEL, f"{'ABCD'[i]}) "), "option not clickable"
    return q["section_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7861/")
    ap.add_argument("--out", default=str(ROOT / "media" / "frames"))
    ap.add_argument("--driver", default=shutil.which("geckodriver") or "/snap/bin/firefox.geckodriver")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    b = Browser(args.driver)
    n = [0]

    def shot(name, h=900):
        n[0] += 1
        b.shot(out / f"{n[0]:02d}_{name}.png", h)

    try:
        b.size(900)
        b.goto(args.url)
        time.sleep(5)
        for subject, sample, knows_n, tutor_q, socratic in [
            ("algebra", "Algebra", 2, "When an equation has variables on both sides, which side should I put the "
                                      "variable on?", ("How do I solve x + 7 = 12?", ["Subtract 7?",
                                                                                     "x = 5, and 5 + 7 = 12 checks out."])),
            ("controls", "Control systems", 2, "Why does a P controller leave a steady-state error, and what "
                                               "removes it?", None)]:
            bank = json.loads((ROOT / "data" / "banks" / f"{subject}.json").read_text())
            index = json.loads((ROOT / "data" / "indexes" / subject / "curriculum.json").read_text())
            order = [s["section_id"] for s in index["sections"]]
            knows = set(order[:knows_n])
            print(f"== {subject}")
            b.js(TAB, "1 · Curriculum")
            b.js(LABEL, sample)
            b.js(CLICK, "Load curriculum")
            b.wait("return document.body.innerText.includes('checked practice questions') && "
                   "document.body.innerText.includes(arguments[0])", 900, sample.split(":")[0].split()[0])
            idle(b)
            shot(f"{subject}_curriculum", 1000)
            b.js(TAB, "2 · Diagnostic")
            b.js(CLICK, "Start the diagnostic")
            idle(b)
            first = True
            while "Placement:" not in b.js(VISIBLE_TEXT):
                answer_mcq(b, bank, knows)
                if first:
                    shot(f"{subject}_diagnostic_question", 700)
                    first = False
                b.js(CLICK, "Submit")
                idle(b)
            shot(f"{subject}_placement", 760)
            b.js(TAB, "3 · Tutor")
            b.choose("Topic", "All topics")
            b.js(LABEL, "Explain")
            b.type_into("Ask a question about the material", tutor_q, enter=True)
            b.wait("return document.querySelectorAll('[data-testid=user],[data-testid=bot]').length >= 2", 300)
            idle(b)
            shot(f"{subject}_tutor_explain", 1000)
            if socratic:
                # a learner working on section 2.1 picks that topic: retrieval then stays within it
                b.choose("Topic", index["sections"][0]["number"] + " ")
                b.js(LABEL, "Socratic")
                q, replies = socratic
                for msg in [q] + replies:
                    before = b.js("return document.querySelectorAll('[data-testid=user],[data-testid=bot]').length")
                    b.type_into("Ask a question about the material", msg, enter=True)
                    b.wait("return document.querySelectorAll('[data-testid=user],[data-testid=bot]').length >= arguments[0] + 2", 300,
                           before)
                    idle(b)
                b.js("const c=[...document.querySelectorAll('.bubble-wrap, [role=log]')].find(x=>x.offsetParent!==null);"
                     " if(c) c.scrollTop=c.scrollHeight; return true")
                shot(f"{subject}_tutor_socratic", 1000)
                b.js(TAB, "4 · Practice")
                b.js(CLICK, "Start a practice quiz")
                idle(b)
                for i in range(6):                         # answer until the free-text question comes up
                    text = b.js(VISIBLE_TEXT)
                    if "Done." in text:
                        break
                    if b.js("return [...document.querySelectorAll('label')].some(x=>x.offsetParent!==null && "
                            "x.innerText.trim().startsWith('A) '))"):
                        answer_mcq(b, bank, knows if i % 2 == 0 else set())
                    else:
                        short = max((q for q in bank["questions"] if q["kind"] == "short"),
                                    key=lambda q: shown(q["prompt"], text))
                        b.type_into("Your answer", short["model_answer"])
                    b.js(CLICK, "Submit")
                    idle(b)
                    if i == 1:
                        shot(f"{subject}_practice_feedback", 900)
                shot(f"{subject}_practice_rubric", 1000)
                b.js(TAB, "5 · Review")
                b.js(CLICK, "Pretend it's the next day (demo)")
                idle(b)
                b.js(CLICK, "Review what's due")
                idle(b)
                shot(f"{subject}_review", 760)
            b.js(TAB, "6 · Progress")
            b.js(CLICK, "Refresh")
            idle(b)
            time.sleep(1.5)
            shot(f"{subject}_progress", 1000)
    finally:
        b.close()


if __name__ == "__main__":
    main()
