"""Grading: deterministic wherever possible; the LLM only judges free text, and never does the arithmetic.

* **Multiple choice**: exact comparison with the key.
* **Exact answers** (an equation's solution): the learner's answer and the key are parsed to exact
  fractions, so "x = 3/2", "1.5" and "x=1.5" are all correct. ``solve_linear`` solves one-variable
  linear equations exactly (a small safe expression evaluator), which is also how generated algebra
  questions are checked before they are used.
* **Rubric (free text)**: each rubric point has keywords taken from the course material. A point is
  met if the answer contains one of its keywords (after normalising case, punctuation and simple word
  endings). Points the keywords miss can be checked by the LLM, which must **quote the part of the
  learner's answer** that makes the point; the quote is verified to be in the answer, so the model
  can't award credit for something the learner didn't write. The score is computed here, from the
  rubric weights, and the rubric is shown to the learner with the result.

Every grade also yields an SM-2 quality (0-5) for the spaced-repetition scheduler.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from fractions import Fraction

MINUS = str.maketrans({"−": "-", "–": "-", "—": "-", "×": "*", "·": "*", "÷": "/"})


def normalise(text: str) -> str:
    """Lower-case, unify minus signs, keep letters/digits/basic math, collapse spaces."""
    t = text.translate(MINUS).lower()
    t = re.sub(r"[^a-z0-9.+\-*/=^%\s]", " ", t)
    t = re.sub(r"(?<![0-9])\.|\.(?![0-9])", " ", t)          # full stops, not decimal points
    return re.sub(r"\s+", " ", t).strip()


def stem(word: str) -> str:
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(word) > len(suf) + 2 and word.endswith(suf):
            return word[: -len(suf)] + ("y" if suf == "ies" else "")
    return word


def contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word phrase match on stemmed tokens ('closed loops' matches 'closed loop')."""
    t = [stem(w) for w in normalise(text).split()]
    p = [stem(w) for w in normalise(phrase).split()]
    if not p:
        return False
    return any(t[i:i + len(p)] == p for i in range(len(t) - len(p) + 1))


# ------------------------------------------------------------------ exact numbers and equations


def parse_number(s: str) -> Fraction | None:
    """'5', '-3/2', '1.5', 'x = 5', 'x=−3/4' -> Fraction; anything else -> None."""
    s = s.translate(MINUS).strip().rstrip(".")
    s = re.sub(r"^[a-z]\s*=\s*", "", s, flags=re.I)
    m = re.fullmatch(r"\s*(-?\s*\d+(?:\.\d+)?)\s*(?:/\s*(-?\d+(?:\.\d+)?))?\s*", s)
    if not m:
        return None
    try:
        num = Fraction(m.group(1).replace(" ", ""))
        return num / Fraction(m.group(2)) if m.group(2) else num
    except (ValueError, ZeroDivisionError):
        return None


def _to_python(expr: str, var: str) -> str:
    e = expr.translate(MINUS).replace("^", "**")
    e = re.sub(r"(\d)\s*(?=[a-z(])", r"\1*", e)             # 4x, 5(  -> 4*x, 5*(
    e = re.sub(rf"({var}|\))\s*(?=[\d(a-z])", r"\1*", e)    # x(, )(, )x -> x*(, )*(, )*x
    return e


def _eval(node, x: Fraction):
    if isinstance(node, ast.Expression):
        return _eval(node.body, x)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.Name):
        return x
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand, x)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = _eval(node.left, x), _eval(node.right, x)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        return a / b
    raise ValueError("unsupported expression")


def solve_linear(equation: str) -> Fraction | None:
    """Exact solution of a one-variable linear equation such as '5(n - 4) - 4n = -8', else None.

    Evaluates f(v) = left - right at v = 0, 1, 2 with exact fractions; f must be linear with a
    non-zero slope. Only numbers, one single-letter variable, + - * / and parentheses are accepted.
    """
    eq = equation.translate(MINUS).strip().rstrip(".").lower()
    eq = re.sub(r"^(solve|find|evaluate)\s*[:\-]?\s*", "", eq)
    if eq.count("=") != 1:
        return None
    variables = set(re.findall(r"(?<![a-z])[a-z](?![a-z])", eq))
    if len(variables) != 1 or re.search(r"[a-z]{2,}", eq):
        return None
    var = variables.pop()
    left, right = eq.split("=")
    try:
        trees = [ast.parse(_to_python(side, var).strip(), mode="eval") for side in (left, right)]
        names = {n.id for t in trees for n in ast.walk(t) if isinstance(n, ast.Name)}
        if names - {var}:
            return None
        f = [_eval(trees[0], Fraction(v)) - _eval(trees[1], Fraction(v)) for v in (0, 1, 2)]
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None
    slope = f[1] - f[0]
    if slope == 0 or f[2] - f[1] != slope:              # no unique solution, or not linear
        return None
    return -f[0] / slope


# ------------------------------------------------------------------ grades


@dataclass
class PointResult:
    point: str
    met: bool
    method: str            # "keyword" | "llm-quote" | "none"
    evidence: str = ""


@dataclass
class Grade:
    score: float                   # 0..1
    correct: bool
    quality: int                   # SM-2 quality 0..5
    feedback: str
    points: list[PointResult] = field(default_factory=list)


def sm2_quality(score: float, kind: str) -> int:
    """Map a score to SM-2's 0-5 scale. A correct multiple-choice answer is 4 (correct, but recognition
    is easier than recall); rubric scores map in steps of a quarter."""
    if kind == "mcq":
        return 4 if score >= 1 else 1
    if score >= 0.999:
        return 5
    return 4 if score >= 0.75 else 3 if score >= 0.5 else 2 if score >= 0.25 else 1 if score > 0 else 0


def grade_mcq(q, choice: int | None) -> Grade:
    ok = choice is not None and int(choice) == q.answer_index
    right = "ABCD"[q.answer_index] + ") " + q.options[q.answer_index]
    fb = ("Correct. " if ok else f"Not quite: the answer is {right}. ") + (q.explanation or "")
    return Grade(1.0 if ok else 0.0, ok, sm2_quality(1.0 if ok else 0.0, "mcq"), f"{fb} (Source: {q.citation})")


def grade_exact(q, answer: str) -> Grade:
    want, got = Fraction(q.exact_answer), parse_number(answer)
    ok = got is not None and got == want
    if got is None:
        fb = f"I couldn't read a number in your answer. The solution is {q.model_answer}."
    else:
        fb = "Correct." if ok else f"Not quite: the solution is {q.model_answer}."
    return Grade(1.0 if ok else 0.0, ok, 5 if ok else 1, f"{fb} (Source: {q.citation})")


GRADE_SCHEMA = {"type": "object", "required": ["points"], "properties": {"points": {"type": "array", "items": {
    "type": "object", "required": ["index", "met", "quote"],
    "properties": {"index": {"type": "integer"}, "met": {"type": "boolean"}, "quote": {"type": "string"}}}}}}

GRADE_SYSTEM = """You check a learner's short answer against rubric points. For each point, decide whether the \
learner's answer makes that point (in any wording). If it does, copy into "quote" the exact words from the \
learner's answer that make it. If it does not, set met to false and quote to "". Judge only the rubric points; \
do not grade style or spelling."""


def grade_rubric(q, answer: str, provider=None) -> Grade:
    results = []
    for p in q.rubric:
        kw = next((k for k in p.keywords if contains_phrase(answer, k)), None)
        results.append(PointResult(p.point, kw is not None, "keyword" if kw else "none", kw or ""))
    missing = [i for i, r in enumerate(results) if not r.met]
    if missing and provider is not None and answer.strip():
        listing = "\n".join(f"{i}. {q.rubric[i].point}" for i in missing)
        try:
            d = provider.json(GRADE_SYSTEM, f"Question: {q.prompt}\n\nRubric points to check:\n{listing}\n\n"
                                            f"Learner's answer:\n{answer}", GRADE_SCHEMA, temperature=0.0, seed=0)
            for item in d.get("points", []):
                i, quote = item.get("index"), str(item.get("quote", "")).strip()
                # credit only if the quoted words really are in the learner's answer (and say something)
                if i in missing and item.get("met") and len(quote.split()) >= 2 and \
                        normalise(quote) in normalise(answer):
                    results[i] = PointResult(q.rubric[i].point, True, "llm-quote", quote)
        except Exception:  # the LLM is optional here: keyword grading already stands on its own
            pass
    total = sum(p.weight for p in q.rubric) or 1.0
    score = sum(p.weight for p, r in zip(q.rubric, results) if r.met) / total
    lines = [f"{'✓' if r.met else '✗'} {r.point}" for r in results]
    fb = (f"Score {score:.0%}.\n" + "\n".join(lines) + f"\nModel answer: {q.model_answer} (Source: {q.citation})")
    return Grade(round(score, 3), score >= 0.999, sm2_quality(score, "short"), fb, results)


def grade(q, response, provider=None) -> Grade:
    """Dispatch on the question type. ``response`` is an option index (mcq) or text (short)."""
    if q.kind == "mcq":
        return grade_mcq(q, response)
    if q.exact_answer:
        return grade_exact(q, str(response))
    return grade_rubric(q, str(response), provider)
