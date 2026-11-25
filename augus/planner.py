"""Planner Agent: TF-IDF (word 1-3 grams) + Logistic Regression (C=4) template classifier."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from augus.parser import extract_years

TEMPLATES: tuple[str, ...] = (
    "ratio",
    "percent change",
    "difference",
    "sum",
    "product",
    "other",
)

EXECUTABLE_TEMPLATES: frozenset[str] = frozenset(
    {"ratio", "percent change", "difference", "sum", "product"}
)

_STEP_RE = re.compile(r"([a-zA-Z_]+)\(([^()]*)\)")


def _split_program_steps(program: str) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = []
    for match in _STEP_RE.finditer(program or ""):
        op = match.group(1).strip().lower()
        args = [a.strip() for a in match.group(2).split(",")]
        steps.append((op, args))
    return steps


def _is_constant_arg(arg: str) -> bool:
    return arg.startswith("const_")


def classify_program_template(program: str) -> str:
    """Map a FinQA program string to one of the 6 AUGUS templates."""
    steps = _split_program_steps(program)
    if not steps:
        return "other"

    if len(steps) == 1:
        op, args = steps[0]
        if len(args) != 2 or any(_is_constant_arg(a) or a.startswith("#") for a in args):
            return "other"
        if op == "divide":
            return "ratio"
        if op == "subtract":
            return "difference"
        if op == "add":
            return "sum"
        if op == "multiply":
            return "product"
        return "other"

    if len(steps) == 2:
        (op1, args1), (op2, args2) = steps
        if (
            op1 == "subtract"
            and op2 == "divide"
            and len(args1) == 2
            and len(args2) == 2
            and not any(_is_constant_arg(a) or a.startswith("#") for a in args1)
            and args2[0] == "#0"
            and args2[1] == args1[1]
        ):
            return "percent change"

    return "other"


@dataclass
class PlannerOutput:
    """Structured output of the Planner Agent."""

    template: str
    pi_t: float
    other_prob: float
    probs: dict[str, float]
    question_years: list[int]


class PlannerAgent:
    """Classifies financial questions into reasoning templates and extracts years."""

    def __init__(self, seed: int = 42, c_reg: float = 4.0) -> None:
        self.seed = seed
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            lowercase=True,
            max_features=40000,
        )
        self.clf = LogisticRegression(
            C=c_reg,
            max_iter=1000,
            random_state=seed,
        )
        self._classes: list[str] = []

    def fit(self, questions: list[str], templates: list[str]) -> "PlannerAgent":
        """Train TF-IDF + LogisticRegression on FinQA training questions."""
        X = self.vectorizer.fit_transform(questions)
        self.clf.fit(X, templates)
        self._classes = list(self.clf.classes_)
        return self

    def predict(self, question: str) -> PlannerOutput:
        """Predict template, confidence pi_t, other_prob, and question years."""
        X = self.vectorizer.transform([question])
        proba = self.clf.predict_proba(X)[0]
        probs = {cls: float(p) for cls, p in zip(self._classes, proba)}
        best_idx = int(proba.argmax())
        template = self._classes[best_idx]
        pi_t = float(proba[best_idx])
        other_prob = float(probs.get("other", 0.0))
        return PlannerOutput(
            template=template,
            pi_t=pi_t,
            other_prob=other_prob,
            probs=probs,
            question_years=extract_years(question),
        )

    def predict_batch(self, questions: list[str]) -> list[PlannerOutput]:
        """Batch prediction for fast evaluation."""
        X = self.vectorizer.transform(questions)
        probas = self.clf.predict_proba(X)
        outputs: list[PlannerOutput] = []
        for q, proba in zip(questions, probas):
            probs = {cls: float(p) for cls, p in zip(self._classes, proba)}
            best_idx = int(proba.argmax())
            outputs.append(
                PlannerOutput(
                    template=self._classes[best_idx],
                    pi_t=float(proba[best_idx]),
                    other_prob=float(probs.get("other", 0.0)),
                    probs=probs,
                    question_years=extract_years(q),
                )
            )
        return outputs
