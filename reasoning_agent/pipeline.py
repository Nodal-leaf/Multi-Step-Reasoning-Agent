from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from .llm import OllamaClient

SYSTEM = (
    "You are a rigorous, meticulous reasoning agent. You work step by step, never "
    "skipping steps, and you catch your own mistakes before presenting an answer. "
    "You reason honestly about the limits of what you know."
)

PLAN_PROMPT = (
    "QUESTION: {question}\n\n"
    "Break the question down into the key sub-problems that must be solved, in the "
    "order they should be tackled. Put each sub-problem on its own numbered line. "
    "Be specific and concise."
)

REASON_PROMPT = (
    "QUESTION: {question}\n\n"
    "PLAN:\n{plan}\n\n"
    "REASONING SO FAR:\n{scratchpad}\n\n"
    "Produce ONLY the next reasoning step (step {step} of {max_steps}) that continues "
    "from the reasoning so far. Build directly on the previous steps; do not repeat "
    "them. Work through the relevant sub-problems carefully. If a critique is listed "
    "in the reasoning so far, fix or address it in this step. End the step with your "
    "intermediate conclusion. Do not write the final answer here."
)

CRITIQUE_PROMPT = (
    "QUESTION: {question}\n\n"
    "REASONING SO FAR:\n{scratchpad}\n\n"
    "Assess the reasoning above. Reply with ONLY JSON matching this schema:\n"
    '{{"decision": "satisfactory" | "continue" | "revise", "critique": "one short sentence"}}\n\n'
    'Use "satisfactory" when the reasoning is complete, correct, self-consistent and '
    'sufficient to answer the question; "continue" when it must go deeper; '
    '"revise" when there is an error or gap to correct.'
)

SYNTHESIZE_PROMPT = (
    "QUESTION: {question}\n\n"
    "REASONING TRACE:\n{scratchpad}\n\n"
    "Write the final answer to the question. Use the reasoning trace, but present the "
    "answer cleanly and directly for the reader. Do not mention the planning or "
    "reasoning process. Give the best, most complete answer you can."
)

Event = Callable[[str, object], None]


@dataclass
class Verdict:
    decision: str
    critique: str

    @property
    def satisfactory(self) -> bool:
        return self.decision == "satisfactory"


class Scratchpad:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def add(self, label: str, content: str) -> None:
        self.entries.append((label, content))

    def render(self) -> str:
        blocks = [f"### {label}\n{content}" for label, content in self.entries]
        return "\n\n".join(blocks).strip()


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    decision = "continue"
    dm = re.search(r'"decision"\s*[:=]\s*"?(\w+)"?', text)
    if dm:
        decision = dm.group(1)
    return {"decision": decision, "critique": text}


def parse_verdict(raw: str) -> Verdict:
    data = _extract_json(raw)
    decision = data.get("decision", "continue")
    if decision not in {"satisfactory", "continue", "revise"}:
        decision = "continue"
    critique = data.get("critique", raw)
    return Verdict(decision=decision, critique=critique.strip())


class ReasoningAgent:
    def __init__(
        self,
        llm: OllamaClient,
        *,
        max_steps: int = 4,
        on_event: Event | None = None,
    ) -> None:
        self.llm = llm
        self.max_steps = max(1, int(max_steps))
        self.on_event = on_event

    def _emit(self, kind: str, data: object) -> None:
        if self.on_event:
            self.on_event(kind, data)

    def plan(self, question: str) -> str:
        raw = self.llm.chat(SYSTEM, PLAN_PROMPT.format(question=question))
        return raw.strip()

    def reason_step(self, question: str, plan: str, scratchpad: Scratchpad, step: int) -> str:
        prompt = REASON_PROMPT.format(
            question=question,
            plan=plan,
            scratchpad=scratchpad.render(),
            step=step,
            max_steps=self.max_steps,
        )
        return self.llm.chat(SYSTEM, prompt).strip()

    def critique(self, question: str, scratchpad: Scratchpad) -> Verdict:
        raw = self.llm.chat(
            SYSTEM,
            CRITIQUE_PROMPT.format(question=question, scratchpad=scratchpad.render()),
            json_mode=True,
            temperature=0.2,
        )
        return parse_verdict(raw)

    def synthesize(self, question: str, plan: str, scratchpad: Scratchpad) -> str:
        prompt = SYNTHESIZE_PROMPT.format(
            question=question, scratchpad=scratchpad.render()
        )
        return self.llm.chat(SYSTEM, prompt).strip()

    def run(self, question: str) -> str:
        question = question.strip()
        steps_taken = 0

        self._emit("plan_start", question)
        plan = self.plan(question)
        self._emit("plan_end", plan)

        scratchpad = Scratchpad()
        scratchpad.add("Question", question)
        scratchpad.add("Plan", plan)

        for step in range(1, self.max_steps + 1):
            steps_taken = step
            self._emit("reason_start", step)
            content = self.reason_step(question, plan, scratchpad, step)
            scratchpad.add(f"Reasoning step {step}", content)
            self._emit("reason_end", {"step": step, "content": content})

            self._emit("critique_start", step)
            verdict = self.critique(question, scratchpad)
            scratchpad.add(f"Critique {step}", f"[{verdict.decision}] {verdict.critique}")
            self._emit("critique_end", {"step": step, "verdict": verdict})

            if verdict.satisfactory:
                break

        self._emit("synthesize_start", steps_taken)
        answer = self.synthesize(question, plan, scratchpad)
        self._emit("synthesize_end", {"answer": answer, "steps": steps_taken})
        return answer