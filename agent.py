from __future__ import annotations

import argparse
import sys

from rich.box import ROUNDED
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from reasoning_agent import LLMException, OllamaClient, ReasoningAgent

QUIT_WORDS = {"quit", "exit", "q"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Multi-step reasoning agent powered by a local Ollama model.",
    )
    parser.add_argument("question", nargs="*", help="question to reason about")
    parser.add_argument("--model", default="qwen3:8b", help="ollama model (default: qwen3:8b)")
    parser.add_argument("--host", default="http://localhost:11434", help="ollama server URL")
    parser.add_argument("--ctx", type=int, default=8192, help="context window size")
    parser.add_argument("--max-steps", type=int, default=4, help="max reasoning steps (default: 4)")
    parser.add_argument(
        "-t",
        "--show-thought",
        action="store_true",
        help="show the reasoning trace (default: hidden)",
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="run an interactive REPL")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    return parser


class TracePrinter:
    DECISION_STYLE = {
        "satisfactory": "green",
        "continue": "yellow",
        "revise": "magenta",
    }

    def __init__(self, console: Console, show_thought: bool, max_steps: int) -> None:
        self.console = console
        self.show = show_thought
        self.max_steps = max_steps

    def event(self, kind: str, data: object) -> None:
        if kind == "plan_start":
            self._status("Planning...")
        elif kind == "plan_end":
            if self.show:
                self._block("Plan", "bold cyan", str(data))
        elif kind == "reason_start":
            self._status(f"Reasoning step {data}/{self.max_steps}...")
        elif kind == "reason_end":
            if self.show:
                self._block(f"Step {data['step']} of {self.max_steps}", "bold yellow", data["content"])
        elif kind == "critique_start":
            self._status("Checking...")
        elif kind == "critique_end":
            if self.show:
                verdict = data["verdict"]
                style = f"bold {self.DECISION_STYLE.get(verdict.decision, 'white')}"
                self.console.print(Text(f"[{verdict.decision}] {verdict.critique}", style=style))
                self.console.print()
        elif kind == "synthesize_start":
            self._status("Writing final answer...")

    def _status(self, text: str) -> None:
        if not self.show:
            self.console.print(Text(text, style="dim"))

    def _block(self, title: str, style: str, content: str) -> None:
        self.console.print()
        self.console.print(
            Panel(
                Markdown(content.strip()),
                title=Text(title, style=style),
                title_align="left",
                box=ROUNDED,
                border_style="blue",
            )
        )


def run_single(question: str, args: argparse.Namespace, console: Console, show_thought: bool) -> bool:
    llm = OllamaClient(host=args.host, model=args.model, num_ctx=args.ctx)
    printer = TracePrinter(console, show_thought, args.max_steps)
    agent = ReasoningAgent(llm, max_steps=args.max_steps, on_event=printer.event)
    try:
        answer = agent.run(question)
    except LLMException as exc:
        console.print(Text(f"Error: {exc}", style="bold red"))
        return False
    console.print()
    console.print(Markdown(answer))
    return True


def interactive(args: argparse.Namespace, console: Console) -> None:
    show_thought = args.show_thought
    console.print(
        Text(
            "Ollama reasoning agent. Ask a question; 't' toggles the thought trace; 'quit' exits.",
            style="dim",
        )
    )
    while True:
        try:
            raw = console.input("[bold cyan]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        command = raw.lower()
        if command in QUIT_WORDS:
            return
        if command == "t":
            show_thought = not show_thought
            console.print(Text(f"Thought trace {('shown' if show_thought else 'hidden')}.", style="yellow"))
            continue
        if not command:
            continue
        run_single(raw, args, console, show_thought)


def main() -> None:
    args = build_parser().parse_args()
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="replace")
    console = Console(no_color=args.no_color)

    try:
        if args.question:
            run_single(" ".join(args.question), args, console, args.show_thought)
        elif not sys.stdin.isatty():
            question = sys.stdin.read().strip()
            if question:
                run_single(question, args, console, args.show_thought)
            else:
                interactive(args, console)
        else:
            interactive(args, console)
    except KeyboardInterrupt:
        console.print()
        sys.exit(130)
    except LLMException as exc:
        console.print(Text(f"Error: {exc}", style="bold red"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()