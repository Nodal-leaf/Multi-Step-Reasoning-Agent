import io
import unittest

from rich.console import Console

from agent import TracePrinter
from reasoning_agent import Verdict


def make_console() -> Console:
    return Console(file=io.StringIO(), no_color=True, force_terminal=False)


def emit_all(printer: TracePrinter) -> None:
    printer.event("plan_start", None)
    printer.event("plan_end", "PLAN TEXT")
    printer.event("reason_start", 1)
    printer.event("reason_end", {"step": 1, "content": "REASON BODY"})
    printer.event("critique_start", None)
    printer.event("critique_end", {"step": 1, "verdict": Verdict("continue", "go deeper")})
    printer.event("synthesize_start", None)
    printer.event("synthesize_end", {"answer": "FINAL", "steps": 1})


class TracePrinterTest(unittest.TestCase):
    def test_thought_hidden_by_default(self):
        console = make_console()
        emit_all(TracePrinter(console, show_thought=False, max_steps=4))
        out = console.file.getvalue()
        self.assertIn("Planning...", out)
        self.assertIn("Reasoning step 1/4...", out)
        self.assertNotIn("PLAN TEXT", out)
        self.assertNotIn("REASON BODY", out)
        self.assertNotIn("go deeper", out)

    def test_thought_shown_when_enabled(self):
        console = make_console()
        emit_all(TracePrinter(console, show_thought=True, max_steps=4))
        out = console.file.getvalue()
        self.assertIn("Plan", out)
        self.assertIn("PLAN TEXT", out)
        self.assertIn("Step 1 of 4", out)
        self.assertIn("REASON BODY", out)
        self.assertIn("[continue] go deeper", out)


if __name__ == "__main__":
    unittest.main()