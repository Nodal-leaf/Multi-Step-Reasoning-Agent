import unittest

from reasoning_agent import LLMException, MemoryOllamaClient, ReasoningAgent
from reasoning_agent.pipeline import parse_verdict


class PipelineTest(unittest.TestCase):
    def test_breaks_early_when_satisfactory(self):
        replies = [
            "1. Understand the problem\n2. Solve it",
            "Reasoning step 1 body",
            '{"decision": "satisfactory", "critique": "looks solid"}',
            "THE FINAL ANSWER",
        ]
        llm = MemoryOllamaClient(replies)
        agent = ReasoningAgent(llm, max_steps=4)
        answer = agent.run("What is 2+2?")

        self.assertEqual(answer, "THE FINAL ANSWER")
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(llm.calls[2].get("json_mode"), True)

    def test_runs_all_steps_when_verdict_continues(self):
        replies = [
            "plan body",
            "reason #1",
            '{"decision": "continue", "critique": "go deeper"}',
            "reason #2",
            '{"decision": "revise", "critique": "fix the math"}',
            "FINAL",
        ]
        llm = MemoryOllamaClient(replies)
        agent = ReasoningAgent(llm, max_steps=2)
        answer = agent.run("question?")

        self.assertEqual(answer, "FINAL")
        self.assertEqual(len(llm.calls), 6)
        self.assertIn("go deeper", " ".join(str(c) for c in llm.calls))

    def test_scratchpad_carries_prior_steps_and_critiques(self):
        replies = [
            "plan body",
            "reason #1",
            '{"decision": "continue", "critique": "go deeper"}',
            "reason #2",
            '{"decision": "continue", "critique": "more"}',
            "FINAL",
        ]
        llm = MemoryOllamaClient(replies)
        agent = ReasoningAgent(llm, max_steps=2)
        agent.run("question?")

        reason_calls = [
            c for c in llm.calls
            if "Produce ONLY the next reasoning step" in c.get("user", "")
        ]
        self.assertEqual(len(reason_calls), 2)
        self.assertIn("Reasoning step 1", reason_calls[1]["user"])

    def test_runs_out_of_replies_raises(self):
        llm = MemoryOllamaClient([])
        agent = ReasoningAgent(llm, max_steps=2)
        with self.assertRaises(LLMException):
            agent.run("boom?")

    def test_parse_verdict_messy_text(self):
        verdict = parse_verdict('Here is the answer {"decision": "revise", "critique": "oops"} done')
        self.assertEqual(verdict.decision, "revise")
        self.assertEqual(verdict.critique, "oops")

    def test_parse_verdict_bad_value_defaults(self):
        verdict = parse_verdict('{"decision": "unknown", "critique": "x"}')
        self.assertEqual(verdict.decision, "continue")


if __name__ == "__main__":
    unittest.main()