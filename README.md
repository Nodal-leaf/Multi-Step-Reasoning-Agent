# Multi-Step Reasoning Agent

A CLI agent that answers questions using plan → reason → critique → synthesize, powered by a local [Ollama](https://ollama.com) model.

Instead of generating an answer in one shot, the agent explicitly plans its approach, works through a bounded number of reasoning steps, critiques each step (deciding whether to continue, revise, or stop), and finally synthesizes a clean answer from the full reasoning trace.

## Features

- **Plan → Reason → Critique → Synthesize** pipeline with a persistent reasoning scratchpad.
- **Self-critique loop**: each reasoning step is verified; the agent decides to continue, revise, or accept, and stops early once the reasoning is satisfactory.
- **Local and private**: everything runs against your own Ollama server; no external API calls.
- **Optional thought trace**: reveal the full reasoning trace (`-t`) or hide it for a clean answer.
- **Interactive REPL** (`-i`) with a `t` hotkey to toggle the trace.
- Rich terminal output (colors, panels, markdown) via [Rich](https://github.com/Textualize/rich).

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- A pulled model (default: `qwen3:8b`)

```bash
ollama pull qwen3:8b
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### One-off question

```bash
python agent.py "What is the capital of France?"
```

### Show the reasoning trace

```bash
python agent.py -t "Why is the sky blue?"
```

### Interactive REPL

```bash
python agent.py -i
```

Inside the REPL, type `t` to toggle the thought trace and `quit` / `exit` / `q` to leave.

### Piping input

```bash
echo "Explain recursion to a 10 year old" | python agent.py
```

## Options

| Flag | Description |
|------|-------------|
| `question` | The question to reason about (as positional args) |
| `--model` | Ollama model to use (default: `qwen3:8b`) |
| `--host` | Ollama server URL (default: `http://localhost:11434`) |
| `--ctx` | Context window size (default: `8192`) |
| `--max-steps` | Max reasoning steps (default: `4`) |
| `-t, --show-thought` | Show the reasoning trace (default: hidden) |
| `-i, --interactive` | Run an interactive REPL |
| `--no-color` | Disable ANSI colors |

## How it works

1. **Plan** – the question is broken down into ordered sub-problems.
2. **Reason** – each step builds directly on the previous ones, addressing critiques when present.
3. **Critique** – a lower-temperature pass assesses the reasoning so far and returns a JSON verdict: `satisfactory`, `continue`, or `revise`. The loop stops early on `satisfactory`, bounded by `--max-steps`.
4. **Synthesize** – the final answer is written cleanly from the reasoning trace, without referencing the process itself.

The pipeline lives in `reasoning_agent/pipeline.py`, and LLM interaction (including stripping model "thinking" blocks) is in `reasoning_agent/llm.py`.

## Project layout

```
agent.py               CLI entry point
reasoning_agent/
  llm.py               Ollama client, LLMException, MemoryOllamaClient
  pipeline.py          ReasoningAgent, Scratchpad, Verdict, prompts
  __init__.py          public exports
tests/
  test_cli.py          TracePrinter tests
  test_pipeline.py     pipeline / verdict parsing tests
```

## Running tests

```bash
python -m unittest discover -s tests
```