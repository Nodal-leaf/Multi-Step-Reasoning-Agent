from __future__ import annotations

import re
import threading
from typing import Any

import requests


class LLMException(Exception):
    pass


_THINKING_RE = re.compile(
    r"(?:<\|im_start\|>?think.*?<\|im_end\|>|```?thinking.*?```?|"
    r"\bthink\b\s*" + r"\n[\s\S]*?" + r"\bassistant\b)",
    re.I,
)


def _strip_thinking(text: str) -> str:
    stripped = _THINKING_RE.sub("", text)
    return stripped.strip() if stripped else text.strip()


def _normalize(text: str) -> str:
    table = {
        "\u2018": "'", "\u2019": "'", "\u201b": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"',
        "\u2014": " -- ", "\u2013": "-", "\u2026": "...",
        "\u2022": "-", "\u2023": "-", "\u2043": "-", "\u00b7": "-",
    }
    for src, dst in table.items():
        text = text.replace(src, dst)
    return text


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        num_ctx: int = 8192,
        timeout: float = 600.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()

    def chat(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.7,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
        except requests.ConnectionError as exc:
            raise LLMException(
                f"Cannot reach Ollama at {self.base_url!r}. "
                "Start the server (`ollama serve`) and try again."
            ) from exc
        except requests.Timeout as exc:
            raise LLMException(f"Ollama request timed out after {self.timeout}s.") from exc
        except requests.RequestException as exc:
            raise LLMException(f"Ollama request failed: {exc}") from exc

        if response.status_code == 404:
            raise LLMException(
                f"Model {self.model!r} is not available. Pull it with: ollama pull {self.model}"
            )
        if response.status_code != 200:
            raise LLMException(f"Ollama API error {response.status_code}: {response.text[:500]}")

        data: dict[str, Any] = response.json()
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMException(f"Unexpected Ollama response shape: {str(data)[:500]}")
        return _normalize(_strip_thinking(message["content"]))


class MemoryOllamaClient(OllamaClient):
    def __init__(self, replies: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def chat(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.7,
    ) -> str:
        with self._lock:
            self.calls.append(
                dict(system=system, user=user, json_mode=json_mode, temperature=temperature)
            )
            if not self.replies:
                raise LLMException("No canned replies left in MemoryOllamaClient")
            return self.replies.pop(0)