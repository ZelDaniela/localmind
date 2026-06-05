"""Agent connectors for LocalMind (Ollama, Claude)."""

from __future__ import annotations

import logging
from typing import Any

import requests

from localmind.config import Config
from localmind.memory import MemoryStore
from localmind.rag import RAGPipeline

logger = logging.getLogger(__name__)


class OllamaAgent:
    def __init__(self, config: Config) -> None:
        self.base_url = config.agents.ollama.base_url.rstrip("/")
        self.model = config.agents.ollama.model

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def generate(self, prompt: str, context: str = "") -> str:
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        payload = {"model": self.model, "prompt": full_prompt, "stream": False}
        try:
            r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            return r.json().get("response", "")
        except Exception as e:
            logger.error("Ollama generate failed: %s", e)
            return f"Error: {e}"


class ClaudeAgent:
    def is_available(self) -> bool:
        try:
            import anthropic  # noqa: F401

            return True
        except ImportError:
            return False

    def generate(self, prompt: str, context: str = "") -> str:
        try:
            import anthropic

            client = anthropic.Anthropic()
            system = f"Relevant context:\n{context}" if context else "You are a helpful assistant."
            msg = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            logger.error("Claude generate failed: %s", e)
            return f"Error: {e}"


class AgentRegistry:
    def __init__(self, memory: MemoryStore, config: Config) -> None:
        self.memory = memory
        self.config = config
        self.ollama = OllamaAgent(config)
        self.claude = ClaudeAgent()
        self.rag = RAGPipeline(memory)

    def chat_with_memory(
        self,
        agent: str,
        message: str,
        use_rag: bool = True,
        project: str | None = None,
    ) -> dict[str, Any]:
        context = ""
        if use_rag:
            context = self.rag.get_relevant_context(message, project=project)

        if agent == "ollama":
            response = self.ollama.generate(message, context)
        elif agent == "claude":
            response = self.claude.generate(message, context)
        else:
            return {"error": f"Unknown agent: {agent}. Use 'ollama' or 'claude'."}

        # Store the exchange as a memory
        self.memory.add(
            f"Q: {message}\nA: {response}",
            metadata={"type": "conversation", "agent": agent},
            project=project,
        )

        return {
            "response": response,
            "agent": agent,
            "context_used": bool(context),
            "project": project,
        }
