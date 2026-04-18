import json
from abc import ABC, abstractmethod
from typing import Any, Optional

import requests

from localmind.config import Config
from localmind.memory import MemoryStore
from localmind.rag import RAGPipeline


class BaseAgentConnector(ABC):
    @abstractmethod
    def chat(self, message: str, context: Optional[str] = None) -> str:
        pass


class OllamaConnector(BaseAgentConnector):
    def __init__(self, config: Config):
        self.base_url = config.agents.ollama.base_url
        self.model = config.agents.ollama.model

    def chat(self, message: str, context: Optional[str] = None) -> str:
        prompt = message
        if context:
            prompt = f"Context:\n{context}\n\nUser: {message}"

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.text}")

        return response.json().get("response", "")

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return response.json().get("models", [])
        except requests.RequestException:
            pass
        return []


class ClaudeConnector(BaseAgentConnector):
    def __init__(self, config: Config):
        self.config = config

    def chat(self, message: str, context: Optional[str] = None) -> str:
        raise NotImplementedError(
            "Claude CLI integration requires subprocess call. "
            "Use CLI commands for now."
        )

    def is_available(self) -> bool:
        import shutil

        return shutil.which("claude") is not None


class AgentRegistry:
    def __init__(self, memory: MemoryStore, config: Config):
        self.memory = memory
        self.config = config
        self.rag = RAGPipeline(memory)
        self.ollama = OllamaConnector(config)
        self.claude = ClaudeConnector(config)

    def get_connector(self, agent_name: str) -> Optional[BaseAgentConnector]:
        connectors = {
            "ollama": self.ollama,
            "claude": self.claude,
        }
        return connectors.get(agent_name)

    def chat_with_memory(
        self,
        agent: str,
        message: str,
        use_rag: bool = True,
        project: Optional[str] = None,
    ) -> dict[str, Any]:
        context = None

        if use_rag:
            context = self.rag.get_relevant_context(message, project=project)
        else:
            memories = self.memory.search(message, project=project)
            if memories:
                context = "\n".join(m["content"] for m in memories)

        connector = self.get_connector(agent)
        if not connector:
            raise ValueError(f"Unknown agent: {agent}")

        if agent == "claude":
            response = f"[Claude CLI required] Message: {message}\nContext: {context or 'None'}"
        else:
            response = connector.chat(message, context)

        self.memory.add(
            content=f"User: {message}\nAgent: {response}",
            metadata={"agent": agent, "type": "conversation"},
            project=project,
        )

        return {
            "response": response,
            "context_used": context is not None,
            "memories_found": self.memory.search(message, project=project).__len__(),
        }