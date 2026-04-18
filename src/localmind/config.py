from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class StorageConfig:
    path: Path = field(default_factory=lambda: Path.home() / ".localmind" / "data")
    vector_db: str = "chroma"
    sqlite_db: str = "localmind.db"


@dataclass
class RAGConfig:
    embeddings: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama2"


@dataclass
class ClaudeConfig:
    enabled: bool = True


@dataclass
class AgentConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)


@dataclass
class Config:
    storage: StorageConfig = field(default_factory=StorageConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        if config_path is None:
            config_path = Path.home() / ".localmind" / "config.yaml"

        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        if not data:
            return cls()

        storage_data = data.get("storage") or {}
        rag_data = data.get("rag") or {}
        agents_data = data.get("agents") or {}
        ollama_data = agents_data.get("ollama") or {}
        claude_data = agents_data.get("claude") or {}

        if "path" in storage_data and isinstance(storage_data["path"], str):
            storage_data["path"] = Path(storage_data["path"])

        return cls(
            storage=StorageConfig(**storage_data),
            rag=RAGConfig(**rag_data),
            agents=AgentConfig(
                ollama=OllamaConfig(**ollama_data),
                claude=ClaudeConfig(**claude_data),
            ),
        )

    def save(self, config_path: Optional[Path] = None) -> None:
        if config_path is None:
            config_path = Path.home() / ".localmind" / "config.yaml"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "storage": {
                "path": str(self.storage.path),
                "vector_db": self.storage.vector_db,
                "sqlite_db": self.storage.sqlite_db,
            },
            "rag": {
                "embeddings": self.rag.embeddings,
                "chunk_size": self.rag.chunk_size,
                "chunk_overlap": self.rag.chunk_overlap,
            },
            "agents": {
                "ollama": {
                    "base_url": self.agents.ollama.base_url,
                    "model": self.agents.ollama.model,
                },
                "claude": {
                    "enabled": self.agents.claude.enabled,
                },
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)