from pathlib import Path
from typing import Any, Optional

from localmind.memory import MemoryStore


class RAGPipeline:
    def __init__(self, memory_store: MemoryStore):
        self.memory = memory_store

    def index_directory(
        self,
        directory: Path,
        project: str,
        extensions: Optional[list[str]] = None,
        exclude_dirs: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        if extensions is None:
            extensions = [".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml"]

        if exclude_dirs is None:
            exclude_dirs = [
                "node_modules",
                ".git",
                "__pycache__",
                ".venv",
                "venv",
                "dist",
                "build",
            ]

        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        indexed_count = 0
        errors = []

        for ext in extensions:
            for file_path in directory.rglob(f"*{ext}"):
                if any(excluded in file_path.parts for excluded in exclude_dirs):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8")
                    chunks = self._chunk_text(content)

                    for chunk in chunks:
                        self.memory.add(
                            content=f"File: {file_path.relative_to(directory)}\n\n{chunk}",
                            metadata={"file": str(file_path.relative_to(directory))},
                            project=project,
                        )
                        indexed_count += 1

                except Exception as e:
                    errors.append({"file": str(file_path), "error": str(e)})

        return {
            "indexed": indexed_count,
            "project": project,
            "errors": errors,
        }

    def _chunk_text(self, text: str) -> list[str]:
        chunk_size = self.memory.config.rag.chunk_size
        chunk_overlap = self.memory.config.rag.chunk_overlap

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                newline_pos = text.rfind("\n", start, end)
                if newline_pos > start:
                    end = newline_pos

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start += chunk_size - chunk_overlap

            if start >= len(text):
                break

        return chunks

    def index_file(self, file_path: Path, project: str) -> dict[str, Any]:
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        chunks = self._chunk_text(content)

        indexed_count = 0
        for chunk in chunks:
            self.memory.add(
                content=f"File: {file_path.name}\n\n{chunk}",
                metadata={"file": str(file_path)},
                project=project,
            )
            indexed_count += 1

        return {
            "indexed": indexed_count,
            "file": str(file_path),
            "project": project,
        }

    def get_relevant_context(
        self, query: str, project: Optional[str] = None, max_tokens: int = 2000
    ) -> str:
        memories = self.memory.search(query, n_results=10, project=project)

        context_parts = []
        current_tokens = 0

        for memory in memories:
            chunk = memory["content"]
            chunk_tokens = len(chunk) // 4

            if current_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk)
            current_tokens += chunk_tokens

        return "\n\n---\n\n".join(context_parts)