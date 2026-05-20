"""RAG (Retrieval-Augmented Generation) pipeline for LocalMind."""

from pathlib import Path
from typing import Any, Optional

from localmind.memory import MemoryStore

# Extensions indexed by default
DEFAULT_EXTENSIONS = [
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".md", ".txt", ".rst",
    ".json", ".yaml", ".yml", ".toml",
    ".html", ".css",
    ".sh", ".bash",
    ".go", ".rs", ".java", ".cpp", ".c", ".h",
]

DEFAULT_EXCLUDE_DIRS = [
    "node_modules", ".git", "__pycache__",
    ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt",
    ".mypy_cache", ".ruff_cache", ".pytest_cache",
    "coverage", "htmlcov",
]

# Safety: never index these regardless of user input
BLOCKED_ROOTS = ["/etc", "/sys", "/proc", "/dev", "/root", "/boot"]


class RAGPipeline:
    def __init__(self, memory_store: MemoryStore):
        self.memory = memory_store

    def _is_safe_path(self, path: Path) -> bool:
        resolved = path.resolve()
        return not any(str(resolved).startswith(b) for b in BLOCKED_ROOTS)

    def index_directory(
        self,
        directory: Path,
        project: str,
        extensions: Optional[list[str]] = None,
        exclude_dirs: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        directory = Path(directory).resolve()

        if not self._is_safe_path(directory):
            raise ValueError(f"Indexing {directory} is not permitted for safety reasons.")

        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        extensions = extensions or DEFAULT_EXTENSIONS
        exclude_dirs = exclude_dirs or DEFAULT_EXCLUDE_DIRS

        indexed_count = 0
        errors: list[dict[str, str]] = []

        for ext in extensions:
            for file_path in directory.rglob(f"*{ext}"):
                if any(excluded in file_path.parts for excluded in exclude_dirs):
                    continue
                if not self._is_safe_path(file_path):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    if not content.strip():
                        continue
                    chunks = self._chunk_text(content)

                    for chunk in chunks:
                        self.memory.add(
                            content=f"File: {file_path.relative_to(directory)}\n\n{chunk}",
                            metadata={
                                "file": str(file_path.relative_to(directory)),
                                "type": "rag",
                                "extension": ext,
                            },
                            project=project,
                        )
                        indexed_count += 1

                except Exception as e:
                    errors.append({"file": str(file_path), "error": str(e)})

        return {
            "indexed": indexed_count,
            "project": project,
            "directory": str(directory),
            "errors": errors,
        }

    def index_file(self, file_path: Path, project: str) -> dict[str, Any]:
        file_path = Path(file_path).resolve()

        if not self._is_safe_path(file_path):
            raise ValueError(f"Indexing {file_path} is not permitted for safety reasons.")

        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")

        content = file_path.read_text(encoding="utf-8", errors="replace")
        chunks = self._chunk_text(content)

        indexed_count = 0
        for chunk in chunks:
            self.memory.add(
                content=f"File: {file_path.name}\n\n{chunk}",
                metadata={
                    "file": str(file_path),
                    "type": "rag",
                    "extension": file_path.suffix,
                },
                project=project,
            )
            indexed_count += 1

        return {
            "indexed": indexed_count,
            "file": str(file_path),
            "project": project,
        }

    def _chunk_text(self, text: str) -> list[str]:
        chunk_size = self.memory.config.rag.chunk_size
        chunk_overlap = self.memory.config.rag.chunk_overlap

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                # Prefer splitting on newline boundaries
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
