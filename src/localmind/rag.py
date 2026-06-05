"""RAG (Retrieval-Augmented Generation) pipeline for LocalMind."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from localmind.memory import MemoryStore

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = [
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".md", ".txt", ".rst",
    ".json", ".yaml", ".yml", ".toml",
    ".html", ".css", ".sh", ".bash",
    ".go", ".rs", ".java", ".cpp", ".c", ".h",
]

DEFAULT_EXCLUDE_DIRS: set[str] = {
    "node_modules", ".git", "__pycache__",
    ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt",
    ".mypy_cache", ".ruff_cache", ".pytest_cache",
    "coverage", "htmlcov", ".tox", ".eggs", "chroma",
}

_BLOCKED_ROOTS: tuple[str, ...] = (
    "/etc", "/sys", "/proc", "/dev", "/root", "/boot",
    "/run", "/snap", "/lost+found",
)

_MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MB per file


class RAGPipeline:
    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory = memory_store

    def _is_safe_path(self, path: Path) -> bool:
        resolved = str(path.resolve())
        return not any(
            resolved == b or resolved.startswith(b + "/")
            for b in _BLOCKED_ROOTS
        )

    def index_directory(
        self,
        directory: Path,
        project: str,
        extensions: Optional[list[str]] = None,
        exclude_dirs: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        directory = Path(directory).resolve()

        if not self._is_safe_path(directory):
            raise ValueError(f"Indexing '{directory}' is not permitted for safety reasons.")
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        exts = extensions or DEFAULT_EXTENSIONS
        excluded = exclude_dirs or DEFAULT_EXCLUDE_DIRS

        indexed_count = 0
        skipped_count = 0
        errors: list[dict[str, str]] = []

        for ext in exts:
            for file_path in directory.rglob(f"*{ext}"):
                if any(part in excluded for part in file_path.parts):
                    skipped_count += 1
                    continue
                if not self._is_safe_path(file_path):
                    skipped_count += 1
                    continue
                try:
                    if file_path.stat().st_size > _MAX_FILE_BYTES:
                        logger.warning("Skipping large file: %s", file_path)
                        skipped_count += 1
                        continue
                except OSError:
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    if not content.strip():
                        continue
                    rel = file_path.relative_to(directory)
                    for chunk in self._chunk_text(content):
                        self.memory.add(
                            content=f"File: {rel}\n\n{chunk}",
                            metadata={"file": str(rel), "type": "rag", "extension": ext},
                            project=project,
                        )
                        indexed_count += 1
                except Exception as e:
                    errors.append({"file": str(file_path), "error": str(e)})
                    logger.warning("Error indexing %s: %s", file_path, e)

        return {
            "indexed": indexed_count,
            "skipped": skipped_count,
            "project": project,
            "directory": str(directory),
            "errors": errors,
        }

    def index_file(self, file_path: Path, project: str) -> dict[str, Any]:
        file_path = Path(file_path).resolve()

        if not self._is_safe_path(file_path):
            raise ValueError(f"Indexing '{file_path}' is not permitted for safety reasons.")
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        size = file_path.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise ValueError(
                f"File too large ({size // 1024} KB). Max: {_MAX_FILE_BYTES // 1024} KB."
            )

        content = file_path.read_text(encoding="utf-8", errors="replace")
        chunks = self._chunk_text(content)
        indexed_count = 0
        for chunk in chunks:
            self.memory.add(
                content=f"File: {file_path.name}\n\n{chunk}",
                metadata={"file": str(file_path), "type": "rag", "extension": file_path.suffix},
                project=project,
            )
            indexed_count += 1

        return {"indexed": indexed_count, "file": str(file_path), "project": project}

    def _chunk_text(self, text: str) -> list[str]:
        chunk_size = self.memory.config.rag.chunk_size
        chunk_overlap = self.memory.config.rag.chunk_overlap

        if not text.strip():
            return []

        chunks: list[str] = []
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
            next_start = start + chunk_size - chunk_overlap
            if next_start <= start:
                break
            start = next_start

        return chunks

    def get_relevant_context(
        self,
        query: str,
        project: Optional[str] = None,
        max_tokens: int = 2000,
    ) -> str:
        if not query.strip():
            return ""
        memories = self.memory.search(query, n_results=10, project=project)
        parts: list[str] = []
        tokens = 0
        for mem in memories:
            chunk = mem["content"]
            chunk_tokens = len(chunk) // 4
            if tokens + chunk_tokens > max_tokens:
                break
            parts.append(chunk)
            tokens += chunk_tokens
        return "\n\n---\n\n".join(parts)
