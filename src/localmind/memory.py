"""Persistent memory store for LocalMind."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from localmind.config import Config

logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.load()
        self._init_storage()
        self._init_vector_store()
        self._init_embeddings()
        self._ensure_schema()

    def _init_storage(self) -> None:
        self.data_path = self.config.storage.path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.data_path / self.config.storage.sqlite_db

    def _init_vector_store(self) -> None:
        chroma_path = self.data_path / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"},
        )

    def _init_embeddings(self) -> None:
        self.embeddings = SentenceTransformer(self.config.rag.embeddings)

    @contextmanager
    def _db(self) -> Generator[sqlite3.Cursor, None, None]:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._db() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_created
                ON memories(created_at DESC)
            """)

    def _generate_id(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def add(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        project: Optional[str] = None,
    ) -> str:
        if not content or not content.strip():
            raise ValueError("content cannot be empty")

        entry_id = self._generate_id(content)
        now = datetime.now().isoformat()

        memory_metadata: dict[str, Any] = dict(metadata or {})
        if project:
            memory_metadata["project"] = project
        memory_metadata["created_at"] = now

        self.collection.upsert(
            ids=[entry_id],
            documents=[content],
            metadatas=[memory_metadata],
        )

        with self._db() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO memories (id, content, metadata, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (entry_id, content, json.dumps(memory_metadata), now),
            )

        logger.debug("Memory stored: %s", entry_id)
        return entry_id

    def search(
        self,
        query: str,
        n_results: int = 5,
        project: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        n_results = max(1, min(n_results, 50))
        query_embedding = self.embeddings.encode([query]).tolist()
        where = {"project": project} if project else None

        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            logger.warning("Search failed: %s", e)
            return []

        memories = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                memories.append({
                    "id": results["ids"][0][i],
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })

        return memories

    def get(self, entry_id: str) -> Optional[dict[str, Any]]:
        result = self.collection.get(ids=[entry_id])
        if not result["documents"]:
            return None
        return {
            "id": result["ids"][0],
            "content": result["documents"][0],
            "metadata": result["metadatas"][0] if result["metadatas"] else {},
        }

    def delete(self, entry_id: str) -> bool:
        existing = self.collection.get(ids=[entry_id])
        if not existing["ids"]:
            return False
        self.collection.delete(ids=[entry_id])
        with self._db() as cursor:
            cursor.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
        return True

    def list_all(
        self, limit: int = 100, project: Optional[str] = None
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 10000))
        where = {"project": project} if project else None
        result = self.collection.get(where=where, limit=limit)

        if not result["documents"]:
            return []

        return [
            {
                "id": result["ids"][i],
                "content": result["documents"][i],
                "metadata": result["metadatas"][i] if result["metadatas"] else {},
            }
            for i in range(len(result["documents"]))
        ]

    def clear(self, project: Optional[str] = None) -> int:
        if project:
            self.collection.delete(where={"project": project})
            with self._db() as cursor:
                cursor.execute(
                    "DELETE FROM memories WHERE json_extract(metadata, '$.project') = ?",
                    (project,),
                )
                return cursor.rowcount
        else:
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
            with self._db() as cursor:
                cursor.execute("DELETE FROM memories")
                return cursor.rowcount

    def export_json(self, output_path: Path, project: Optional[str] = None) -> int:
        memories = self.list_all(limit=10000, project=project)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=2, ensure_ascii=False)
        return len(memories)

    def import_json(self, input_path: Path, project: Optional[str] = None) -> int:
        with open(input_path, encoding="utf-8") as f:
            memories = json.load(f)

        if not isinstance(memories, list):
            raise ValueError("Import file must contain a JSON array.")

        imported = 0
        for mem in memories:
            content = mem.get("content", "").strip()
            if not content:
                continue
            metadata = dict(mem.get("metadata", {}))
            if project:
                metadata["project"] = project
            self.add(content, metadata)
            imported += 1

        return imported

    def get_stats(self) -> dict[str, Any]:
        db_size = self.sqlite_path.stat().st_size if self.sqlite_path.exists() else 0
        chroma_path = self.data_path / "chroma"
        chroma_size = (
            sum(f.stat().st_size for f in chroma_path.rglob("*") if f.is_file())
            if chroma_path.exists() else 0
        )
        return {
            "total_memories": self.collection.count(),
            "vector_db": self.config.storage.vector_db,
            "storage_path": str(self.data_path),
            "sqlite_size_kb": round(db_size / 1024, 1),
            "chroma_size_kb": round(chroma_size / 1024, 1),
            "embeddings_model": self.config.rag.embeddings,
        }
