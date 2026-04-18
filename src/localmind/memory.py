import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

import os
os.environ["HF_HUB_DISABLE_DOWNLOAD_WARNINGS"] = "1"

from localmind.config import Config


@dataclass
class MemoryEntry:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class MemoryStore:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load()
        self._init_storage()
        self._init_vector_store()
        self._init_embeddings()

    def _init_storage(self) -> None:
        self.data_path = self.config.storage.path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.data_path / self.config.storage.sqlite_db

    def _init_vector_store(self) -> None:
        self.chroma = chromadb.PersistentClient(
            path=str(self.data_path / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"},
        )

    def _init_embeddings(self) -> None:
        self.embeddings = SentenceTransformer(self.config.rag.embeddings)

    def _generate_id(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def add(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        project: Optional[str] = None,
    ) -> str:
        entry_id = self._generate_id(content)

        memory_metadata = metadata or {}
        if project:
            memory_metadata["project"] = project
        memory_metadata["created_at"] = datetime.now().isoformat()

        self.collection.upsert(
            ids=[entry_id],
            documents=[content],
            metadatas=[memory_metadata],
        )

        self._save_to_sqlite(entry_id, content, memory_metadata)

        return entry_id

    def _save_to_sqlite(
        self, entry_id: str, content: str, metadata: dict
    ) -> None:
        import sqlite3

        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            INSERT OR REPLACE INTO memories (id, content, metadata, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (entry_id, content, json.dumps(metadata), datetime.now().isoformat()),
        )

        conn.commit()
        conn.close()

    def search(
        self,
        query: str,
        n_results: int = 5,
        project: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        query_embedding = self.embeddings.encode([query]).tolist()

        where = {"project": project} if project else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
        )

        memories = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                memories.append(
                    {
                        "id": results["ids"][0][i],
                        "content": doc,
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                        "distance": results["distances"][0][i]
                        if results["distances"]
                        else None,
                    }
                )

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
        deleted = False

        import sqlite3

        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
        conn.commit()
        if cursor.rowcount > 0:
            deleted = True
        conn.close()

        try:
            self.collection.delete(ids=[entry_id])
            deleted = True
        except Exception:
            pass

        return deleted

    def list_all(
        self, limit: int = 100, project: Optional[str] = None
    ) -> list[dict[str, Any]]:
        where = {"project": project} if project else None

        result = self.collection.get(where=where, limit=limit)

        memories = []
        if result["documents"]:
            for i, doc in enumerate(result["documents"]):
                memories.append(
                    {
                        "id": result["ids"][i],
                        "content": doc,
                        "metadata": result["metadatas"][i] if result["metadatas"] else {},
                    }
                )

        return memories

    def clear(self, project: Optional[str] = None) -> int:
        if project:
            self.collection.delete(where={"project": project})
        else:
            self.collection.delete(where=None)

        import sqlite3

        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        if project:
            try:
                project_filter = f'%"project":"{project}"%'
                cursor.execute(
                    "DELETE FROM memories WHERE metadata LIKE ? ESCAPE ''", (project_filter,)
                )
            except sqlite3.OperationalError:
                cursor.execute("SELECT id, metadata FROM memories")
                rows = cursor.fetchall()
                for row in rows:
                    try:
                        meta = json.loads(row[1])
                        if meta.get("project") == project:
                            cursor.execute("DELETE FROM memories WHERE id = ?", (row[0],))
                    except (json.JSONDecodeError, KeyError):
                        continue
        else:
            cursor.execute("DELETE FROM memories")

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_memories": self.collection.count(),
            "vector_db": self.config.storage.vector_db,
            "storage_path": str(self.data_path),
        }