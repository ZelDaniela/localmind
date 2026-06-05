"""
Pytest configuration — installs in-memory stubs for heavy dependencies
(chromadb, sentence-transformers, slowapi) so the full test suite runs
in CI without GPU drivers or large model downloads.

Stubs are installed before any test module is imported.
"""

from __future__ import annotations

import sys
import types


def _make_chroma_stub() -> None:
    chroma = types.ModuleType("chromadb")
    chroma_config = types.ModuleType("chromadb.config")

    class Settings:
        def __init__(self, **kwargs: object) -> None:
            pass

    class Collection:
        def __init__(self) -> None:
            self._store: dict[str, dict] = {}

        def upsert(self, ids: list, documents: list, metadatas: list | None = None) -> None:
            for i, id_ in enumerate(ids):
                self._store[id_] = {
                    "doc": documents[i],
                    "meta": (metadatas or [{}])[i],
                }

        def query(self, query_embeddings: list, n_results: int = 5, where: dict | None = None) -> dict:
            items = list(self._store.items())
            if where:
                k, v = next(iter(where.items()))
                items = [(i, d) for i, d in items if d["meta"].get(k) == v]
            items = items[:n_results]
            return {
                "ids": [[i for i, _ in items]],
                "documents": [[d["doc"] for _, d in items]],
                "metadatas": [[d["meta"] for _, d in items]],
                "distances": [[0.1] * len(items)],
            }

        def get(self, ids: list | None = None, where: dict | None = None, limit: int = 100) -> dict:
            if ids is not None:
                items = [(i, self._store[i]) for i in ids if i in self._store]
            else:
                items = list(self._store.items())
                if where:
                    k, v = next(iter(where.items()))
                    items = [(i, d) for i, d in items if d["meta"].get(k) == v]
                items = items[:limit]
            return {
                "ids": [i for i, _ in items],
                "documents": [d["doc"] for _, d in items],
                "metadatas": [d["meta"] for _, d in items],
            }

        def delete(self, ids: list | None = None, where: dict | None = None) -> None:
            if ids is not None:
                for i in ids:
                    self._store.pop(i, None)
            elif where:
                k, v = next(iter(where.items()))
                gone = [i for i, d in self._store.items() if d["meta"].get(k) == v]
                for i in gone:
                    del self._store[i]

        def count(self) -> int:
            return len(self._store)

    class PersistentClient:
        """Each instance has its own isolated collection registry."""

        def __init__(self, path: str | None = None, settings: object = None) -> None:
            # Per-instance collections — no shared state between tests
            self._collections: dict[str, Collection] = {}

        def get_or_create_collection(self, name: str, metadata: dict | None = None) -> Collection:
            if name not in self._collections:
                self._collections[name] = Collection()
            return self._collections[name]

    chroma.PersistentClient = PersistentClient  # type: ignore[attr-defined]
    chroma_config.Settings = Settings  # type: ignore[attr-defined]
    sys.modules["chromadb"] = chroma
    sys.modules["chromadb.config"] = chroma_config


def _make_sentence_transformers_stub() -> None:
    st = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, model_name: str) -> None:
            pass

        def encode(self, texts: list) -> object:
            import numpy as np
            return np.zeros((len(texts), 384))

    st.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = st


def _make_slowapi_stub() -> None:
    slow = types.ModuleType("slowapi")
    slow_util = types.ModuleType("slowapi.util")
    slow_err = types.ModuleType("slowapi.errors")

    class Limiter:
        def __init__(self, **kwargs: object) -> None:
            pass

    class RateLimitExceeded(Exception):
        pass

    slow.Limiter = Limiter  # type: ignore[attr-defined]
    slow._rate_limit_exceeded_handler = lambda req, exc: None  # type: ignore[attr-defined]
    slow_util.get_remote_address = lambda req: "127.0.0.1"  # type: ignore[attr-defined]
    slow_err.RateLimitExceeded = RateLimitExceeded  # type: ignore[attr-defined]
    sys.modules["slowapi"] = slow
    sys.modules["slowapi.util"] = slow_util
    sys.modules["slowapi.errors"] = slow_err


if "chromadb" not in sys.modules:
    _make_chroma_stub()

if "sentence_transformers" not in sys.modules:
    _make_sentence_transformers_stub()

if "slowapi" not in sys.modules:
    _make_slowapi_stub()
