import pytest
from pathlib import Path
from localmind.memory import MemoryStore
from localmind.config import Config, StorageConfig, RAGConfig


@pytest.fixture
def temp_config(tmp_path):
    config = Config(
        storage=StorageConfig(path=tmp_path / "data"),
        rag=RAGConfig(chunk_size=100),
    )
    return config


@pytest.fixture
def memory_store(temp_config, monkeypatch):
    monkeypatch.setattr("localmind.config.Config.load", lambda x=None: temp_config)
    return MemoryStore(temp_config)


def test_add_memory(memory_store):
    entry_id = memory_store.add("Test memory content")
    assert entry_id is not None
    assert len(entry_id) == 16


def test_search_memory(memory_store):
    memory_store.add("Python is a great programming language")
    results = memory_store.search("Python programming")
    assert len(results) > 0
    assert "Python" in results[0]["content"]


def test_get_memory(memory_store):
    entry_id = memory_store.add("Get test content")
    retrieved = memory_store.get(entry_id)
    assert retrieved is not None
    assert retrieved["content"] == "Get test content"


def test_delete_memory(memory_store):
    entry_id = memory_store.add("Delete test")
    deleted = memory_store.delete(entry_id)
    assert deleted is True
    assert memory_store.get(entry_id) is None


def test_list_all(memory_store):
    memory_store.add("Content 1")
    memory_store.add("Content 2")
    results = memory_store.list_all()
    assert len(results) >= 2


def test_clear_all(memory_store):
    memory_store.add("To be cleared")
    count = memory_store.clear()
    assert count >= 1


def test_project_filter(memory_store):
    memory_store.add("Project A memory", project="project-a")
    memory_store.add("Project B memory", project="project-b")
    results = memory_store.search("memory", project="project-a")
    assert all(r["metadata"].get("project") == "project-a" for r in results)


def test_get_stats(memory_store):
    memory_store.add("Stats test")
    stats = memory_store.get_stats()
    assert "total_memories" in stats
    assert stats["total_memories"] >= 1