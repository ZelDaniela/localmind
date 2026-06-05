"""Tests for MemoryStore."""
import pytest
from localmind.memory import MemoryStore
from localmind.config import Config, StorageConfig, RAGConfig


@pytest.fixture
def memory(tmp_path):
    config = Config(storage=StorageConfig(path=tmp_path / "data"), rag=RAGConfig(chunk_size=100))
    return MemoryStore(config)


def test_add_returns_id(memory):
    assert len(memory.add("Test content")) == 16


def test_same_content_same_id(memory):
    assert memory.add("Same") == memory.add("Same")


def test_search_relevant(memory):
    memory.add("Python is great for AI")
    results = memory.search("Python AI")
    assert len(results) > 0
    assert "Python" in results[0]["content"]


def test_search_empty_query(memory):
    assert memory.search("") == []


def test_search_project_filter(memory):
    memory.add("A memory", project="proj-a")
    memory.add("B memory", project="proj-b")
    results = memory.search("memory", project="proj-a")
    assert all(r["metadata"].get("project") == "proj-a" for r in results)


def test_get_existing(memory):
    id_ = memory.add("Get test")
    result = memory.get(id_)
    assert result is not None
    assert result["content"] == "Get test"


def test_get_nonexistent(memory):
    assert memory.get("doesnotexist00") is None


def test_delete_existing(memory):
    id_ = memory.add("Delete me")
    assert memory.delete(id_) is True
    assert memory.get(id_) is None


def test_delete_nonexistent(memory):
    assert memory.delete("doesnotexist00") is False


def test_list_all(memory):
    memory.add("A")
    memory.add("B")
    assert len(memory.list_all()) >= 2


def test_list_limit(memory):
    for i in range(5):
        memory.add(f"Entry {i}")
    assert len(memory.list_all(limit=2)) <= 2


def test_clear_all(memory):
    memory.add("X")
    memory.clear()
    assert memory.collection.count() == 0


def test_clear_by_project(memory):
    memory.add("Keep", project="keep")
    memory.add("Clear", project="clear-me")
    memory.clear(project="clear-me")
    assert len(memory.search("Keep", project="keep")) > 0


def test_stats_fields(memory):
    memory.add("Stats")
    s = memory.get_stats()
    assert "total_memories" in s
    assert s["total_memories"] >= 1
    assert "storage_path" in s


def test_export_import(memory, tmp_path):
    memory.add("Export this", project="exp")
    out = tmp_path / "backup.json"
    assert memory.export_json(out, project="exp") >= 1
    assert out.exists()

    config2 = Config(storage=StorageConfig(path=tmp_path / "new_data"))
    store2 = MemoryStore(config2)
    assert store2.import_json(out) >= 1


def test_add_empty_raises(memory):
    with pytest.raises(ValueError):
        memory.add("   ")


def test_metadata_stored(memory):
    memory.add("Meta", metadata={"tag": "work"})
    results = memory.search("Meta")
    assert results[0]["metadata"].get("tag") == "work"
