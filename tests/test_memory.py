"""Tests for MemoryStore."""

import pytest
from localmind.memory import MemoryStore
from localmind.config import Config, StorageConfig, RAGConfig


@pytest.fixture
def temp_config(tmp_path):
    return Config(
        storage=StorageConfig(path=tmp_path / "data"),
        rag=RAGConfig(chunk_size=100),
    )


@pytest.fixture
def memory_store(temp_config):
    return MemoryStore(temp_config)


def test_add_returns_id(memory_store):
    entry_id = memory_store.add("Test memory content")
    assert entry_id is not None
    assert len(entry_id) == 16


def test_add_same_content_same_id(memory_store):
    """Same content should produce the same deterministic ID."""
    id1 = memory_store.add("Identical content")
    id2 = memory_store.add("Identical content")
    assert id1 == id2


def test_search_returns_relevant_result(memory_store):
    memory_store.add("Python is a great programming language")
    results = memory_store.search("Python programming")
    assert len(results) > 0
    assert "Python" in results[0]["content"]


def test_search_with_project_filter(memory_store):
    memory_store.add("Project A memory", project="project-a")
    memory_store.add("Project B memory", project="project-b")
    results = memory_store.search("memory", project="project-a")
    assert all(r["metadata"].get("project") == "project-a" for r in results)


def test_get_existing_memory(memory_store):
    entry_id = memory_store.add("Get test content")
    retrieved = memory_store.get(entry_id)
    assert retrieved is not None
    assert retrieved["content"] == "Get test content"


def test_get_nonexistent_returns_none(memory_store):
    result = memory_store.get("nonexistent000000")
    assert result is None


def test_delete_existing_memory(memory_store):
    entry_id = memory_store.add("Delete test")
    deleted = memory_store.delete(entry_id)
    assert deleted is True
    assert memory_store.get(entry_id) is None


def test_delete_nonexistent_returns_false(memory_store):
    deleted = memory_store.delete("doesnotexist000")
    assert deleted is False


def test_list_all_returns_entries(memory_store):
    memory_store.add("Content 1")
    memory_store.add("Content 2")
    results = memory_store.list_all()
    assert len(results) >= 2


def test_list_all_with_limit(memory_store):
    for i in range(10):
        memory_store.add(f"Entry {i}")
    results = memory_store.list_all(limit=3)
    assert len(results) <= 3


def test_clear_all(memory_store):
    memory_store.add("To be cleared 1")
    memory_store.add("To be cleared 2")
    memory_store.clear()
    assert memory_store.collection.count() == 0


def test_clear_by_project(memory_store):
    memory_store.add("Keep this", project="keep")
    memory_store.add("Clear this", project="clear-me")
    memory_store.clear(project="clear-me")
    kept = memory_store.search("Keep", project="keep")
    assert len(kept) > 0


def test_get_stats_fields(memory_store):
    memory_store.add("Stats test")
    s = memory_store.get_stats()
    assert "total_memories" in s
    assert "storage_path" in s
    assert "sqlite_size_kb" in s
    assert "chroma_size_kb" in s
    assert s["total_memories"] >= 1


def test_export_and_import(memory_store, tmp_path):
    memory_store.add("Exportable content", project="export-test")
    out_file = tmp_path / "backup.json"
    exported = memory_store.export_json(out_file, project="export-test")
    assert exported >= 1
    assert out_file.exists()

    # Import into a fresh store
    from localmind.config import StorageConfig
    new_config = Config(storage=StorageConfig(path=tmp_path / "new_data"))
    new_store = MemoryStore(new_config)
    imported = new_store.import_json(out_file)
    assert imported >= 1


def test_metadata_stored_correctly(memory_store):
    memory_store.add("Meta test", metadata={"tag": "work", "priority": "high"})
    results = memory_store.search("Meta test")
    assert len(results) > 0
    assert results[0]["metadata"].get("tag") == "work"
