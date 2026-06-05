"""Tests for RAGPipeline."""
import pytest
from pathlib import Path
from localmind.rag import RAGPipeline
from localmind.memory import MemoryStore
from localmind.config import Config, StorageConfig, RAGConfig


@pytest.fixture
def rag(tmp_path):
    config = Config(storage=StorageConfig(path=tmp_path / "data"), rag=RAGConfig(chunk_size=50, chunk_overlap=5))
    return RAGPipeline(MemoryStore(config))


def test_chunk_basic(rag):
    chunks = rag._chunk_text("Hello\nWorld\nFoo")
    assert len(chunks) > 0


def test_chunk_empty(rag):
    assert rag._chunk_text("") == []
    assert rag._chunk_text("   ") == []


def test_index_file(rag, tmp_path):
    f = tmp_path / "app.py"
    f.write_text("def hello(): pass")
    result = rag.index_file(f, "proj")
    assert result["indexed"] > 0


def test_index_directory(rag, tmp_path):
    (tmp_path / "app.py").write_text("def hello(): pass")
    (tmp_path / "README.md").write_text("# Docs")
    result = rag.index_directory(tmp_path, "proj")
    assert result["indexed"] > 0
    assert result["errors"] == []


def test_index_excludes_pycache(rag, tmp_path):
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "cached.py").write_text("cached")
    (tmp_path / "real.py").write_text("real code")
    result = rag.index_directory(tmp_path, "proj")
    assert result["skipped"] >= 1


def test_index_file_not_found(rag):
    with pytest.raises(ValueError, match="does not exist"):
        rag.index_file(Path("/nonexistent/file.py"), "proj")


def test_index_directory_not_found(rag):
    with pytest.raises(ValueError, match="does not exist"):
        rag.index_directory(Path("/nonexistent/dir"), "proj")


def test_safety_blocks_etc(rag):
    with pytest.raises(ValueError, match="not permitted"):
        rag.index_directory(Path("/etc"), "proj")


def test_safety_blocks_file_in_etc(rag):
    with pytest.raises(ValueError, match="not permitted"):
        rag.index_file(Path("/etc/passwd"), "proj")


def test_context_retrieval(rag, tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def main(): run_app()")
    rag.index_file(f, "proj")
    ctx = rag.get_relevant_context("function main", project="proj")
    assert len(ctx) > 0


def test_context_empty_query(rag):
    assert rag.get_relevant_context("") == ""
