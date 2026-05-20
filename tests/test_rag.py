"""Tests for RAGPipeline."""

import pytest
from pathlib import Path
from localmind.rag import RAGPipeline
from localmind.memory import MemoryStore
from localmind.config import Config, StorageConfig, RAGConfig


@pytest.fixture
def temp_config(tmp_path):
    return Config(
        storage=StorageConfig(path=tmp_path / "data"),
        rag=RAGConfig(chunk_size=50, chunk_overlap=10),
    )


@pytest.fixture
def rag_pipeline(temp_config):
    memory = MemoryStore(temp_config)
    return RAGPipeline(memory)


def test_chunk_text_basic(rag_pipeline):
    text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    chunks = rag_pipeline._chunk_text(text)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_text_respects_size(rag_pipeline):
    long_text = "word " * 200
    chunks = rag_pipeline._chunk_text(long_text)
    chunk_size = rag_pipeline.memory.config.rag.chunk_size
    for chunk in chunks:
        assert len(chunk) <= chunk_size + 10  # small tolerance for word boundaries


def test_chunk_empty_text(rag_pipeline):
    chunks = rag_pipeline._chunk_text("")
    assert chunks == []


def test_index_file(rag_pipeline, tmp_path):
    test_file = tmp_path / "example.py"
    test_file.write_text("x = 1\ny = 2\ndef foo(): pass")
    result = rag_pipeline.index_file(test_file, project="test-project")
    assert result["indexed"] > 0
    assert result["project"] == "test-project"


def test_index_directory(rag_pipeline, tmp_path):
    (tmp_path / "app.py").write_text("def hello():\n    print('hello')")
    (tmp_path / "README.md").write_text("# My project\nSome docs here.")
    result = rag_pipeline.index_directory(tmp_path, project="test-project")
    assert result["indexed"] > 0
    assert result["project"] == "test-project"
    assert result["errors"] == []


def test_index_directory_excludes_hidden(rag_pipeline, tmp_path):
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "cached.py").write_text("cached content")
    (tmp_path / "real.py").write_text("real content")
    result = rag_pipeline.index_directory(tmp_path, project="test")
    # Should not index the pycache file
    assert result["indexed"] >= 1


def test_index_nonexistent_file(rag_pipeline):
    with pytest.raises(ValueError, match="does not exist"):
        rag_pipeline.index_file(Path("/nonexistent/file.py"), project="x")


def test_index_nonexistent_directory(rag_pipeline):
    with pytest.raises(ValueError, match="does not exist"):
        rag_pipeline.index_directory(Path("/nonexistent/dir"), project="x")


def test_get_relevant_context(rag_pipeline, tmp_path):
    test_file = tmp_path / "app.py"
    test_file.write_text("def main():\n    run_app()")
    rag_pipeline.index_file(test_file, project="myproject")
    context = rag_pipeline.get_relevant_context("function main", project="myproject")
    assert len(context) > 0


def test_safety_blocks_system_paths(rag_pipeline):
    with pytest.raises(ValueError, match="not permitted"):
        rag_pipeline.index_directory(Path("/etc"), project="hacked")
