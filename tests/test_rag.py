import pytest
from pathlib import Path
import tempfile
from localmind.rag import RAGPipeline
from localmind.memory import MemoryStore
from localmind.config import Config, StorageConfig, RAGConfig


@pytest.fixture
def temp_config(tmp_path):
    config = Config(
        storage=StorageConfig(path=tmp_path / "data"),
        rag=RAGConfig(chunk_size=50, chunk_overlap=10),
    )
    return config


@pytest.fixture
def rag_pipeline(temp_config, monkeypatch):
    monkeypatch.setattr("localmind.config.Config.load", lambda x=None: temp_config)
    memory = MemoryStore(temp_config)
    return RAGPipeline(memory)


def test_chunk_text(rag_pipeline):
    text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    chunks = rag_pipeline._chunk_text(text)
    assert len(chunks) > 0


def test_index_directory(rag_pipeline, tmp_path):
    test_file = tmp_path / "test.py"
    test_file.write_text("def hello():\n    print('hello')")

    result = rag_pipeline.index_directory(tmp_path, project="test-project")
    assert result["indexed"] > 0
    assert result["project"] == "test-project"


def test_index_file(rag_pipeline, tmp_path):
    test_file = tmp_path / "example.py"
    test_file.write_text("x = 1\ny = 2")

    result = rag_pipeline.index_file(test_file, project="test-project")
    assert result["indexed"] > 0


def test_get_relevant_context(rag_pipeline, tmp_path):
    test_file = tmp_path / "app.py"
    test_file.write_text("def main():\n    run_app()")

    rag_pipeline.index_file(test_file, project="myproject")
    context = rag_pipeline.get_relevant_context("function main", project="myproject")
    assert len(context) > 0