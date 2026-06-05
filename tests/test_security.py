"""Tests for the REST API and security layer."""
import pytest
from fastapi.testclient import TestClient
from localmind.server import create_app
from localmind.config import Config, StorageConfig, SecurityConfig


@pytest.fixture
def client(tmp_path):
    config = Config(
        storage=StorageConfig(path=tmp_path / "data"),
        security=SecurityConfig(api_key_enabled=False),
    )
    return TestClient(create_app())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_root(client):
    r = client.get("/")
    assert r.json()["name"] == "LocalMind"


def test_add_memory(client):
    r = client.post("/memory", json={"content": "API test"})
    assert r.status_code == 200
    assert "id" in r.json()


def test_add_empty_rejected(client):
    r = client.post("/memory", json={"content": "  "})
    assert r.status_code == 422


def test_add_content_too_large(client):
    r = client.post("/memory", json={"content": "x" * (1024 * 1024 + 1)})
    assert r.status_code == 413


def test_search(client):
    client.post("/memory", json={"content": "FastAPI is great"})
    r = client.post("/search", json={"query": "FastAPI"})
    assert r.status_code == 200
    assert "results" in r.json()


def test_search_empty_query_rejected(client):
    r = client.post("/search", json={"query": "  "})
    assert r.status_code == 422


def test_search_clamps_n_results(client):
    r = client.post("/search", json={"query": "test", "n_results": 9999})
    assert r.status_code == 200


def test_list_memories(client):
    client.post("/memory", json={"content": "List test"})
    r = client.get("/memory")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_delete_memory(client):
    id_ = client.post("/memory", json={"content": "Delete me"}).json()["id"]
    r = client.delete(f"/memory/{id_}")
    assert r.status_code == 200


def test_delete_nonexistent_404(client):
    r = client.delete("/memory/doesnotexist")
    assert r.status_code == 404


def test_invalid_memory_id_rejected(client):
    r = client.get("/memory/../../etc/passwd")
    assert r.status_code in (400, 422)


def test_index_blocked_path(client):
    r = client.post("/index", json={"path": "/etc/passwd", "project": "x"})
    assert r.status_code == 403


def test_index_invalid_path(client):
    r = client.post("/index", json={"path": "", "project": "x"})
    assert r.status_code in (400, 422)


def test_invalid_project_name_rejected(client):
    r = client.post("/memory", json={"content": "test", "project": "../../evil"})
    assert r.status_code == 400


def test_stats_endpoint(client):
    r = client.get("/stats")
    assert r.status_code == 200
    assert "total_memories" in r.json()
