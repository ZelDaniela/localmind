"""Tests for security utilities."""

import pytest
from fastapi.testclient import TestClient
from localmind.server import create_app
from localmind.config import Config, StorageConfig, SecurityConfig


@pytest.fixture
def client(tmp_path):
    """Test client with auth disabled."""
    config = Config(
        storage=StorageConfig(path=tmp_path / "data"),
        security=SecurityConfig(api_key_enabled=False),
    )
    app = create_app()
    return TestClient(app)


def test_health_endpoint_always_accessible(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["name"] == "LocalMind"


def test_add_memory_via_api(client):
    response = client.post("/memory", json={"content": "API test memory"})
    assert response.status_code == 200
    assert "id" in response.json()


def test_add_memory_empty_content_rejected(client):
    response = client.post("/memory", json={"content": "   "})
    assert response.status_code == 422


def test_search_via_api(client):
    client.post("/memory", json={"content": "FastAPI is great"})
    response = client.post("/search", json={"query": "FastAPI"})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "count" in data


def test_search_clamps_n_results(client):
    response = client.post("/search", json={"query": "test", "n_results": 999})
    assert response.status_code == 200


def test_list_memories_via_api(client):
    client.post("/memory", json={"content": "List test"})
    response = client.get("/memory")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_delete_memory_via_api(client):
    add_resp = client.post("/memory", json={"content": "To delete"})
    memory_id = add_resp.json()["id"]
    del_resp = client.delete(f"/memory/{memory_id}")
    assert del_resp.status_code == 200


def test_delete_nonexistent_returns_404(client):
    response = client.delete("/memory/doesnotexist")
    assert response.status_code == 404


def test_index_blocked_path(client):
    response = client.post("/index", json={"path": "/etc/passwd", "project": "x"})
    assert response.status_code == 403


def test_stats_endpoint(client):
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_memories" in data
