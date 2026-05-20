"""Tests for the MCP server."""

import json
import pytest
from localmind.mcp_server import MCPHandler
from localmind.config import Config, StorageConfig


@pytest.fixture
def handler(tmp_path):
    config = Config(storage=StorageConfig(path=tmp_path / "data"))
    h = MCPHandler.__new__(MCPHandler)
    from localmind.memory import MemoryStore
    from localmind.rag import RAGPipeline
    h.memory = MemoryStore(config)
    h.rag = RAGPipeline(h.memory)
    return h


# ── Protocol ─────────────────────────────────────────────────────────────────

def test_initialize(handler):
    response = handler.handle({
        "jsonrpc": "2.0", "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "clientInfo": {}}
    })
    assert response["result"]["protocolVersion"] == "2024-11-05"
    assert response["result"]["serverInfo"]["name"] == "localmind"


def test_tools_list(handler):
    response = handler.handle({
        "jsonrpc": "2.0", "id": 2,
        "method": "tools/list", "params": {}
    })
    tools = response["result"]["tools"]
    names = [t["name"] for t in tools]
    assert "memory_add" in names
    assert "memory_search" in names
    assert "memory_list" in names
    assert "memory_delete" in names
    assert "memory_stats" in names
    assert "index_directory" in names


def test_ping(handler):
    response = handler.handle({
        "jsonrpc": "2.0", "id": 3,
        "method": "ping", "params": {}
    })
    assert response["result"] == {}


def test_unknown_method_returns_error(handler):
    response = handler.handle({
        "jsonrpc": "2.0", "id": 4,
        "method": "nonexistent/method", "params": {}
    })
    assert "error" in response


def test_notification_returns_none(handler):
    """Notifications (no id) should not produce a response."""
    response = handler.handle({
        "jsonrpc": "2.0",
        "method": "notifications/initialized", "params": {}
    })
    assert response is None


# ── Tools ─────────────────────────────────────────────────────────────────────

def call_tool(handler, name, arguments):
    response = handler.handle({
        "jsonrpc": "2.0", "id": 10,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments}
    })
    assert "result" in response
    text = response["result"]["content"][0]["text"]
    return json.loads(text)


def test_memory_add(handler):
    result = call_tool(handler, "memory_add", {"content": "MCP test memory"})
    assert result["success"] is True
    assert "id" in result


def test_memory_add_with_project(handler):
    result = call_tool(handler, "memory_add", {
        "content": "Project-scoped memory",
        "project": "test-proj"
    })
    assert result["success"] is True


def test_memory_search(handler):
    call_tool(handler, "memory_add", {"content": "Python is great for AI"})
    result = call_tool(handler, "memory_search", {"query": "Python AI"})
    assert result["count"] > 0
    assert "content" in result["results"][0]
    assert "score" in result["results"][0]


def test_memory_search_n_results_clamped(handler):
    for i in range(5):
        call_tool(handler, "memory_add", {"content": f"Entry {i}"})
    result = call_tool(handler, "memory_search", {"query": "Entry", "n_results": 999})
    assert result["count"] <= 20


def test_memory_list(handler):
    call_tool(handler, "memory_add", {"content": "List test entry"})
    result = call_tool(handler, "memory_list", {})
    assert result["count"] >= 1
    assert "id" in result["memories"][0]


def test_memory_list_with_project(handler):
    call_tool(handler, "memory_add", {"content": "Proj A", "project": "proj-a"})
    call_tool(handler, "memory_add", {"content": "Proj B", "project": "proj-b"})
    result = call_tool(handler, "memory_list", {"project": "proj-a"})
    assert result["count"] >= 1


def test_memory_delete(handler):
    add_result = call_tool(handler, "memory_add", {"content": "To delete via MCP"})
    mem_id = add_result["id"]
    del_result = call_tool(handler, "memory_delete", {"id": mem_id})
    assert del_result["success"] is True


def test_memory_delete_nonexistent(handler):
    result = call_tool(handler, "memory_delete", {"id": "doesnotexist"})
    assert result["success"] is False


def test_memory_stats(handler):
    call_tool(handler, "memory_add", {"content": "Stats test"})
    result = call_tool(handler, "memory_stats", {})
    assert "total_memories" in result
    assert result["total_memories"] >= 1


def test_index_directory(handler, tmp_path):
    (tmp_path / "app.py").write_text("def hello(): pass")
    result = call_tool(handler, "index_directory", {
        "path": str(tmp_path),
        "project": "mcp-test"
    })
    assert result["success"] is True
    assert result["indexed"] > 0


def test_index_nonexistent_path(handler):
    result = call_tool(handler, "index_directory", {
        "path": "/nonexistent/path/xyz",
        "project": "x"
    })
    assert result["success"] is False


def test_unknown_tool_returns_error(handler):
    response = handler.handle({
        "jsonrpc": "2.0", "id": 99,
        "method": "tools/call",
        "params": {"name": "nonexistent_tool", "arguments": {}}
    })
    assert "error" in response
