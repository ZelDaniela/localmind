"""Tests for the MCP server."""
import json
import pytest
from localmind.mcp_server import MCPHandler
from localmind.config import Config, StorageConfig
from localmind.memory import MemoryStore
from localmind.rag import RAGPipeline


@pytest.fixture
def handler(tmp_path):
    config = Config(storage=StorageConfig(path=tmp_path / "data"))
    h = MCPHandler.__new__(MCPHandler)
    h.memory = MemoryStore(config)
    h.rag = RAGPipeline(h.memory)
    return h


def call_tool(handler, name, arguments):
    resp = handler.handle({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })
    assert "result" in resp
    return json.loads(resp["result"]["content"][0]["text"])


def test_initialize(handler):
    resp = handler.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "localmind"


def test_tools_list(handler):
    resp = handler.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    names = [t["name"] for t in resp["result"]["tools"]]
    assert set(names) == {"memory_add", "memory_search", "memory_list", "memory_delete", "memory_stats", "index_directory"}


def test_ping(handler):
    resp = handler.handle({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}})
    assert resp["result"] == {}


def test_notification_no_response(handler):
    assert handler.handle({"jsonrpc": "2.0", "method": "notifications/init", "params": {}}) is None


def test_unknown_method_error(handler):
    resp = handler.handle({"jsonrpc": "2.0", "id": 1, "method": "bad/method", "params": {}})
    assert "error" in resp


def test_memory_add(handler):
    result = call_tool(handler, "memory_add", {"content": "MCP test"})
    assert result["success"] is True
    assert "id" in result


def test_memory_search(handler):
    call_tool(handler, "memory_add", {"content": "Python is awesome"})
    result = call_tool(handler, "memory_search", {"query": "Python"})
    assert result["count"] > 0
    assert "score" in result["results"][0]


def test_memory_list(handler):
    call_tool(handler, "memory_add", {"content": "List test"})
    result = call_tool(handler, "memory_list", {})
    assert result["count"] >= 1


def test_memory_delete(handler):
    id_ = call_tool(handler, "memory_add", {"content": "Delete me"})["id"]
    result = call_tool(handler, "memory_delete", {"id": id_})
    assert result["success"] is True


def test_memory_delete_nonexistent(handler):
    result = call_tool(handler, "memory_delete", {"id": "doesnotexist"})
    assert result["success"] is False


def test_memory_stats(handler):
    call_tool(handler, "memory_add", {"content": "Stats"})
    result = call_tool(handler, "memory_stats", {})
    assert result["total_memories"] >= 1


def test_index_directory(handler, tmp_path):
    (tmp_path / "app.py").write_text("def hello(): pass")
    result = call_tool(handler, "index_directory", {"path": str(tmp_path), "project": "test"})
    assert result["success"] is True
    assert result["indexed"] > 0


def test_index_nonexistent(handler):
    result = call_tool(handler, "index_directory", {"path": "/nonexistent/xyz", "project": "x"})
    assert result["success"] is False


def test_unknown_tool_error(handler):
    resp = handler.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": "fake_tool", "arguments": {}}})
    assert "error" in resp
