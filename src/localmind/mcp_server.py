"""
LocalMind MCP Server — Model Context Protocol integration.

Usage
-----
  localmind mcp          # stdio transport (Claude Code)
  localmind mcp --sse    # SSE transport on port 8001

Claude Code config (~/.claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "localmind": {
        "command": "localmind",
        "args": ["mcp"]
      }
    }
  }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from localmind.config import Config
from localmind.memory import MemoryStore
from localmind.rag import RAGPipeline

JSONRPC_VERSION = "2.0"
MCP_VERSION = "2024-11-05"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "memory_add",
        "description": (
            "Store a memory or piece of information for later retrieval. "
            "Use this to remember facts, decisions, user preferences, or any "
            "context that should persist across sessions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The content to remember."},
                "project": {
                    "type": "string",
                    "description": "Optional project/namespace (e.g. 'myapp').",
                },
                "metadata": {"type": "object", "description": "Optional key-value metadata."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_search",
        "description": (
            "Search memories using semantic similarity. "
            "Returns the most relevant stored memories for a given query. "
            "Use this to recall past context, decisions, or user preferences."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "n_results": {
                    "type": "integer",
                    "description": "Number of results (1-20, default 5).",
                    "default": 5,
                },
                "project": {"type": "string", "description": "Optional project/namespace filter."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_list",
        "description": "List stored memories, optionally filtered by project.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Optional project/namespace filter."},
                "limit": {
                    "type": "integer",
                    "description": "Max memories to return (default 20).",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "memory_delete",
        "description": "Delete a specific memory by its ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The memory ID to delete."},
            },
            "required": ["id"],
        },
    },
    {
        "name": "memory_stats",
        "description": "Get statistics about LocalMind storage (total memories, disk usage, model).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "index_directory",
        "description": (
            "Index a local directory or file so its content can be retrieved via memory_search. "
            "Useful for giving the agent context about a codebase or document collection."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file or directory.",
                },
                "project": {
                    "type": "string",
                    "description": "Project/namespace for indexed content.",
                },
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'File extensions to include (e.g. [".py", ".md"]).',
                },
            },
            "required": ["path", "project"],
        },
    },
]


class MCPHandler:
    def __init__(self) -> None:
        config = Config.load()
        self.memory = MemoryStore(config)
        self.rag = RAGPipeline(self.memory)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        dispatch = {
            "memory_add": self._memory_add,
            "memory_search": self._memory_search,
            "memory_list": self._memory_list,
            "memory_delete": self._memory_delete,
            "memory_stats": self._memory_stats,
            "index_directory": self._index_directory,
        }
        fn = dispatch.get(name)
        if fn is None:
            raise ValueError(f"Unknown tool: {name}")
        return fn(arguments)

    def _memory_add(self, args: dict[str, Any]) -> dict[str, Any]:
        entry_id = self.memory.add(
            content=args["content"],
            metadata=args.get("metadata"),
            project=args.get("project"),
        )
        return {"success": True, "id": entry_id, "message": f"Memory stored with ID {entry_id}."}

    def _memory_search(self, args: dict[str, Any]) -> dict[str, Any]:
        n = max(1, min(args.get("n_results", 5), 20))
        results = self.memory.search(
            query=args["query"],
            n_results=n,
            project=args.get("project"),
        )
        formatted = []
        for r in results:
            score = round(1 - r["distance"], 3) if r.get("distance") is not None else None
            formatted.append(
                {
                    "id": r["id"],
                    "content": r["content"],
                    "score": score,
                    "project": r["metadata"].get("project"),
                    "metadata": {
                        k: v for k, v in r["metadata"].items() if k not in ("project", "created_at")
                    },
                }
            )
        return {"results": formatted, "count": len(formatted)}

    def _memory_list(self, args: dict[str, Any]) -> dict[str, Any]:
        memories = self.memory.list_all(
            limit=args.get("limit", 20),
            project=args.get("project"),
        )
        return {
            "memories": [
                {
                    "id": m["id"],
                    "content": m["content"][:200] + ("…" if len(m["content"]) > 200 else ""),
                    "project": m["metadata"].get("project"),
                    "created_at": m["metadata"].get("created_at", "unknown"),
                }
                for m in memories
            ],
            "count": len(memories),
        }

    def _memory_delete(self, args: dict[str, Any]) -> dict[str, Any]:
        deleted = self.memory.delete(args["id"])
        return {
            "success": deleted,
            "message": f"Memory {args['id']} deleted."
            if deleted
            else f"Memory {args['id']} not found.",
        }

    def _memory_stats(self, _args: dict[str, Any]) -> dict[str, Any]:
        return self.memory.get_stats()

    def _index_directory(self, args: dict[str, Any]) -> dict[str, Any]:
        path = Path(args["path"])
        project = args["project"]
        extensions = args.get("extensions")

        try:
            if path.is_file():
                result = self.rag.index_file(path, project)
            elif path.is_dir():
                result = self.rag.index_directory(path, project, extensions)
            else:
                return {"success": False, "message": f"Path not found: {path}"}
        except ValueError as e:
            return {"success": False, "message": str(e)}

        return {
            "success": True,
            "indexed": result["indexed"],
            "project": project,
            "errors": result.get("errors", []),
        }

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method", "")
        msg_id = message.get("id")

        if msg_id is None and method not in ("initialize",):
            return None

        try:
            result = self._dispatch(method, message.get("params", {}))
            return {"jsonrpc": JSONRPC_VERSION, "id": msg_id, "result": result}
        except Exception as exc:
            return {
                "jsonrpc": JSONRPC_VERSION,
                "id": msg_id,
                "error": {"code": -32603, "message": str(exc)},
            }

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": MCP_VERSION,
                "serverInfo": {"name": "localmind", "version": "0.2.0"},
                "capabilities": {"tools": {}},
            }
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            tool_result = self.call_tool(name, arguments)
            return {
                "content": [
                    {"type": "text", "text": json.dumps(tool_result, indent=2, ensure_ascii=False)}
                ],
                "isError": False,
            }
        if method == "ping":
            return {}
        raise ValueError(f"Method not found: {method}")


def run_stdio() -> None:
    handler = MCPHandler()
    print("LocalMind MCP server started (stdio)", file=sys.stderr, flush=True)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            message = json.loads(raw_line)
        except json.JSONDecodeError as e:
            error_resp = {
                "jsonrpc": JSONRPC_VERSION,
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            print(json.dumps(error_resp), flush=True)
            continue

        response = handler.handle(message)
        if response is not None:
            print(json.dumps(response), flush=True)


def run_sse(host: str = "127.0.0.1", port: int = 8001) -> None:
    try:
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError:
        print("fastapi and uvicorn required for SSE mode.", file=sys.stderr)
        sys.exit(1)

    handler = MCPHandler()
    sse_app = FastAPI(title="LocalMind MCP SSE")

    @sse_app.post("/mcp")
    async def mcp_endpoint(request: Request) -> JSONResponse:
        body = await request.json()
        response = handler.handle(body)
        return JSONResponse(content=response or {})

    @sse_app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "server": "localmind-mcp"})

    print(f"LocalMind MCP SSE server on http://{host}:{port}/mcp", file=sys.stderr)
    uvicorn.run(sse_app, host=host, port=port)
