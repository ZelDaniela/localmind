# LocalMind 🧠

> Persistent memory system for local AI agents. Your AI remembers everything, offline.

[![CI](https://github.com/ZelDaniela/localmind/actions/workflows/ci.yml/badge.svg)](https://github.com/ZelDaniela/localmind/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/ZelDaniela/localmind)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple.svg)](https://modelcontextprotocol.io)

**LocalMind** gives your local AI agents (Claude, Ollama, GPT4All, etc.) persistent memory across sessions. No cloud APIs, no data leaves your machine.

## Features

- 🔒 **Privacy-first**: Everything runs locally — no telemetry, no cloud
- 💾 **Persistent memory**: Remembers conversations, context, and preferences across sessions
- 📚 **RAG integration**: Index codebases, documents, and knowledge bases
- 🔗 **MCP server**: Native Model Context Protocol support for Claude Code
- 🤖 **Multi-agent support**: Works with Claude Code, Ollama, GPT4All, llama.cpp
- 🔑 **Optional API key auth**: Protect the local server when needed
- 📤 **Export / import**: Backup and migrate memories as JSON
- ⚡ **Fast local inference**: Optimized for edge deployment

## Why LocalMind?

| Feature      | LocalMind             | Cloud solutions     |
| ------------ | --------------------- | ------------------- |
| Privacy      | 100% local            | Data leaves device  |
| Cost         | Free (once installed) | API costs           |
| Offline      | ✅ Works offline       | ❌ Requires internet |
| Memory       | Persistent sessions   | Context limits      |
| MCP support  | ✅ Native              | Varies              |
| Customizable | Full source access    | Limited control     |

## Quick start

```bash
pip install localmind
localmind init
localmind add "User prefers concise answers"
localmind search "What are the user preferences?"
localmind serve        # REST API on :8000
localmind mcp          # MCP server (stdio, for Claude Code)
```

## Claude Code integration (MCP)

LocalMind exposes all its tools natively via the **Model Context Protocol**, so Claude Code can read and write memories transparently — no manual API calls needed.

### 1. Install LocalMind

```bash
pip install localmind
localmind init
```

### 2. Register with Claude Code

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "localmind": {
      "command": "localmind",
      "args": ["mcp"]
    }
  }
}
```

### 3. That's it

Claude Code now has access to these tools automatically:

| Tool | Description |
|------|-------------|
| `memory_add` | Store any information to remember across sessions |
| `memory_search` | Semantic search over all stored memories |
| `memory_list` | List all memories, optionally filtered by project |
| `memory_delete` | Delete a specific memory by ID |
| `memory_stats` | Show storage statistics |
| `index_directory` | Index a codebase or docs folder for RAG |

### Example interaction

```
You: Remember that this project uses PostgreSQL and the team prefers async patterns.

Claude: [calls memory_add] ✓ Stored.

--- next session ---

You: What database does this project use?

Claude: [calls memory_search "database project"] → PostgreSQL.
        The team also prefers async patterns.
```

## CLI reference

```bash
localmind init                          # Initialize config & storage
localmind add "content" [-p project]   # Add a memory
localmind search "query" [-p project]  # Semantic search
localmind list [-p project]            # List all memories
localmind delete <id>                  # Delete a memory
localmind clear [-p project] [-f]      # Clear memories
localmind stats                        # Storage statistics
localmind index <path> -p <project>    # Index files for RAG
localmind export output.json           # Export memories to JSON
localmind import backup.json           # Import memories from JSON
localmind keygen                       # Generate an API key
localmind serve [--host] [--port]      # REST API server (:8000)
localmind mcp                          # MCP server (stdio)
localmind mcp --http [--port 8001]     # MCP server (HTTP/SSE)
localmind version                      # Show version
```

## REST API

```bash
localmind serve   # http://127.0.0.1:8000 — docs at /docs
```

```bash
curl -X POST http://localhost:8000/memory \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello", "project": "myapp"}'

curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "framework", "n_results": 5}'

curl "http://localhost:8000/memory?project=myapp&limit=20"
curl "http://localhost:8000/stats"
```

## Enabling API key auth

```bash
localmind keygen   # prints a key

# Add to ~/.localmind/config.yaml:
# security:
#   api_key_enabled: true
#   api_key: <your-key>

curl -H "X-API-Key: <your-key>" http://localhost:8000/memory
```

## Python API

```python
from localmind import MemoryStore

memory = MemoryStore()
memory.add("Project uses FastAPI", metadata={"project": "myapp"})

results = memory.search("What framework is used?")
print(results[0]["content"])

memory.export_json(Path("backup.json"))
memory.import_json(Path("backup.json"))
```

## Configuration

`~/.localmind/config.yaml`:

```yaml
storage:
  path: ~/.localmind/data
  vector_db: chroma

rag:
  embeddings: sentence-transformers/all-MiniLM-L6-v2
  chunk_size: 512

security:
  api_key_enabled: false
  rate_limit_enabled: true
  rate_limit_per_minute: 60

agents:
  ollama:
    base_url: http://localhost:11434
    model: llama2
  claude:
    enabled: true
```

## Architecture

```
LocalMind
├── Memory Engine    → SQLite (structured) + ChromaDB (vectors)
├── RAG Pipeline     → Retrieval-Augmented Generation
├── MCP Server       → stdio + HTTP/SSE (Claude Code native)
├── Security Layer   → API key auth + path validation
├── Agent Connectors → Ollama, Claude Code adapters
└── REST API         → FastAPI endpoints
```

## Development

```bash
git clone https://github.com/ZelDaniela/localmind.git
cd localmind
pip install -e ".[dev]"
pytest tests/ -v --cov=src/localmind
ruff check src/
mypy src/localmind/ --ignore-missing-imports
```

## MCP server (Claude Code integration)

LocalMind implements the [Model Context Protocol](https://modelcontextprotocol.io) so Claude Code and other MCP-compatible agents can use persistent memory **transparently**, without any manual API calls.

### Quick setup for Claude Code

**1.** Install and initialize:
```bash
pip install localmind
localmind init
```

**2.** Add to `~/.claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "localmind": {
      "command": "localmind",
      "args": ["mcp"]
    }
  }
}
```

**3.** Restart Claude Code — LocalMind appears as tools automatically.

### MCP tools exposed to the agent

| Tool | Description |
|------|-------------|
| `memory_add` | Store a memory with optional project and metadata |
| `memory_search` | Semantic search across stored memories |
| `memory_list` | List all memories, optionally filtered by project |
| `memory_delete` | Delete a memory by ID |
| `memory_stats` | Storage statistics (count, disk usage, model) |
| `index_directory` | Index a codebase or directory for RAG |

### SSE mode (for web-based agents)

```bash
localmind mcp --sse --port 8001
# Endpoint: http://127.0.0.1:8001/mcp
```

### Test manually

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | localmind mcp
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | localmind mcp
```

## Roadmap

- [x] Core memory storage (SQLite + ChromaDB)
- [x] Vector similarity search
- [x] REST API server
- [x] Multi-project isolation
- [x] API key authentication
- [x] Export / import memories
- [x] GitHub Actions CI
- [x] MCP server (stdio + HTTP/SSE)
- [x] Claude Code integration
- [ ] WebUI dashboard
- [ ] Memory summarization agent

## License

MIT — see [LICENSE](LICENSE) for details.

---

⭐ Star if this project helps you!
