# LocalMind 🧠

> Persistent memory system for local AI agents. Your AI remembers everything, offline.

[![CI](https://github.com/ZelDaniela/localmind/actions/workflows/ci.yml/badge.svg)](https://github.com/ZelDaniela/localmind/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/ZelDaniela/localmind)

**LocalMind** gives your local AI agents (Claude, Ollama, GPT4All, etc.) persistent memory across sessions. No cloud APIs, no data leaves your machine.

## Features

- 🔒 **Privacy-first**: Everything runs locally — no telemetry, no cloud
- 💾 **Persistent memory**: Remembers conversations, context, and preferences across sessions
- 📚 **RAG integration**: Index codebases, documents, and knowledge bases
- 🔗 **Multi-agent support**: Works with Claude Code, Ollama, GPT4All, llama.cpp
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
| Customizable | Full source access    | Limited control     |

## Quick start

```bash
# Install
pip install localmind

# Initialize (creates ~/.localmind/)
localmind init

# Add a memory
localmind add "User prefers Python over JavaScript"

# Search memories
localmind search "What does the user prefer?"

# Start REST API server (localhost only by default)
localmind serve
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
localmind serve [--host] [--port]      # Start API server
localmind version                      # Show version
```

## API server

```bash
# Start server (default: http://127.0.0.1:8000)
localmind serve

# Docs available at: http://127.0.0.1:8000/docs
```

```bash
# Add memory
curl -X POST http://localhost:8000/memory \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello", "metadata": {}, "project": "myapp"}'

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "framework", "n_results": 5}'

# List memories
curl "http://localhost:8000/memory?project=myapp&limit=20"

# Export
curl -X POST http://localhost:8000/export \
  -H "Content-Type: application/json" \
  -d '{"output_path": "/tmp/backup.json"}'
```

## Enabling API key authentication

```bash
# 1. Generate a key
localmind keygen

# 2. Add to ~/.localmind/config.yaml:
#    security:
#      api_key_enabled: true
#      api_key: <your-key>

# 3. Use the key in requests:
curl -H "X-API-Key: <your-key>" http://localhost:8000/memory
```

## Python API

```python
from localmind import MemoryStore

memory = MemoryStore()

# Store a memory
memory.add("Project uses FastAPI", metadata={"project": "myapp"})

# Retrieve relevant context
results = memory.search("What framework is used?")
print(results[0]["content"])

# Export / import
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
  api_key_enabled: false   # set true + api_key to enable auth
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
├── Security Layer   → API key auth + path validation
├── Agent Connectors → Ollama, Claude Code adapters
└── API Server       → FastAPI REST endpoints
```

## Supported models

- **Ollama** — `ollama run llama2`
- **llama.cpp** — Local inference with quantised models
- **GPT4All** — Consumer-grade local LLMs
- **Claude Code** — Via CLI integration

## Development

```bash
git clone https://github.com/ZelDaniela/localmind.git
cd localmind

pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=src/localmind

# Lint
ruff check src/

# Type check
mypy src/localmind/ --ignore-missing-imports
```

## Roadmap

- [x] Core memory storage (SQLite + ChromaDB)
- [x] Vector similarity search
- [x] REST API server
- [x] Multi-project isolation
- [x] API key authentication
- [x] Export / import memories
- [x] GitHub Actions CI
- [ ] MCP server integration (Claude Code)
- [ ] WebUI dashboard
- [ ] Memory summarization agent

## License

MIT — see [LICENSE](LICENSE) for details.

---

⭐ Star if this project helps you!
