# LocalMind 🧠

> Persistent memory system for local AI agents. Your AI remembers everything, offline.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-passing-green.svg)]()

**LocalMind** gives your local AI agents (Claude, Ollama, GPT4All, etc.) persistent memory across sessions. No cloud APIs, no data leaves your machine.

## Features

- 🔒 **Privacy-first**: Everything runs locally on your machine
- 💾 **Persistent Memory**: Remembers conversations, context, and preferences across sessions
- 📚 **RAG Integration**: Index your codebases, documents, and knowledge bases
- 🔗 **Multi-Agent Support**: Works with Claude Code, Ollama, GPT4All, llama.cpp
- ⚡ **Fast Local Inference**: Optimized for edge deployment

## Why LocalMind?

| Feature | LocalMind | Cloud Solutions |
|---------|-----------|-----------------|
| Privacy | 100% local | Data leaves device |
| Cost | Free (once installed) | API costs |
| Offline | ✅ Works offline | ❌ Requires internet |
| Memory | Persistent sessions | Context limits |
| Customizable | Full source access | Limited control |

## Quick Start

```bash
# Install
pip install localmind

# Initialize your memory store
localmind init

# Add context to memory
localmind add "User prefers Python over JavaScript"

# Query your memory
localmind search "What does the user prefer?"

# Start API server for AI agents
localmind serve
```

## Architecture

```
LocalMind
├── Memory Engine      → Persistent storage (SQLite + ChromaDB)
├── RAG Pipeline       → Retrieval-Augmented Generation
├── Agent Connectors   → Claude, Ollama, GPT4All adapters
└── API Server         → FastAPI REST endpoints
```

## Configuration

Create `~/.localmind/config.yaml`:

```yaml
storage:
  path: ~/.localmind/data
  vector_db: chroma

rag:
  embeddings: sentence-transformers/all-MiniLM-L6-v2
  chunk_size: 512

agents:
  ollama:
    base_url: http://localhost:11434
  claude:
    enabled: true
```

## API Usage

```python
from localmind import MemoryStore

memory = MemoryStore()

# Store a memory
memory.add("Project uses FastAPI", metadata={"project": "myapp"})

# Retrieve relevant context
context = memory.search("What framework is used?")
print(context)
```

REST API:

```bash
# Add memory
curl -X POST http://localhost:8000/memory -d '{"content": "Hello", "metadata": {}}'

# Search
curl "http://localhost:8000/search?q=framework"

# Get conversation history
curl "http://localhost:8000/history?limit=10"
```

## Supported Models

- **Ollama** - `ollama run llama2`
- **llama.cpp** - Local inference with quantised models
- **GPT4All** - Consumer-grade local LLMs
- **Claude Code** - Via CLI integration

## Development

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/localmind.git
cd localmind

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/
```

## Roadmap

- [x] Core memory storage (SQLite)
- [x] Vector similarity search (ChromaDB)
- [x] REST API server
- [ ] Multi-project isolation
- [ ] Claude Code MCP integration
- [ ] WebUI dashboard
- [ ] Memory summarization agent

## License

MIT License - See [LICENSE](LICENSE) for details.

---

⭐ Star if this project helps you!
