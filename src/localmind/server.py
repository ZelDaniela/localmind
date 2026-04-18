from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from localmind import __version__
from localmind.agents import AgentRegistry
from localmind.config import Config
from localmind.memory import MemoryStore
from localmind.rag import RAGPipeline

app = FastAPI(
    title="LocalMind API",
    description="Persistent memory API for local AI agents",
    version=__version__,
)


class AddMemoryRequest(BaseModel):
    content: str
    metadata: Optional[dict[str, Any]] = None
    project: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5
    project: Optional[str] = None


class IndexRequest(BaseModel):
    path: str
    project: str
    extensions: Optional[list[str]] = None


class ChatRequest(BaseModel):
    message: str
    agent: str = "ollama"
    use_rag: bool = True
    project: Optional[str] = None


def get_memory() -> MemoryStore:
    return MemoryStore()


def get_rag() -> RAGPipeline:
    return RAGPipeline(get_memory())


def get_registry() -> AgentRegistry:
    config = Config.load()
    return AgentRegistry(get_memory(), config)


@app.get("/")
def root():
    return {"name": "LocalMind", "version": __version__, "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/stats")
def stats():
    memory = get_memory()
    return memory.get_stats()


@app.post("/memory")
def add_memory(request: AddMemoryRequest):
    memory = get_memory()
    entry_id = memory.add(request.content, request.metadata, request.project)
    return {"id": entry_id, "status": "added"}


@app.get("/memory/{memory_id}")
def get_memory_by_id(memory_id: str):
    memory = get_memory()
    result = memory.get(memory_id)
    if not result:
        raise HTTPException(status_code=404, detail="Memory not found")
    return result


@app.delete("/memory/{memory_id}")
def delete_memory(memory_id: str):
    memory = get_memory()
    deleted = memory.delete(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted"}


@app.get("/memory")
def list_memories(
    project: Optional[str] = Query(None),
    limit: int = Query(100),
):
    memory = get_memory()
    return memory.list_all(limit=limit, project=project)


@app.post("/search")
def search(request: SearchRequest):
    memory = get_memory()
    results = memory.search(request.query, request.n_results, request.project)
    return {"results": results, "count": len(results)}


@app.post("/index")
def index(request: IndexRequest):
    from pathlib import Path

    memory = get_memory()
    rag = get_rag()

    path = Path(request.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    if path.is_file():
        result = rag.index_file(path, request.project)
    else:
        result = rag.index_directory(path, request.project, request.extensions)

    return result


@app.post("/chat")
def chat(request: ChatRequest):
    registry = get_registry()

    result = registry.chat_with_memory(
        agent=request.agent,
        message=request.message,
        use_rag=request.use_rag,
        project=request.project,
    )

    return result


@app.delete("/clear")
def clear_memories(project: Optional[str] = Query(None)):
    memory = get_memory()
    count = memory.clear(project=project)
    return {"deleted": count}


@app.get("/agents")
def list_agents():
    registry = get_registry()
    return {
        "ollama": {
            "available": registry.ollama.is_available(),
            "models": registry.ollama.list_models(),
        },
        "claude": {
            "available": registry.claude.is_available(),
        },
    }


def create_app() -> FastAPI:
    return app