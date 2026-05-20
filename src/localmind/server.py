"""FastAPI REST server for LocalMind."""

from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from localmind import __version__
from localmind.agents import AgentRegistry
from localmind.config import Config
from localmind.memory import MemoryStore
from localmind.rag import RAGPipeline
from localmind.security import get_api_key, validate_path_safety

# ── Rate limiting (optional dep) ────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    RATE_LIMITING = True
except ImportError:
    limiter = None  # type: ignore[assignment]
    RATE_LIMITING = False


def create_app() -> FastAPI:
    config = Config.load()

    app = FastAPI(
        title="LocalMind API",
        description=(
            "Persistent memory API for local AI agents. "
            "All data stays on your machine."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — localhost only by default
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://127.0.0.1"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if RATE_LIMITING:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # ── Request models ───────────────────────────────────────────────────────

    class AddMemoryRequest(BaseModel):
        content: str
        metadata: Optional[dict[str, Any]] = None
        project: Optional[str] = None

        @field_validator("content")
        @classmethod
        def content_not_empty(cls, v: str) -> str:
            if not v.strip():
                raise ValueError("content cannot be empty")
            return v.strip()

    class SearchRequest(BaseModel):
        query: str
        n_results: int = 5
        project: Optional[str] = None

        @field_validator("n_results")
        @classmethod
        def clamp_results(cls, v: int) -> int:
            return max(1, min(v, 50))

    class IndexRequest(BaseModel):
        path: str
        project: str
        extensions: Optional[list[str]] = None

    class ChatRequest(BaseModel):
        message: str
        agent: str = "ollama"
        use_rag: bool = True
        project: Optional[str] = None

    class ExportRequest(BaseModel):
        output_path: str
        project: Optional[str] = None

    # ── Dependencies ─────────────────────────────────────────────────────────

    def get_memory() -> MemoryStore:
        return MemoryStore(config)

    def get_rag(memory: MemoryStore = Depends(get_memory)) -> RAGPipeline:
        return RAGPipeline(memory)

    def get_registry(memory: MemoryStore = Depends(get_memory)) -> AgentRegistry:
        return AgentRegistry(memory, config)

    # ── Routes ───────────────────────────────────────────────────────────────

    @app.get("/", tags=["system"])
    def root() -> dict[str, str]:
        return {"name": "LocalMind", "version": __version__, "status": "running"}

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/stats", tags=["system"])
    def stats(
        memory: MemoryStore = Depends(get_memory),
        _: Optional[str] = Depends(get_api_key),
    ) -> dict[str, Any]:
        return memory.get_stats()

    @app.post("/memory", tags=["memory"])
    def add_memory(
        request: AddMemoryRequest,
        memory: MemoryStore = Depends(get_memory),
        _: Optional[str] = Depends(get_api_key),
    ) -> dict[str, str]:
        entry_id = memory.add(request.content, request.metadata, request.project)
        return {"id": entry_id, "status": "added"}

    @app.get("/memory", tags=["memory"])
    def list_memories(
        project: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        memory: MemoryStore = Depends(get_memory),
        _: Optional[str] = Depends(get_api_key),
    ) -> list[dict[str, Any]]:
        return memory.list_all(limit=limit, project=project)

    @app.get("/memory/{memory_id}", tags=["memory"])
    def get_memory_by_id(
        memory_id: str,
        memory: MemoryStore = Depends(get_memory),
        _: Optional[str] = Depends(get_api_key),
    ) -> dict[str, Any]:
        result = memory.get(memory_id)
        if not result:
            raise HTTPException(status_code=404, detail="Memory not found")
        return result

    @app.delete("/memory/{memory_id}", tags=["memory"])
    def delete_memory(
        memory_id: str,
        memory: MemoryStore = Depends(get_memory),
        _: Optional[str] = Depends(get_api_key),
    ) -> dict[str, str]:
        deleted = memory.delete(memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"status": "deleted"}

    @app.post("/search", tags=["memory"])
    def search(
        request: SearchRequest,
        req: Request,
        memory: MemoryStore = Depends(get_memory),
        _: Optional[str] = Depends(get_api_key),
    ) -> dict[str, Any]:
        results = memory.search(request.query, request.n_results, request.project)
        return {"results": results, "count": len(results)}

    @app.delete("/clear", tags=["memory"])
    def clear_memories(
        project: Optional[str] = Query(None),
        memory: MemoryStore = Depends(get_memory),
        _: Optional[str] = Depends(get_api_key),
    ) -> dict[str, Any]:
        count = memory.clear(project=project)
        return {"deleted": count}

    @app.post("/index", tags=["rag"])
    def index(
        request: IndexRequest,
        rag: RAGPipeline = Depends(get_rag),
        _: Optional[str] = Depends(get_api_key),
    ) -> dict[str, Any]:
        validate_path_safety(request.path)
        path = Path(request.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        if path.is_file():
            return rag.index_file(path, request.project)
        return rag.index_directory(path, request.project, request.extensions)

    @app.post("/chat", tags=["agents"])
    def chat(
        request: ChatRequest,
        registry: AgentRegistry = Depends(get_registry),
        _: Optional[str] = Depends(get_api_key),
    ) -> dict[str, Any]:
        return registry.chat_with_memory(
            agent=request.agent,
            message=request.message,
            use_rag=request.use_rag,
            project=request.project,
        )

    @app.get("/agents", tags=["agents"])
    def list_agents(
        registry: AgentRegistry = Depends(get_registry),
        _: Optional[str] = Depends(get_api_key),
    ) -> dict[str, Any]:
        return {
            "ollama": {
                "available": registry.ollama.is_available(),
                "models": registry.ollama.list_models(),
            },
            "claude": {
                "available": registry.claude.is_available(),
            },
        }

    @app.post("/export", tags=["utils"])
    def export_memories(
        request: ExportRequest,
        memory: MemoryStore = Depends(get_memory),
        _: Optional[str] = Depends(get_api_key),
    ) -> dict[str, Any]:
        validate_path_safety(request.output_path)
        count = memory.export_json(Path(request.output_path), request.project)
        return {"exported": count, "path": request.output_path}

    return app


# Module-level app instance for uvicorn
app = create_app()
