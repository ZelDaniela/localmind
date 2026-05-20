"""CLI for LocalMind - persistent memory for AI agents."""

import secrets
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from localmind import __version__
from localmind.config import Config
from localmind.memory import MemoryStore
from localmind.rag import RAGPipeline
from localmind.agents import AgentRegistry

app = typer.Typer(
    help="LocalMind 🧠 — Persistent memory for local AI agents. 100% offline.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def init() -> None:
    """Initialize LocalMind configuration and storage."""
    config = Config.load()
    config.storage.path.mkdir(parents=True, exist_ok=True)
    config_path = Path.home() / ".localmind" / "config.yaml"
    config.save(config_path)
    console.print(f"[green]✓[/green] LocalMind v{__version__} initialized")
    console.print(f"  Config : {config_path}")
    console.print(f"  Storage: {config.storage.path}")
    console.print("\n[dim]Run [bold]localmind serve[/bold] to start the API server.[/dim]")


@app.command()
def add(
    content: str = typer.Argument(..., help="Content to remember"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
    metadata: Optional[str] = typer.Option(None, "--meta", "-m", help='JSON metadata, e.g. \'{"tag":"work"}\''),
) -> None:
    """Add a memory entry."""
    import json as _json

    memory = MemoryStore()
    meta = {}
    if metadata:
        try:
            meta = _json.loads(metadata)
        except _json.JSONDecodeError:
            console.print("[red]✗[/red] Invalid JSON for --meta")
            raise typer.Exit(1)

    entry_id = memory.add(content, meta or None, project=project)
    console.print(f"[green]✓[/green] Memory added [cyan]{entry_id}[/cyan]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project filter"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
) -> None:
    """Search memories using semantic similarity."""
    memory = MemoryStore()
    results = memory.search(query, n_results=limit, project=project)

    if not results:
        console.print("[yellow]No memories found[/yellow]")
        return

    table = Table(title=f"Search: '{query}'", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Content", style="white")
    table.add_column("Score", style="magenta", justify="right")
    table.add_column("Project", style="blue")

    for result in results:
        distance = result.get("distance")
        score = f"{1 - distance:.2f}" if distance is not None else "N/A"
        content_preview = result["content"]
        if len(content_preview) > 80:
            content_preview = content_preview[:80] + "…"
        table.add_row(
            result["id"][:8],
            content_preview,
            score,
            result["metadata"].get("project", "-"),
        )

    console.print(table)


@app.command(name="list")
def list_memories(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project filter"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of results"),
) -> None:
    """List all memories."""
    memory = MemoryStore()
    results = memory.list_all(limit=limit, project=project)

    if not results:
        console.print("[yellow]No memories found[/yellow]")
        return

    table = Table(title="All Memories", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Content", style="white")
    table.add_column("Project", style="blue")
    table.add_column("Created", style="dim")

    for result in results:
        content_preview = result["content"]
        if len(content_preview) > 70:
            content_preview = content_preview[:70] + "…"
        table.add_row(
            result["id"][:8],
            content_preview,
            result["metadata"].get("project", "-"),
            result["metadata"].get("created_at", "-")[:19],
        )

    console.print(table)
    console.print(f"[dim]{len(results)} memories shown[/dim]")


@app.command()
def delete(
    memory_id: str = typer.Argument(..., help="Memory ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a memory entry."""
    if not force:
        typer.confirm(f"Delete memory {memory_id}?", abort=True)

    memory = MemoryStore()
    deleted = memory.delete(memory_id)

    if deleted:
        console.print(f"[green]✓[/green] Deleted: {memory_id}")
    else:
        console.print(f"[red]✗[/red] Not found: {memory_id}")
        raise typer.Exit(1)


@app.command()
def clear(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project to clear"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Clear memories (all or by project)."""
    target = f"project '{project}'" if project else "ALL memories"
    if not force:
        typer.confirm(f"Delete {target}?", abort=True)

    memory = MemoryStore()
    count = memory.clear(project=project)
    console.print(f"[green]✓[/green] Cleared {count} memories")


@app.command()
def stats() -> None:
    """Show memory statistics."""
    memory = MemoryStore()
    s = memory.get_stats()

    console.print("[bold]LocalMind Statistics[/bold]")
    table = Table(show_header=False)
    table.add_column("Key", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Total memories", str(s["total_memories"]))
    table.add_row("Storage path", s["storage_path"])
    table.add_row("SQLite size", f"{s['sqlite_size_kb']} KB")
    table.add_row("ChromaDB size", f"{s['chroma_size_kb']} KB")
    table.add_row("Embeddings model", s["embeddings_model"])
    console.print(table)


@app.command()
def index(
    path: str = typer.Argument(..., help="File or directory to index"),
    project: str = typer.Option(..., "--project", "-p", help="Project name"),
) -> None:
    """Index a file or directory for RAG."""
    memory = MemoryStore()
    rag = RAGPipeline(memory)
    path_obj = Path(path)

    with console.status(f"Indexing {path}…"):
        if path_obj.is_file():
            result = rag.index_file(path_obj, project)
            console.print(f"[green]✓[/green] Indexed [cyan]{result['indexed']}[/cyan] chunks from file")
        elif path_obj.is_dir():
            result = rag.index_directory(path_obj, project)
            console.print(f"[green]✓[/green] Indexed [cyan]{result['indexed']}[/cyan] chunks")
            if result["errors"]:
                console.print(f"[yellow]⚠[/yellow] {len(result['errors'])} file(s) had errors")
        else:
            console.print(f"[red]✗[/red] Path not found: {path}")
            raise typer.Exit(1)


@app.command()
def export(
    output: str = typer.Argument(..., help="Output JSON file path"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project filter"),
) -> None:
    """Export memories to a JSON file."""
    memory = MemoryStore()
    output_path = Path(output)
    count = memory.export_json(output_path, project=project)
    console.print(f"[green]✓[/green] Exported {count} memories to {output_path}")


@app.command(name="import")
def import_memories(
    input_file: str = typer.Argument(..., help="Input JSON file path"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Override project"),
) -> None:
    """Import memories from a JSON file."""
    memory = MemoryStore()
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]✗[/red] File not found: {input_file}")
        raise typer.Exit(1)
    count = memory.import_json(input_path, project=project)
    console.print(f"[green]✓[/green] Imported {count} memories")


@app.command()
def keygen() -> None:
    """Generate a secure API key for the server."""
    key = secrets.token_urlsafe(32)
    console.print(f"[bold]Generated API key:[/bold]\n[cyan]{key}[/cyan]")
    console.print("\n[dim]Add to ~/.localmind/config.yaml:[/dim]")
    console.print("[dim]security:\\n  api_key_enabled: true\\n  api_key: " + key + "[/dim]")
    console.print("\n[dim]Or set env var: LOCALMIND_API_KEY=" + key + "[/dim]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Server host (default: localhost only)"),
    port: int = typer.Option(8000, "--port", help="Server port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev mode)"),
) -> None:
    """Start the LocalMind API server."""
    import uvicorn
    from localmind.server import create_app

    server_app = create_app()
    console.print(f"[green]✓[/green] LocalMind v{__version__} API starting")
    console.print(f"  Listening: http://{host}:{port}")
    console.print(f"  Docs     : http://{host}:{port}/docs")
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")
    uvicorn.run(server_app, host=host, port=port, reload=reload)


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"LocalMind v{__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()


@app.command()
def mcp(
    http: bool = typer.Option(False, "--http", help="Use HTTP/SSE transport instead of stdio"),
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP server host"),
    port: int = typer.Option(8001, "--port", help="HTTP server port"),
) -> None:
    """Start the LocalMind MCP server for Claude Code and other MCP agents.

    Default transport is stdio (required by Claude Code).
    Use --http for HTTP/SSE transport.

    Claude Code configuration (~/.claude/claude_desktop_config.json):

    \b
    {
      "mcpServers": {
        "localmind": {
          "command": "localmind",
          "args": ["mcp"]
        }
      }
    }
    """
    from localmind.mcp_server import run_stdio, run_http

    if http:
        console.print(f"[green]✓[/green] LocalMind MCP HTTP server: http://{host}:{port}")
        run_http(host=host, port=port)
    else:
        # stdio — don't print anything to stdout, it breaks the protocol
        run_stdio()


@app.command()
def mcp(
    sse: bool = typer.Option(False, "--sse", help="Use SSE transport instead of stdio"),
    host: str = typer.Option("127.0.0.1", "--host", help="SSE server host"),
    port: int = typer.Option(8001, "--port", help="SSE server port"),
) -> None:
    """Start the LocalMind MCP server (for Claude Code and other MCP agents).

    \b
    Stdio mode (default, for Claude Code):
      localmind mcp

    \b
    SSE mode (for web-based agents):
      localmind mcp --sse --port 8001

    \b
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
    from localmind.mcp_server import run_stdio, run_sse

    if sse:
        console.print(f"[green]✓[/green] LocalMind MCP server (SSE) starting on http://{host}:{port}/mcp")
        run_sse(host=host, port=port)
    else:
        console.print("[green]✓[/green] LocalMind MCP server (stdio) starting…", file=sys.stderr)
        run_stdio()
