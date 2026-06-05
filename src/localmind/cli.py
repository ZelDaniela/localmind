"""CLI for LocalMind — persistent memory for local AI agents."""

from __future__ import annotations

import json as _json
import secrets
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from localmind import __version__
from localmind.config import Config
from localmind.memory import MemoryStore
from localmind.rag import RAGPipeline

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
    if not config_path.exists():
        config.save(config_path)
        console.print(f"[green]✓[/green] Config created at {config_path}")
    else:
        console.print(f"[dim]Config already exists at {config_path}[/dim]")
    console.print(f"[green]✓[/green] LocalMind v{__version__} initialized")
    console.print(f"  Storage: {config.storage.path}")
    console.print("\n[dim]Run [bold]localmind serve[/bold] to start the API server.[/dim]")


@app.command()
def add(
    content: str = typer.Argument(..., help="Content to remember"),
    project: str | None = typer.Option(None, "--project", "-p"),
    metadata: str | None = typer.Option(
        None, "--meta", "-m", help='JSON metadata, e.g. \'{"tag":"work"}\''
    ),
) -> None:
    """Add a memory entry."""
    meta: dict = {}
    if metadata:
        try:
            meta = _json.loads(metadata)
        except _json.JSONDecodeError:
            console.print("[red]✗[/red] Invalid JSON for --meta")
            raise typer.Exit(1)

    memory = MemoryStore()
    entry_id = memory.add(content, meta or None, project=project)
    console.print(f"[green]✓[/green] Memory added [cyan]{entry_id}[/cyan]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    project: str | None = typer.Option(None, "--project", "-p"),
    limit: int = typer.Option(5, "--limit", "-n"),
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

    for r in results:
        d = r.get("distance")
        score = f"{1 - d:.2f}" if d is not None else "N/A"
        preview = r["content"][:80] + ("…" if len(r["content"]) > 80 else "")
        table.add_row(r["id"][:8], preview, score, r["metadata"].get("project", "-"))

    console.print(table)


@app.command(name="list")
def list_memories(
    project: str | None = typer.Option(None, "--project", "-p"),
    limit: int = typer.Option(20, "--limit", "-n"),
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

    for r in results:
        preview = r["content"][:70] + ("…" if len(r["content"]) > 70 else "")
        table.add_row(
            r["id"][:8],
            preview,
            r["metadata"].get("project", "-"),
            r["metadata"].get("created_at", "-")[:19],
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
    project: str | None = typer.Option(None, "--project", "-p"),
    force: bool = typer.Option(False, "--force", "-f"),
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
    """Show storage statistics."""
    memory = MemoryStore()
    s = memory.get_stats()
    table = Table(title="LocalMind Statistics", show_header=False)
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
    project: str = typer.Option(..., "--project", "-p"),
) -> None:
    """Index a file or directory for RAG."""
    memory = MemoryStore()
    rag = RAGPipeline(memory)
    path_obj = Path(path)

    with console.status(f"Indexing {path}…"):
        try:
            if path_obj.is_file():
                result = rag.index_file(path_obj, project)
                console.print(
                    f"[green]✓[/green] Indexed [cyan]{result['indexed']}[/cyan] chunks from file"
                )
            elif path_obj.is_dir():
                result = rag.index_directory(path_obj, project)
                console.print(
                    f"[green]✓[/green] Indexed [cyan]{result['indexed']}[/cyan] chunks "
                    f"([dim]{result['skipped']} skipped[/dim])"
                )
                if result["errors"]:
                    console.print(f"[yellow]⚠[/yellow] {len(result['errors'])} file(s) had errors")
            else:
                console.print(f"[red]✗[/red] Path not found: {path}")
                raise typer.Exit(1)
        except ValueError as e:
            console.print(f"[red]✗[/red] {e}")
            raise typer.Exit(1)


@app.command()
def export(
    output: str = typer.Argument(..., help="Output JSON file path"),
    project: str | None = typer.Option(None, "--project", "-p"),
) -> None:
    """Export memories to a JSON file."""
    memory = MemoryStore()
    count = memory.export_json(Path(output), project=project)
    console.print(f"[green]✓[/green] Exported {count} memories to {output}")


@app.command(name="import")
def import_memories(
    input_file: str = typer.Argument(..., help="Input JSON file path"),
    project: str | None = typer.Option(None, "--project", "-p"),
) -> None:
    """Import memories from a JSON file."""
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]✗[/red] File not found: {input_file}")
        raise typer.Exit(1)
    memory = MemoryStore()
    count = memory.import_json(input_path, project=project)
    console.print(f"[green]✓[/green] Imported {count} memories")


@app.command()
def keygen() -> None:
    """Generate a secure API key for the server."""
    key = secrets.token_urlsafe(32)
    console.print(f"[bold]Generated API key:[/bold]\n[cyan]{key}[/cyan]")
    console.print("\n[dim]Add to ~/.localmind/config.yaml:[/dim]")
    console.print(f"[dim]security:\n  api_key_enabled: true\n  api_key: {key}[/dim]")
    console.print(f"\n[dim]Or set env var: LOCALMIND_API_KEY={key}[/dim]")


@app.command()
def mcp(
    sse: bool = typer.Option(False, "--sse", help="Use SSE transport instead of stdio"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8001, "--port"),
) -> None:
    """Start the LocalMind MCP server (for Claude Code and MCP agents).

    \b
    Stdio mode (default, for Claude Code):
      localmind mcp

    \b
    SSE mode:
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
    from localmind.mcp_server import run_sse, run_stdio

    if sse:
        console.print(f"[green]✓[/green] LocalMind MCP (SSE) on http://{host}:{port}/mcp")
        run_sse(host=host, port=port)
    else:
        print("LocalMind MCP server starting (stdio)…", file=sys.stderr)
        run_stdio()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8000, "--port", help="Server port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload (dev mode)"),
) -> None:
    """Start the LocalMind REST API server."""
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
