from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from localmind import __version__
from localmind.config import Config
from localmind.memory import MemoryStore
from localmind.rag import RAGPipeline
from localmind.agents import AgentRegistry

app = typer.Typer(help="LocalMind - Persistent memory for AI agents")
console = Console()


@app.command()
def init():
    """Initialize LocalMind configuration and storage."""
    config = Config.load()
    config.storage.path.mkdir(parents=True, exist_ok=True)
    config.save()
    console.print("[green]✓[/green] LocalMind initialized at ~/.localmind/")


@app.command()
def add(
    content: str = typer.Argument(..., help="Content to remember"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project name"),
):
    """Add a memory entry."""
    memory = MemoryStore()
    entry_id = memory.add(content, project=project)
    console.print(f"[green]✓[/green] Added memory: {entry_id}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project filter"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
):
    """Search memories."""
    memory = MemoryStore()
    results = memory.search(query, n_results=limit, project=project)

    if not results:
        console.print("[yellow]No memories found[/yellow]")
        return

    table = Table(title="Search Results")
    table.add_column("ID", style="cyan")
    table.add_column("Content", style="white")
    table.add_column("Distance", style="magenta")

    for result in results:
        table.add_row(
            result["id"][:8],
            result["content"][:50] + "...",
            f"{result.get('distance', 0):.3f}" if result.get("distance") else "N/A",
        )

    console.print(table)


@app.command()
def list_memories(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project filter"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of results"),
):
    """List all memories."""
    memory = MemoryStore()
    results = memory.list_all(limit=limit, project=project)

    if not results:
        console.print("[yellow]No memories found[/yellow]")
        return

    table = Table(title="All Memories")
    table.add_column("ID", style="cyan")
    table.add_column("Content", style="white")

    for result in results:
        table.add_row(result["id"][:8], result["content"][:60] + "...")

    console.print(table)


@app.command()
def delete(memory_id: str, force: bool = typer.Option(False, "--force", "-f")):
    """Delete a memory entry."""
    if not force:
        confirm = typer.confirm(f"Delete memory {memory_id}?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            return

    memory = MemoryStore()
    deleted = memory.delete(memory_id)

    if deleted:
        console.print(f"[green]✓[/green] Deleted: {memory_id}")
    else:
        console.print(f"[red]✗[/red] Not found: {memory_id}")


@app.command()
def clear(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project to clear"),
    force: bool = typer.Option(False, "--force", "-f"),
):
    """Clear all memories."""
    if not force:
        confirm = typer.confirm("Delete ALL memories?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            return

    memory = MemoryStore()
    count = memory.clear(project=project)
    console.print(f"[green]✓[/green] Cleared {count} memories")


@app.command()
def stats():
    """Show memory statistics."""
    memory = MemoryStore()
    stats = memory.get_stats()

    console.print("[bold]LocalMind Statistics[/bold]")
    console.print(f"Total memories: {stats['total_memories']}")
    console.print(f"Vector DB: {stats['vector_db']}")
    console.print(f"Storage path: {stats['storage_path']}")


@app.command()
def index(
    path: str = typer.Argument(..., help="File or directory to index"),
    project: str = typer.Option(..., "--project", "-p", help="Project name"),
):
    """Index a file or directory for RAG."""
    from pathlib import Path

    memory = MemoryStore()
    rag = RAGPipeline(memory)

    path_obj = Path(path)

    if path_obj.is_file():
        result = rag.index_file(path_obj, project)
        console.print(f"[green]✓[/green] Indexed {result['indexed']} chunks from file")
    elif path_obj.is_dir():
        result = rag.index_directory(path_obj, project)
        console.print(f"[green]✓[/green] Indexed {result['indexed']} chunks")
        if result["errors"]:
            console.print(f"[yellow]Errors: {len(result['errors'])}[/yellow]")
    else:
        console.print(f"[red]✗[/red] Path not found: {path}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Server host"),
    port: int = typer.Option(8000, "--port", help="Server port"),
):
    """Start the LocalMind API server."""
    import uvicorn

    from localmind.server import create_app

    app = create_app()
    console.print(f"[green]✓[/green] Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


@app.command()
def version():
    """Show version."""
    console.print(f"LocalMind v{__version__}")


def main():
    app()


if __name__ == "__main__":
    main()