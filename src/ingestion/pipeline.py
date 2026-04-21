"""
Ingestion pipeline — orchestrates: load → chunk → embed → store.

Usage:
    from src.ingestion.pipeline import run_ingestion
    run_ingestion()                          # uses config.yaml data_directory
    run_ingestion("./data/my_docs")          # custom directory
"""
from pathlib import Path
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_and_store
from src.config import load_config
from langchain_chroma import Chroma
from rich.console import Console
from rich.rule import Rule

console = Console()


def run_ingestion(data_dir: str | Path | None = None) -> Chroma:
    """
    Full ingestion pipeline: load → chunk → embed → store in ChromaDB.

    Args:
        data_dir: Path to documents folder. Defaults to config.yaml value.

    Returns:
        Populated ChromaDB vector store.
    """
    config = load_config()

    if data_dir is None:
        data_dir = config["ingestion"]["data_directory"]

    console.print(Rule("[bold blue]Self-RAG Ingestion Pipeline[/bold blue]"))
    console.print(f"[dim]Source:[/dim] {Path(data_dir).resolve()}\n")

    # Step 1 — Load
    documents = load_documents(data_dir)
    if not documents:
        console.print("[red]No documents loaded. Add files to the data directory.[/red]")
        raise SystemExit(1)

    # Step 2 — Chunk
    chunks = chunk_documents(documents)

    # Step 3 — Embed + Store
    vector_store = embed_and_store(chunks)

    console.print(Rule("[bold green]Ingestion Complete[/bold green]"))
    return vector_store
