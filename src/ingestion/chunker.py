"""
Text chunker — splits loaded documents into smaller, overlapping chunks.

Uses RecursiveCharacterTextSplitter which respects paragraph/sentence
boundaries before falling back to character splits.
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import load_config
from rich.console import Console

console = Console()


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks using settings from config.yaml.
    Returns a list of chunked Document objects.
    """
    if not documents:
        return []

    config = load_config()
    ingest_cfg = config["ingestion"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=ingest_cfg["chunk_size"],
        chunk_overlap=ingest_cfg["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    console.print(
        f"[blue]Chunked[/blue] {len(documents)} page(s) → "
        f"[bold]{len(chunks)}[/bold] chunks "
        f"(size={ingest_cfg['chunk_size']}, overlap={ingest_cfg['chunk_overlap']})"
    )
    return chunks
