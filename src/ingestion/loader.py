"""
Document loader — supports PDF, TXT, Markdown, DOCX.

Returns a list of LangChain Document objects with source metadata.
"""
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from rich.console import Console

console = Console()

SUPPORTED_LOADERS = {".pdf", ".txt", ".md", ".docx"}


def load_documents(data_dir: str | Path) -> List[Document]:
    """
    Recursively load all supported documents from `data_dir`.
    Returns a flat list of LangChain Documents.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    files = [
        f for f in data_path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_LOADERS
    ]

    if not files:
        console.print(
            f"[yellow]Warning:[/yellow] No supported documents found in {data_path}. "
            f"Supported types: {', '.join(SUPPORTED_LOADERS)}"
        )
        return []

    all_docs: List[Document] = []
    for file_path in files:
        docs = _load_single_file(file_path)
        all_docs.extend(docs)
        console.print(f"[green]Loaded[/green] {file_path.name} ({len(docs)} page(s))")

    console.print(f"\n[bold]Total:[/bold] {len(all_docs)} document page(s) from {len(files)} file(s)")
    return all_docs


def _load_single_file(file_path: Path) -> List[Document]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    elif suffix in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    elif suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(str(file_path))
        return loader.load()

    return []
