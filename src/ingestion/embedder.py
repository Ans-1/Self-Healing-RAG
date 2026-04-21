"""
Embedder — converts document chunks into vectors and stores them in ChromaDB.

Embedding providers (config.yaml → embeddings.provider):
  huggingface  — local, no API key (default)
  openai       — requires OPENAI_API_KEY
"""
import os
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from src.config import load_config
from rich.console import Console

console = Console()


def get_embeddings() -> Embeddings:
    config = load_config()
    emb_cfg = config["embeddings"]
    provider = emb_cfg["provider"]
    model = emb_cfg["model"]

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        console.print(f"[blue]Embeddings:[/blue] HuggingFace local model '{model}'")
        return HuggingFaceEmbeddings(model_name=model)

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in your .env file.")
        console.print(f"[blue]Embeddings:[/blue] OpenAI '{model}'")
        return OpenAIEmbeddings(model=model, api_key=api_key)

    else:
        raise ValueError(
            f"Unknown embeddings provider: '{provider}'. "
            "Supported: huggingface | openai"
        )


def get_vector_store(embeddings: Embeddings | None = None) -> Chroma:
    """
    Returns a ChromaDB vector store (creates or loads existing collection).
    """
    config = load_config()
    vs_cfg = config["vector_store"]

    if embeddings is None:
        embeddings = get_embeddings()

    return Chroma(
        collection_name=vs_cfg["collection_name"],
        embedding_function=embeddings,
        persist_directory=vs_cfg["persist_directory"],
    )


def embed_and_store(chunks: List[Document]) -> Chroma:
    """
    Embed chunks and persist them in ChromaDB.
    Returns the populated vector store.
    """
    if not chunks:
        console.print("[yellow]Warning:[/yellow] No chunks to embed.")
        return get_vector_store()

    embeddings = get_embeddings()
    config = load_config()
    vs_cfg = config["vector_store"]

    console.print(f"[blue]Embedding[/blue] {len(chunks)} chunks into ChromaDB...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=vs_cfg["collection_name"],
        persist_directory=vs_cfg["persist_directory"],
    )

    console.print(
        f"[green]Done![/green] {len(chunks)} chunks stored in "
        f"collection '[bold]{vs_cfg['collection_name']}[/bold]' "
        f"at {vs_cfg['persist_directory']}"
    )
    return vector_store
