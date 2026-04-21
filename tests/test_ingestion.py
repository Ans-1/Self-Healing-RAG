"""
Basic sanity tests for the ingestion pipeline.

Run: pytest tests/test_ingestion.py -v
"""
import pytest
from pathlib import Path
from langchain_core.documents import Document
from src.ingestion.chunker import chunk_documents


def test_chunker_splits_long_text():
    docs = [
        Document(
            page_content="word " * 500,  # 2500 chars — exceeds chunk_size=1000
            metadata={"source": "test"},
        )
    ]
    from unittest.mock import patch

    mock_config = {
        "ingestion": {"chunk_size": 1000, "chunk_overlap": 200},
    }
    with patch("src.ingestion.chunker.load_config", return_value=mock_config):
        chunks = chunk_documents(docs)

    assert len(chunks) > 1, "Long document should be split into multiple chunks"
    for chunk in chunks:
        assert len(chunk.page_content) <= 1100  # allow slight overage at boundaries


def test_chunker_empty_input():
    from unittest.mock import patch

    mock_config = {"ingestion": {"chunk_size": 1000, "chunk_overlap": 200}}
    with patch("src.ingestion.chunker.load_config", return_value=mock_config):
        result = chunk_documents([])

    assert result == []


def test_loader_raises_on_missing_dir():
    from src.ingestion.loader import load_documents

    with pytest.raises(FileNotFoundError):
        load_documents("/nonexistent/path/xyz")
