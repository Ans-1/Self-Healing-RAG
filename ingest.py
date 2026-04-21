"""
CLI entry point for the ingestion pipeline.

Usage:
    python ingest.py                        # uses config.yaml data_directory
    python ingest.py --data_dir ./my_docs   # custom directory
"""
import argparse
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.pipeline import run_ingestion  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Self-RAG Document Ingestion")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to documents directory (overrides config.yaml)",
    )
    args = parser.parse_args()
    run_ingestion(args.data_dir)


if __name__ == "__main__":
    main()
