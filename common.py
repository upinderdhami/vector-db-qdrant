"""
Shared config + Qdrant client for all stages.

Keep dataset path OUTSIDE this project (Hyperskill requirement).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

DATA_PATH = os.getenv(
    "ARXIV_DATA_PATH",
    "/Users/upinder.dhami/workday/sample_data/ml-arxiv-embeddings.json",
)

COLLECTION_NAME = "arxiv_papers"
VECTOR_SIZE = 1536  # text-embedding-ada-002
BATCH_SIZE = 500

client = QdrantClient(host="localhost", port=6333, timeout=60)


def require_qdrant() -> None:
    """Fail fast if Qdrant is not running on localhost:6333."""
    client.get_collections()
