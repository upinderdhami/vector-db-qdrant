"""
Stage 3 — Natural search

Embed a natural-language query with OpenAI text-embedding-ada-002
(same model as the stored vectors), then find top-5 similar papers in Qdrant.

Run (Qdrant up + Stage 1 data + OPENAI_* in .env):
  poetry run python stage_3.py

Prints a list of 5 arXiv paper ids, e.g.:
  ['2211.08542', '1706.03762', ...]
"""

from __future__ import annotations

import os

from openai import OpenAI
from qdrant_client.http.models.models import ScoredPoint

from common import COLLECTION_NAME, client, require_qdrant

EMBEDDING_MODEL = "text-embedding-ada-002"
QUERY_TEXT = "the attention mechanism in deep learning"

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)


def embed_query(query: str) -> list[float]:
    """Strip newlines and return the OpenAI embedding as a list of floats."""
    cleaned = query.replace("\n", " ").strip()
    response = openai_client.embeddings.create(
        input=[cleaned],
        model=EMBEDDING_MODEL,
    )
    return list(response.data[0].embedding)


def search_by_embedding(
    query_vector: list[float],
    top_k: int = 5,
) -> list[ScoredPoint]:
    """Nearest neighbors in arxiv_papers for the given embedding."""
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points


def search_natural_language(query: str, top_k: int = 5) -> list[str]:
    """Embed query → Qdrant search → list of arXiv paper ids."""
    vector = embed_query(query)
    hits = search_by_embedding(vector, top_k=top_k)
    return [hit.payload["id"] for hit in hits]


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing — set it in .env (do not hardcode).")

    print("Stage 3 — natural search")
    require_qdrant()
    ids = search_natural_language(QUERY_TEXT, top_k=5)
    print(ids)


if __name__ == "__main__":
    main()
