"""
Stage 4 — Keys and queries (filtered search)

Same idea as Stage 3 (embed English → nearest neighbors), but restrict
candidates with *payload filters* on metadata keys (e.g. categories).

Mental model:
  similarity decides *ranking*;
  filters decide *who is allowed in*.

Run (Qdrant up + Stage 1 data + OPENAI_* in .env):
  poetry run python stage_4.py

Prints a list of 5 arXiv paper ids (all matching the category filter).
"""

from __future__ import annotations

import os

from qdrant_client.http.models.models import ScoredPoint
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchText,
    TextIndexParams,
    TokenizerType,
)

from common import COLLECTION_NAME, client, require_qdrant
from stage_3 import QUERY_TEXT, embed_query

# Demo filter: keep papers tagged with this arXiv category token.
# Dataset stores categories as a space-separated string, e.g. "cs.LG stat.ML".
CATEGORY_TOKEN = "cs.LG"


def ensure_category_text_index() -> None:
    """
    Index the `categories` payload key for full-text MatchText filters.

    Whitespace tokenizer keeps tokens like `cs.LG` intact (a word tokenizer
    would split on the dot into `cs` and `lg`).
    Safe to re-run: skip if the index already exists.
    """
    info = client.get_collection(COLLECTION_NAME)
    existing = info.payload_schema or {}
    if "categories" in existing:
        print("  payload index on 'categories' already present — skipping create.")
        return

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="categories",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WHITESPACE,
            lowercase=True,
        ),
    )
    print("  created text payload index on 'categories'.")


def category_filter(category: str) -> Filter:
    """Papers whose `categories` field contains this token (AND if you add more)."""
    return Filter(
        must=[
            FieldCondition(
                key="categories",
                match=MatchText(text=category),
            )
        ]
    )


def search_filtered(
    query: str,
    *,
    category: str | None = None,
    top_k: int = 5,
) -> list[ScoredPoint]:
    """
    Embed `query`, then query_points with an optional payload filter.

    `query_filter` is applied *during* search (not as a post-pass over
    unfiltered neighbors).
    """
    vector = embed_query(query)
    query_filter = category_filter(category) if category else None

    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points


def filtered_paper_ids(
    query: str = QUERY_TEXT,
    category: str = CATEGORY_TOKEN,
    top_k: int = 5,
) -> list[str]:
    hits = search_filtered(query, category=category, top_k=top_k)
    return [hit.payload["id"] for hit in hits]


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing — set it in .env (do not hardcode).")

    print("Stage 4 — keys and queries (filtered NL search)")
    print(f"  query:    {QUERY_TEXT!r}")
    print(f"  filter:   categories contains {CATEGORY_TOKEN!r}")
    require_qdrant()

    print("Ensuring text payload index on 'categories' ...")
    ensure_category_text_index()

    ids = filtered_paper_ids(QUERY_TEXT, category=CATEGORY_TOKEN, top_k=5)
    print(ids)


if __name__ == "__main__":
    main()
