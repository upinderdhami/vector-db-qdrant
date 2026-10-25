"""
Stage 2 — Similar documents

Find papers most similar to arXiv id 1311.5068 (by embedding cosine similarity).

Run (Qdrant must be up, Stage 1 data loaded):
  poetry run python stage_2.py

Hyperskill expects a printed list of 5 paper ids, e.g.:
  ['1311.5068', '...', '...', '...', '...']
"""

from __future__ import annotations

from qdrant_client.http.models.models import ScoredPoint

from common import COLLECTION_NAME, client, require_qdrant

# Gromov-Hausdorff stability of linkage-based hierarchical clustering methods
QUERY_PAPER_ID = "1311.5068"


def find_similar_papers(
    collection_name: str,
    paper_id: str,
    top_k: int = 5,
) -> list[ScoredPoint]:
    """
    1) scroll — get the point (and vector) for this arXiv id from payload
    2) query_points — nearest neighbors to that vector
    """
    scroll_filter = {"must": [{"key": "id", "match": {"value": paper_id}}]}

    points, _next_offset = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=1,
        with_vectors=True,
        with_payload=True,
    )

    if not points:
        raise ValueError(f"Point not found for paper id: {paper_id}")

    query_vector = points[0].vector
    if isinstance(query_vector, dict):
        query_vector = next(iter(query_vector.values()))

    return client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points


def similar_paper_ids(paper_id: str = QUERY_PAPER_ID, top_k: int = 5) -> list[str]:
    hits = find_similar_papers(COLLECTION_NAME, paper_id, top_k=top_k)
    return [hit.payload["id"] for hit in hits]


def main() -> None:
    print("Stage 2 — similar documents")
    require_qdrant()
    ids = similar_paper_ids(QUERY_PAPER_ID, top_k=5)
    print(ids)


if __name__ == "__main__":
    main()
