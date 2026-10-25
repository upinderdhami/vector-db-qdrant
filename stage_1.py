"""
Stage 1 — Abstract loading

Create collection arxiv_papers and batch-upsert embeddings from the arXiv JSONL.

Run (Qdrant must be up):
  poetry run python stage_1.py
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Generator

from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from common import (
    BATCH_SIZE,
    COLLECTION_NAME,
    DATA_PATH,
    VECTOR_SIZE,
    client,
    require_qdrant,
)


def stream_json(file_path: str) -> Generator[dict[str, Any], None, None]:
    """Yield one JSON object per line (JSONL) without loading the whole file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_collection() -> None:
    """Create arxiv_papers once: 1536-dim vectors, cosine similarity."""
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping create.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}'.")


def record_to_point(record: dict[str, Any]) -> PointStruct | None:
    """Turn one arXiv JSON record into a Qdrant point; skip missing embeddings."""
    embedding = record.get("embedding")
    if not embedding:
        return None

    payload = {k: v for k, v in record.items() if k != "embedding"}
    point_id = str(uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=record["id"]))

    return PointStruct(
        id=point_id,
        vector=embedding,
        payload=payload,
    )


def load_vectors_to_qdrant(file_path: str, batch_size: int = BATCH_SIZE) -> None:
    """Stream file → PointStructs → upsert in batches."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Set ARXIV_DATA_PATH in .env or pass the correct path."
        )

    batch: list[PointStruct] = []
    total_records = 0
    skipped = 0

    print(f"Loading from: {file_path}")
    for record in tqdm(stream_json(file_path), desc="Upserting", unit="rec"):
        point = record_to_point(record)
        if point is None:
            skipped += 1
            continue

        batch.append(point)
        if len(batch) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            total_records += len(batch)
            batch = []

    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        total_records += len(batch)

    print(
        f"Done. Upserted {total_records:,} points "
        f"(skipped {skipped:,} without embeddings)."
    )


def main() -> None:
    print("Stage 1 — connecting to Qdrant at localhost:6333 ...")
    require_qdrant()
    ensure_collection()
    load_vectors_to_qdrant(DATA_PATH)


if __name__ == "__main__":
    main()
