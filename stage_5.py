"""
Stage 5 — FastAPI wrapper

Expose Stages 3–4 as an HTTP API:
  GET /api/search?q=...&category=cs.LG&limit=5

Mental model:
  Browser / curl  →  FastAPI  →  embed (OpenAI)  →  Qdrant query_points
                                    ↑ optional payload filter (category)

Run (Qdrant up + OPENAI_* in .env):
  poetry run python stage_5.py
  # then open http://localhost:8000/docs

Try:
  curl 'http://localhost:8000/api/search?q=the%20attention%20mechanism&category=cs.LG'
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from common import require_qdrant
from stage_4 import ensure_category_text_index, search_filtered


class PaperHit(BaseModel):
    """One search result returned to the client (payload + score, no vectors)."""

    id: str
    title: str | None = None
    categories: str | None = None
    score: float
    abstract: str | None = None


class SearchResponse(BaseModel):
    query: str
    category: str | None = None
    result: list[PaperHit]


def _hit_to_paper(hit: Any) -> PaperHit:
    payload = hit.payload or {}
    return PaperHit(
        id=str(payload.get("id", "")),
        title=payload.get("title"),
        categories=payload.get("categories"),
        score=float(hit.score),
        abstract=payload.get("abstract"),
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start-up: fail fast if Qdrant / API key missing; ensure category index."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing — set it in .env (do not hardcode).")
    require_qdrant()
    ensure_category_text_index()
    yield


app = FastAPI(
    title="arXiv semantic search",
    description=(
        "Natural-language search over the Qdrant `arxiv_papers` collection "
        "(OpenAI text-embedding-ada-002 + optional category filter)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check — no embedding call."""
    return {"status": "ok"}


@app.get("/api/search", response_model=SearchResponse)
def search_papers(
    q: str = Query(..., min_length=1, description="Natural-language search query"),
    category: str | None = Query(
        None,
        description="Optional arXiv category token filter, e.g. cs.LG",
    ),
    limit: int = Query(5, ge=1, le=20, description="Max number of papers to return"),
) -> SearchResponse:
    """
    Embed `q`, search Qdrant, optionally restrict to papers whose
    `categories` payload contains `category` (Stage 4 MatchText filter).
    """
    try:
        hits = search_filtered(q, category=category, top_k=limit)
    except Exception as exc:  # noqa: BLE001 — surface upstream errors as 502
        raise HTTPException(status_code=502, detail=f"search failed: {exc}") from exc

    return SearchResponse(
        query=q,
        category=category,
        result=[_hit_to_paper(h) for h in hits],
    )


def main() -> None:
    print("Stage 5 — FastAPI on http://127.0.0.1:8000")
    print("  interactive docs: http://127.0.0.1:8000/docs")
    print("  example: curl 'http://127.0.0.1:8000/api/search?q=attention&category=cs.LG'")
    uvicorn.run(
        "stage_5:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
