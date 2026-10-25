# Vector Database with Qdrant (Hyperskill)

**Learning notes (stages, mental maps, libraries):** see [`LEARNING.md`](./LEARNING.md) — update that file as new stages land.

One file per stage — earlier stages stay intact.

| File | Stage | What it does |
|------|-------|----------------|
| `common.py` | — | Shared Qdrant client + paths |
| `stage_1.py` | 1/5 | Create `arxiv_papers`, batch upsert embeddings |
| `stage_2.py` | 2/5 | Top-5 similar papers to `1311.5068` |
| `stage_3.py` | 3/5 | Natural language → OpenAI embed → top-5 papers |
| `stage_4.py` | 4/5 | NL search + payload filter (`categories` contains `cs.LG`) |
| `stage_5.py` | 5/5 | FastAPI `GET /api/search` (NL + optional category) |

## Prerequisites

- Qdrant on `localhost:6333` (Docker or binary)
- Dataset **outside** this repo, e.g.  
  `/Users/upinder.dhami/workday/sample_data/ml-arxiv-embeddings.json`
- Stage 3+: `.env` with `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`) — never commit `.env`

## Commands

```bash
poetry install
poetry run python stage_1.py   # once (~411k points; slow)
poetry run python stage_2.py   # prints a list of 5 paper ids
poetry run python stage_3.py   # NL query "the attention mechanism in deep learning"
poetry run python stage_4.py   # same NL query, filtered to categories containing cs.LG
poetry run python stage_5.py   # FastAPI on :8000 — open /docs
# curl 'http://127.0.0.1:8000/api/search?q=attention&category=cs.LG'
```
