# Vector Database with Qdrant (Hyperskill)

One file per stage — earlier stages stay intact.

| File | Stage | What it does |
|------|-------|----------------|
| `common.py` | — | Shared Qdrant client + paths |
| `stage_1.py` | 1/5 | Create `arxiv_papers`, batch upsert embeddings |
| `stage_2.py` | 2/5 | Top-5 similar papers to `1311.5068` |
| `stage_3.py` | 3/5 | *(coming)* natural language → embedding → search |
| `stage_4.py` | 4/5 | *(coming)* metadata filters |
| `stage_5.py` | 5/5 | *(coming)* FastAPI |

## Prerequisites

- Qdrant on `localhost:6333` (Docker or binary)
- Dataset **outside** this repo, e.g.  
  `/Users/upinder.dhami/workday/sample_data/ml-arxiv-embeddings.json`

## Commands

```bash
poetry install
poetry run python stage_1.py   # once (~411k points; slow)
poetry run python stage_2.py   # prints a list of 5 paper ids
```
