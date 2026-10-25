# Vector Database with Qdrant — Learning notes

Living document for the Hyperskill project. Update this file as each new stage lands.

**Last covered:** Stage 5 (FastAPI wrapper) — project complete  
**Repo layout:** one script per stage (`stage_1.py` … `stage_5.py`) + shared `common.py`

---

## Big-picture mental map

Semantic search = put documents and queries into the **same vector space**, then find nearest neighbors.

```text
                    ┌─────────────────────────────────────┐
                    │         Embedding model             │
                    │   text-embedding-ada-002 (1536-d)   │
                    └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
   Paper abstract            Paper abstract            English query
   (offline, in JSONL)       already stored            (Stage 3 live)
          │                        │                        │
          ▼                        ▼                        ▼
   [0.01, -0.2, …]           stored in Qdrant          [0.02, -0.19, …]
          │                        │                        │
          └──────────► upsert ─────┘                        │
                           │                                │
                           ▼                                │
                    Qdrant collection                       │
                    arxiv_papers                            │
                    (~411k points, Cosine)                  │
                           │◄───────────────────────────────┘
                           │         query_points
                           ▼
                    top-k similar papers (by id)
```

| Stage | Question you can answer | Input | How you get a vector |
|-------|-------------------------|-------|----------------------|
| 1 | “Can I store paper embeddings?” | JSONL file | Pre-computed in the dataset |
| 2 | “Papers like *this paper*?” | arXiv id `1311.5068` | Look up that paper’s stored vector |
| 3 | “Papers about *this topic*?” | English sentence | Call OpenAI to embed the sentence |
| 4 | “About this topic, but only in *cs.LG*?” | English + payload filter | Same as Stage 3 + `query_filter` |
| 5 | “Call search over HTTP” | `GET /api/search?q=…` | Same as Stage 4, behind FastAPI |

---

## Project structure

```text
vector-db-qdrant/
├── common.py          # Shared: client, paths, collection name, require_qdrant()
├── stage_1.py         # Create collection + batch upsert
├── stage_2.py         # Similar-to-paper search (scroll → query)
├── stage_3.py         # NL query → OpenAI embed → query
├── stage_4.py         # NL search + Filter on payload keys
├── stage_5.py         # FastAPI GET /api/search
├── main.py            # Entry map / pointer to stages
├── pyproject.toml     # Poetry deps (+ Workday Artifactory source)
├── .env.template      # What to put in .env (never commit .env)
├── README.md          # Short how-to-run
└── LEARNING.md        # This document
```

**Why one file per stage?** Hyperskill checks stages independently; keeping earlier scripts intact means you can re-run Stage 1/2 without rewriting Stage 3 logic, and git branches (`stage-1`, `stage-2`, `stage-3`) stay clear.

**Dataset stays outside the repo** (Hyperskill requirement), e.g.:

```text
/Users/upinder.dhami/workday/sample_data/ml-arxiv-embeddings.json
```

Override with `ARXIV_DATA_PATH` in `.env` if needed.

---

## Libraries and what each is for

| Library | Where used | Purpose in *this* project |
|---------|------------|---------------------------|
| **qdrant-client** | `common`, stages 1–4 | Talk to Qdrant over HTTP: create collections, upsert, scroll, `query_points`, payload filters/indexes. |
| **openai** | Stages 3–5 | Call OpenAI Embeddings API (`text-embedding-ada-002`) so a natural-language string becomes a 1536-float vector in the **same** space as the stored papers. |
| **fastapi** | Stage 5 | Declare HTTP routes (`GET /api/search`); auto OpenAPI docs at `/docs`. |
| **uvicorn** | Stage 5 | ASGI server that runs the FastAPI app (listens on port 8000). |
| **pydantic** | Stage 5 (via FastAPI) | Request/response models (`PaperHit`, `SearchResponse`) — validates + documents JSON shapes. |
| **python-dotenv** | `common.py` | Load `.env` into `os.environ` at import time (`OPENAI_API_KEY`, `ARXIV_DATA_PATH`, …) without hardcoding secrets. |
| **tqdm** | Stage 1 | Progress bar while streaming ~411k records so a long upsert isn’t a silent hang. |
| **json** (stdlib) | Stage 1 | Parse one JSON object per line (JSONL). |
| **uuid** (stdlib) | Stage 1 | Deterministic Qdrant point ids from arXiv ids (`uuid5`), so re-upserts update the same points. |
| **Poetry** | tooling | Lock deps; on Workday laptop packages come from Artifactory (public PyPI blocked). |

### Qdrant concepts (vocabulary)

| Term | Meaning | Example in our code |
|------|---------|---------------------|
| **Collection** | Named table of vectors + payloads | `arxiv_papers` |
| **Point** | One item: `id` + `vector` + optional `payload` | One paper |
| **Vector** | List of floats (embedding) | 1536 dims, Cosine distance |
| **Payload** | Metadata JSON attached to the point | `id`, title, abstract, … (everything except `embedding`) |
| **Upsert** | Insert or replace points by id | Stage 1 batches of 500 |
| **Scroll** | Filter/page through points (not similarity) | Stage 2: find point where `payload.id == "1311.5068"` |
| **query_points** | Nearest-neighbor search by vector | Stages 2–4 |
| **Filter / FieldCondition** | Restrict which points may be returned | Stage 4: `categories` contains `cs.LG` |
| **Payload index** | Index on a payload key for efficient filters | Stage 4: text index on `categories` |

### Tiny payload vs vector example

One JSONL record (simplified):

```json
{
  "id": "1706.03762",
  "title": "Attention Is All You Need",
  "abstract": "The dominant sequence transduction models...",
  "embedding": [0.012, -0.084, 0.003, "...1536 floats"]
}
```

What we store in Qdrant:

```text
PointStruct(
  id      = uuid5("1706.03762"),     # Qdrant's internal point id (UUID string)
  vector  = embedding,               # used for similarity
  payload = { id, title, abstract, … }  # embedding stripped out — searchable metadata
)
```

When we print results we use **`payload["id"]`** (arXiv id), not the UUID point id.

---

## Shared layer: `common.py`

Loads env, defines constants, builds one `QdrantClient`.

```text
load_dotenv()
    │
    ├── DATA_PATH        → JSONL location
    ├── COLLECTION_NAME  → "arxiv_papers"
    ├── VECTOR_SIZE      → 1536
    ├── BATCH_SIZE       → 500
    └── client           → QdrantClient(localhost:6333)

require_qdrant()  → client.get_collections()  # fail fast if Docker/Qdrant is down
```

Every stage starts with `require_qdrant()` so a missing server fails immediately with a clear error.

---

## Stage 1 — Abstract loading (ingest)

**Goal:** Create `arxiv_papers` and load all paper embeddings from the JSONL.

### Mental map

```text
ml-arxiv-embeddings.json  (JSONL, one paper per line)
        │
        │  stream_json() — never load whole file into RAM
        ▼
   for each record:
        │
        ├─ no embedding? → skip
        └─ record_to_point() → PointStruct
                │
                ▼
           batch list (size 500)
                │
                │  client.upsert(...)
                ▼
           Qdrant: arxiv_papers
```

### Key design choices

1. **Stream, don’t `json.load` the whole file** — hundreds of thousands of lines with large vectors; streaming keeps memory flat.
2. **Cosine + 1536** — must match the embedding model used to build the dataset (`text-embedding-ada-002`).
3. **Batch upsert (500)** — far fewer round-trips than one HTTP call per paper.
4. **Deterministic UUID5 from arXiv id** — re-running Stage 1 updates the same points instead of duplicating them.
5. **Payload = everything except `embedding`** — vectors live in the vector index; metadata stays queryable/filterable later.

### Run

```bash
# Qdrant must be listening on localhost:6333
poetry run python stage_1.py
```

Expect ~411,220 upserted points (exact count depends on how many rows have embeddings).

### Mini example (3 papers)

```text
Input lines → 3 PointStructs → upsert once
Qdrant now has 3 nearest-neighbor candidates for later stages.
```

---

## Stage 2 — Similar documents (vector → vector)

**Goal:** Given paper `1311.5068`, print the top-5 most similar paper ids (including itself as nearest).

### Mental map

```text
arXiv id "1311.5068"
        │
        │  scroll + filter: payload.id == "1311.5068"
        │  with_vectors=True
        ▼
   stored embedding of that paper
        │
        │  query_points(query=that_vector, limit=5)
        ▼
   [ScoredPoint, ScoredPoint, ...]  ranked by cosine similarity
        │
        ▼
   print [payload["id"], ...]
```

### Why two Qdrant calls?

| Call | Role |
|------|------|
| `scroll` | “Find the point whose **metadata** says this arXiv id” — exact match on payload, not similarity. |
| `query_points` | “Given that point’s **vector**, who is nearby?” — similarity search. |

Stage 2 never calls OpenAI: the query vector already lives in the collection.

### Concrete walkthrough

```text
Paper A  id=1311.5068  vector ≈ [0.1, 0.2, ...]
Paper B  (clustering, related)   close in space
Paper C  (unrelated NLP)         far away

query with A's vector → ranking: A, B, … (C far down the list)
```

Hyperskill expects a **printed Python list of 5 strings** (arXiv ids).

### Run

```bash
poetry run python stage_2.py
# e.g. ['1311.5068', '....', '....', '....', '....']
```

---

## Stage 3 — Natural search (text → vector → vector)

**Goal:** Embed the sentence `"the attention mechanism in deep learning"` with the **same** model as the corpus, then return top-5 paper ids.

### Mental map

```text
QUERY_TEXT = "the attention mechanism in deep learning"
        │
        │  replace newlines with spaces (OpenAI embedding hygiene)
        ▼
   OpenAI embeddings.create(model="text-embedding-ada-002")
        │
        ▼
   query_vector: list[float] length 1536
        │
        │  query_points (same as Stage 2 from here on)
        ▼
   top-5 payload["id"] values printed
```

### Stage 2 vs Stage 3 (side by side)

```text
Stage 2                          Stage 3
───────                          ───────
Have a paper id                  Have English words
  │                                │
  scroll → get stored vector       OpenAI → create new vector
  │                                │
  └────────── query_points ────────┘
                    │
                    ▼
              top-k arXiv ids
```

**Critical rule:** query embedding model **must** match corpus embedding model. Mixing models = random-looking neighbors.

### Code shape

| Function | Job |
|----------|-----|
| `embed_query(query)` | Clean text → OpenAI → `list[float]` |
| `search_by_embedding(vector)` | Qdrant `query_points` |
| `search_natural_language(query)` | Glue: embed → search → extract ids |

### Sanity check

For “attention mechanism in deep learning”, ids near the famous Transformer paper (`1706.03762`) are a good sign the pipeline is correct (exact ranking can vary slightly).

### Run

```bash
# .env must include OPENAI_API_KEY (optional OPENAI_BASE_URL)
poetry install   # pulls openai if not already
poetry run python stage_3.py
# e.g. ['1706.03762', '....', ...]
```

---

## Stage 4 — Keys and queries (filtered search)

**Goal:** Keep Stage 3’s natural-language search, but only return papers whose **payload** matches extra conditions (here: arXiv category `cs.LG`).

### Why filters exist

Embeddings capture *meaning*, not hard business rules. You often need both:

```text
“papers about attention”           ← vector similarity (soft)
“AND tagged Machine Learning”      ← payload filter (hard)
```

Without a filter, a great neighbor might be in `cs.CV` or `physics.*`. With a filter, those points are **never candidates**.

### Mental map

```text
QUERY_TEXT  ──► OpenAI embed ──► query_vector
                                      │
                                      ▼
                              query_points(
                                query=vector,
                                query_filter=Filter(   ◄── NEW in Stage 4
                                  must=[ FieldCondition(
                                    key="categories",
                                    match=MatchText(text="cs.LG")
                                  )]
                                ),
                                limit=5
                              )
                                      │
                                      ▼
                              top-5 ids (all contain cs.LG)
```

### Stage 3 vs Stage 4

```text
Stage 3:  English → embed → nearest among ALL papers
Stage 4:  English → embed → nearest among papers that PASS the filter
```

Same ranking idea; smaller allowed set.

### Filter vocabulary (the “keys” part)

| Piece | Role | Analogy |
|-------|------|---------|
| **payload key** | Field name in metadata | SQL column name (`categories`) |
| **FieldCondition** | One check on one key | `categories LIKE …` |
| **MatchValue** | Exact equality | `category = 'Technology'` |
| **MatchText** | Full-text / token contains | “field contains this token” |
| **Filter.must** | AND — all conditions true | |
| **Filter.should** | OR — at least one true | |
| **Filter.must_not** | NOT — none of these | |
| **payload index** | Index on a key for faster filters | SQL index on a column |

### Why `MatchText` on `categories` (not `MatchValue`)

In our JSONL, categories look like:

```text
"cs.LG"                  ← single token
"cs.LG stat.ML"          ← multiple tokens in ONE string
"stat.ML cs.LG"          ← same tokens, different order
```

`MatchValue(value="cs.LG")` only matches the **exact full string** `"cs.LG"`, and misses `"cs.LG stat.ML"`.

`MatchText(text="cs.LG")` matches if that **token** appears inside the field — correct for multi-label papers.

### Why a whitespace text index?

Default word tokenizer splits on punctuation: `cs.LG` → `cs` + `lg` (noisy).

Stage 4 creates a **text** payload index with `TokenizerType.WHITESPACE` so `cs.LG` stays one token. That is the “keys” half of *Keys and queries*: you declare which payload keys matter and how they are indexed.

### Tiny example

```text
Unfiltered top hits for "attention…":   maybe mix of cs.LG, cs.CL, cs.AI, …
With filter categories ∋ cs.LG:         every returned paper has cs.LG in categories
```

### Run

```bash
poetry run python stage_4.py
# prints e.g. ['1706.03762', '...', ...]  — list of 5 ids
```

First run may spend a bit building the `categories` index; later runs skip create if it already exists.

---

## Stage 5 — FastAPI wrapper (HTTP API)

**Goal:** Turn the Stage 3/4 search functions into a small web service so any client (browser, curl, another app) can search without importing Python modules.

### Mental map

```text
Client (curl /browser /docs UI)
        │
        │  GET /api/search?q=attention&category=cs.LG&limit=5
        ▼
   FastAPI route  search_papers()
        │
        │  reuses stage_4.search_filtered()
        ▼
   OpenAI embed  →  Qdrant query_points (+ optional filter)
        │
        ▼
   JSON { query, category, result: [ {id, title, score, …}, … ] }
```

Scripts (Stages 1–4) are great for learning and Hyperskill checks. An **API** is how you ship the same logic to other programs.

### What each library does here

| Piece | Job |
|-------|-----|
| **FastAPI `app`** | Registers routes; builds OpenAPI schema |
| **`lifespan`** | On startup: check API key, ping Qdrant, ensure payload index |
| **`Query(...)`** | Declares query-string params (`q`, `category`, `limit`) with validation |
| **Pydantic models** | Shape of the JSON response (`PaperHit`, `SearchResponse`) |
| **uvicorn** | Process that binds TCP port 8000 and serves the app |

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Cheap liveness (no OpenAI call) |
| `GET` | `/api/search` | Semantic search; optional `category` filter |
| `GET` | `/docs` | Interactive Swagger UI (auto-generated) |

### Example request / response

```bash
curl -s 'http://127.0.0.1:8000/api/search?q=the%20attention%20mechanism&category=cs.LG&limit=3'
```

```json
{
  "query": "the attention mechanism",
  "category": "cs.LG",
  "result": [
    {
      "id": "2203.14263",
      "title": "A General Survey on Attention Mechanisms in Deep Learning",
      "categories": "cs.LG",
      "score": 0.91,
      "abstract": "..."
    }
  ]
}
```

`score` is cosine similarity from Qdrant (higher ≈ closer). Vectors are **not** returned — clients usually only need metadata.

### Run

```bash
poetry install          # pulls fastapi + uvicorn
poetry run python stage_5.py
# open http://127.0.0.1:8000/docs  and try the endpoint
```

Stop with `Ctrl+C`.

### How this reuses earlier stages (no rewrite)

```text
stage_5.search_papers
        │
        └── stage_4.search_filtered
                  │
                  ├── stage_3.embed_query   (OpenAI)
                  └── common.client         (Qdrant)
```

That layering is the main engineering lesson: **core search logic stays importable; the API is a thin shell.**

---

## Environment & ops cheat sheet

| Need | How |
|------|-----|
| Qdrant | Docker (or binary) on `localhost:6333` |
| Python deps | `poetry install` (Artifactory source on Workday laptop) |
| Secrets | `.env` from `.env.template` — **never commit `.env`** |
| Stage 1 data | JSONL outside repo; path via `ARXIV_DATA_PATH` or default in `common.py` |
| Hyperskill Check | Run the matching `stage_N.py` with Qdrant up |

Typical `.env` keys:

```bash
ARXIV_DATA_PATH=/path/to/ml-arxiv-embeddings.json
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=          # optional proxy / compatible endpoint
```

---

## End-to-end story (one narrative)

1. **Stage 1** fills a vector warehouse with ~411k arXiv papers (vectors pre-baked in a file).
2. **Stage 2** proves similarity works: pick a paper, reuse its vector, find neighbors.
3. **Stage 3** proves *search* works for humans: turn a sentence into a vector with OpenAI, then reuse the same Qdrant search path.
4. **Stage 4** adds hard constraints: same search, but only among papers whose payload keys match (category filter + payload index).
5. **Stage 5** wraps that search in FastAPI so other clients call it over HTTP (`/api/search`, `/docs`).

You now have a full path: **ingest → similarity-by-id → NL search → filtered search → HTTP API**.

---

## Stages still to document

*(none — Stages 1–5 covered. Extend this file if you add extras beyond Hyperskill.)*

---

## Quick command reference

```bash
cd /Users/upinder.dhami/workday/src/ai-course/vector-db-qdrant

poetry install
poetry run python stage_1.py   # once (slow)
poetry run python stage_2.py   # list of 5 ids
poetry run python stage_3.py   # NL → list of 5 ids
poetry run python stage_4.py   # NL + categories filter → list of 5 ids
poetry run python stage_5.py   # FastAPI on :8000
```
