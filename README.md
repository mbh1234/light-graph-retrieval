# Light Graph Retrieval

A lightweight graph-enhanced retrieval system built for fast document search and grounded question answering. The project blends dense vector retrieval with graph-style entity relationships so queries can benefit from both semantic similarity and structured context without needing a heavyweight knowledge-graph stack.

This repo is intentionally compact and operationally simple: it can run as a FastAPI service, store chunks in FAISS, persist graph edges in Neo4j when available, and degrade gracefully when external services are not running.

## Overview

The system follows a practical retrieval pattern:

1. split text into chunks
2. generate embeddings for each chunk
3. extract entities and relationships from the chunk
4. store embeddings in a FAISS index
5. maintain graph relations in Neo4j when configured
6. retrieve both vector and graph context during a user search
7. return a unified answer grounded in those sources

This is a lighter, more deployable alternative to a large enterprise knowledge graph stack, but it keeps the essential idea: combine semantic retrieval with structured relationships.

## Why this design

A pure vector index is fast and useful for fuzzy semantic matching, but it misses explicit relational context. A pure graph system is structured and explainable, but often harder to scale and slower to query over large unstructured corpora.

This project balances the two:

- vector search handles fast approximate relevance lookup
- graph traversal adds relationship-aware context
- fallback logic keeps the service usable when Neo4j or model services are unavailable

## Architecture

### Core flow

- `main.py` exposes the FastAPI app and routes.
- `retriever.py` contains the async extractor and hybrid retrieval logic.
- `storage.py` owns FAISS and Neo4j integration with offline fallback behavior.
- `config.py` centralizes environment-based settings.
- `prompts.py` holds the extraction and synthesis prompt templates.

### Data flow

- ingest request enters the API
- chunks are embedded using a deterministic vector generator
- the embedding is inserted into FAISS
- each chunk is processed by an extraction pass to collect entities and relations
- graph edges are written to Neo4j if the database is reachable
- query processing performs a vector lookup and optionally a 1-hop graph lookup
- the API returns both the raw results and a combined answer string

## Implementation details

### 1. Embedding and indexing

Each chunk is transformed into a normalized vector. The vector is stored in a FAISS index for nearest-neighbor retrieval.

In the current implementation, the embedding step is intentionally lightweight and deterministic so it works without an external embedding model. That keeps the project easy to run locally and easy to test in CI.

### 2. Graph extraction

The extractor tries to call a model endpoint if configured, but it also gracefully falls back to a heuristic extraction path when the backend is unreachable.

The heuristic pass:

- tokenizes the text
- extracts entity-like phrases
- looks for simple relationship patterns such as "X uses Y", "X stores Y", or "X indexes Y"
- creates simple graph triples and keywords for retrieval

This keeps the project functional even when no LLM server is running.

### 3. Storage layer

The storage class handles two backends:

- FAISS for vector storage and similarity lookups
- Neo4j for graph persistence when connection is available

If Neo4j is unavailable, the system stores relation data in memory so the app still behaves sensibly. This is especially useful for local demos, smoke testing, and offline development.

### 4. Retrieval logic

Search supports three modes:

- `local`: vector-only retrieval
- `global`: graph-focused retrieval
- `hybrid`: combined vector + graph results

The hybrid flow combines:

- top matching chunk results from FAISS
- structured relations from the graph neighborhood
- a summary answer assembled from both sources

## Project structure

- `main.py` — FastAPI application and endpoint wiring
- `retriever.py` — extraction + retrieval logic
- `storage.py` — vector/graph store implementation
- `config.py` — app settings and environment config
- `prompts.py` — prompts for extraction and synthesis
- `benchmark.py` — latency/throughput benchmark script
- `requirements.txt` — Python dependencies
- `Dockerfile` — container image definition
- `docker-compose.yml` — Neo4j container setup
- `.env.example` — environment variable template
- `tests/test_smoke.py` — minimal smoke test for import and retrieval behavior

## Local setup

### Requirements

- Python 3.10+
- optional Neo4j instance
- optional local LLM endpoint if you want real model-based extraction

### Install dependencies

#### Windows / PowerShell

```powershell
cd F:\Downloads\rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

#### macOS / Linux

```bash
cd /path/to/project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and adjust values as needed.

```env
APP_NAME=Graph Retrieval Service
DEBUG=True
HOST=0.0.0.0
PORT=8000

VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
MAX_ASYNC_LLM=8

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

EMBEDDING_DIM=768
```

## Run the app

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You can also run the app with Docker:

```bash
docker-compose up --build
```

## API usage

### Health check

```bash
curl http://localhost:8000/health
```

### Ingest text chunks

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      "Alpha builds products for teams.",
      "Beta uses a graph store for retrieval.",
      "Vector search helps find relevant documents quickly."
    ]
  }'
```

### Search

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does graph retrieval help answer retrieval questions?",
    "mode": "hybrid"
  }'
```

### Supported modes

- `local`: semantic retrieval using vector similarity
- `global`: relation-oriented subgraph retrieval
- `hybrid`: combination of vector and graph context

## Search response structure

The service returns structured results like this:

```json
{
  "query": "How does graph retrieval help answer retrieval questions?",
  "mode": "hybrid",
  "results": {
    "vector": [
      {"id": "chunk-0", "text": "..."}
    ],
    "graph": [
      {"source": "Alpha", "relation": "uses", "target": "graph"}
    ]
  },
  "answer": "..."
}
```

## Benchmarking

A basic benchmark script is included to test ingestion throughput and query latency across modes.

```bash
python benchmark.py
```

## Testing

A smoke test checks the app import path and the retrieval logic.

```bash
python -m pytest tests/test_smoke.py -q
```

## Design trade-offs

This repository intentionally prefers simplicity over maximum sophistication.

### Strengths

- quick to run locally
- minimal dependencies
- good for demos and experimentation
- graceful fallback without a live model or database
- simple API contract

### Limitations

- the embedding path is lightweight and deterministic rather than using a production embedding model
- the extraction layer is heuristic when the external model is unavailable
- graph relations are intentionally simple rather than full enterprise ontology modeling
- the current service is best suited for small-to-medium document collections

## Future improvements

Possible next improvements include:

- true embedding model integration
- better chunking and metadata management
- richer entity normalization and deduplication
- smarter reranking across vector and graph results
- persistent FAISS index storage
- support for async background indexing jobs

## Summary

This project demonstrates a practical lightweight graph retrieval pipeline: fast vector search, light graph reasoning, and a thin API layer. It is designed to be easy to understand and easy to extend, while still staying close to the core ideas behind modern retrieval systems.
