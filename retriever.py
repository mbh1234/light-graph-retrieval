from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

from config import settings
from prompts import GRAPH_EXTRACTION_PROMPT
from storage import Neo4jFAISSStore


class AsyncGraphExtractor:
    def __init__(self, semaphore_limit: int = 8):
        self.semaphore = asyncio.Semaphore(max(1, semaphore_limit))
        self.client = httpx.AsyncClient(timeout=30.0)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return [token for token in cleaned.split() if len(token) > 2]

    def generate_embedding(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        if not tokens:
            tokens = ["empty"]

        rng = np.random.default_rng(abs(hash(" ".join(tokens))) % (2**32))
        vector = rng.standard_normal(settings.EMBEDDING_DIM).astype(np.float32)
        for i, token in enumerate(tokens):
            vector[i % settings.EMBEDDING_DIM] += (len(token) * (i + 1)) / 10.0
        vector = vector / np.linalg.norm(vector)
        return vector

    def _heuristic_extraction(self, text_chunk: str) -> Dict[str, Any]:
        phrases = re.findall(r"[A-Z][A-Za-z0-9_/-]+(?:\s+[A-Z][A-Za-z0-9_/-]+)*|[a-zA-Z]+(?:\s+[a-zA-Z]+){0,2}", text_chunk)
        entities = []
        seen = set()
        for phrase in phrases:
            cleaned = phrase.strip()
            if not cleaned or len(cleaned) < 3:
                continue
            if cleaned.lower() in seen:
                continue
            seen.add(cleaned.lower())
            entities.append({"name": cleaned, "type": "CONCEPT", "description": cleaned})

        relationships: List[Dict[str, Any]] = []
        pattern = r"([A-Za-z0-9_\- ]+?)\s+(uses|stores|contains|supports|connects|indexes|retrieves|depends on|builds)\s+([A-Za-z0-9_\- ]+)"
        for source, relation, target in re.findall(pattern, text_chunk, flags=re.IGNORECASE):
            source = source.strip()
            target = target.strip()
            if source and target:
                relationships.append({
                    "source": source,
                    "target": target,
                    "relation": relation.lower(),
                    "keywords": self._tokenize(f"{source} {target}")
                })

        keywords = self._tokenize(text_chunk)
        if not keywords:
            keywords = ["retrieval", "graph"]
        return {
            "entities": entities[:10],
            "relationships": relationships[:10],
            "high_level_keywords": list(dict.fromkeys(keywords))[:10],
        }

    async def extract_knowledge_async(self, text_chunk: str) -> Dict[str, Any]:
        async with self.semaphore:
            prompt = GRAPH_EXTRACTION_PROMPT.format(text_chunk=text_chunk)
            payload = {
                "model": settings.VLLM_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1024,
            }

            try:
                response = await self.client.post(
                    f"{settings.VLLM_BASE_URL}/chat/completions",
                    json=payload,
                )
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        return parsed
            except Exception:
                pass

            return self._heuristic_extraction(text_chunk)

    async def close(self):
        await self.client.aclose()


class HybridGraphRetriever:
    def __init__(self, store: Neo4jFAISSStore, extractor: Optional[AsyncGraphExtractor] = None):
        self.store = store
        self.extractor = extractor or AsyncGraphExtractor(semaphore_limit=settings.MAX_ASYNC_LLM)

    async def ingest_chunks(self, chunks: List[str]):
        indexed = []
        for idx, chunk in enumerate(chunks):
            chunk_id = f"chunk-{idx}"
            embedding = self.extractor.generate_embedding(chunk)
            self.store.add_chunk(chunk_id, chunk, embedding)

            knowledge = await self.extractor.extract_knowledge_async(chunk)
            for entity in knowledge.get("entities", []):
                name = (entity.get("name") or "").strip()
                if name:
                    self.store.upsert_entity_relation(name, name, "mentions", [name.lower()])
            for relationship in knowledge.get("relationships", []):
                source = (relationship.get("source") or "").strip()
                target = (relationship.get("target") or "").strip()
                relation_type = (relationship.get("relation") or "related_to").strip()
                if source and target:
                    self.store.upsert_entity_relation(source, target, relation_type, relationship.get("keywords", []))

            indexed.append({"id": chunk_id, "status": "indexed"})

        return {"ingested": len(chunks), "result": indexed}

    async def search(self, query: str, mode: str = "hybrid"):
        normalized_mode = mode.lower()
        if normalized_mode not in {"local", "global", "hybrid"}:
            raise ValueError("Mode must be 'local', 'global', or 'hybrid'.")

        query_embedding = self.extractor.generate_embedding(query)
        vector_hits = self.store.search_vectors(query_embedding, top_k=5)
        query_terms = self.extractor._tokenize(query)
        graph_hits = self.store.get_1hop_subgraph(query_terms) if query_terms else []

        if normalized_mode == "local":
            results = {"vector": vector_hits}
        elif normalized_mode == "global":
            results = {"graph": graph_hits}
        else:
            results = {"vector": vector_hits, "graph": graph_hits}

        answer_parts = []
        for item in vector_hits:
            answer_parts.append(item["text"])
        for triple in graph_hits:
            answer_parts.append(f"{triple['source']} {triple['relation']} {triple['target']}")

        return {
            "query": query,
            "mode": normalized_mode,
            "results": results,
            "answer": " ".join(answer_parts[:8]) if answer_parts else "No relevant chunks or graph context found.",
        }
