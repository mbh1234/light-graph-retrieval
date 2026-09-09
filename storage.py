from __future__ import annotations

from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from neo4j import GraphDatabase

from config import settings


class Neo4jFAISSStore:
    def __init__(self):
        self.embedding_dim = settings.EMBEDDING_DIM
        self.chunks: List[Dict[str, Any]] = []
        self.graph_relations: List[Dict[str, Any]] = []

        try:
            self.vector_index = faiss.IndexFlatIP(self.embedding_dim)
        except Exception:
            self.vector_index = None

        self.driver = None
        self.neo4j_available = False
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
            self.driver.verify_connectivity()
            self.neo4j_available = True
            self._init_neo4j_schema()
        except Exception:
            self.neo4j_available = False
            self.driver = None

    def _init_neo4j_schema(self):
        if not self.neo4j_available or self.driver is None:
            return
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

    def close(self):
        if self.driver is not None:
            self.driver.close()

    def add_chunk(self, chunk_id: str, text: str, embedding: np.ndarray):
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.embedding_dim:
            vector = np.resize(vector, (1, self.embedding_dim)).astype(np.float32)

        if self.vector_index is not None:
            faiss.normalize_L2(vector)
            self.vector_index.add(vector)

        self.chunks.append({"id": chunk_id, "text": text, "embedding": vector.reshape(-1)})

    def upsert_entity_relation(self, source: str, target: str, relation: str, keywords: Optional[List[str]] = None):
        if self.neo4j_available and self.driver is not None:
            cypher = (
                "MERGE (a:Entity {name: $source}) "
                "MERGE (b:Entity {name: $target}) "
                "MERGE (a)-[r:RELATION {type: $relation}]->(b) "
                "SET r.keywords = $keywords"
            )
            with self.driver.session() as session:
                session.run(cypher, source=source, target=target, relation=relation, keywords=keywords or [])
            return

        self.graph_relations.append({
            "source": source,
            "relation": relation,
            "target": target,
            "keywords": keywords or [],
        })

    def search_vectors(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.embedding_dim:
            query = np.resize(query, (1, self.embedding_dim)).astype(np.float32)
        if self.vector_index is not None:
            faiss.normalize_L2(query)
            distances, indices = self.vector_index.search(query, min(top_k, len(self.chunks)))
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.chunks):
                    results.append(self.chunks[idx])
            return results

        similarities = []
        for item in self.chunks:
            stored = np.asarray(item["embedding"], dtype=np.float32)
            score = float(np.dot(query.reshape(-1), stored) / (np.linalg.norm(query.reshape(-1)) * np.linalg.norm(stored)))
            similarities.append((score, item))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in similarities[:top_k]]

    def get_1hop_subgraph(self, entities: List[str]) -> List[Dict[str, Any]]:
        entity_names = {name.lower() for name in entities if name}
        if self.neo4j_available and self.driver is not None:
            cypher = (
                "MATCH (a:Entity)-[r:RELATION]->(b:Entity) "
                "WHERE toLower(a.name) IN $entities OR toLower(b.name) IN $entities "
                "RETURN a.name AS source, r.type AS relation, b.name AS target, r.keywords AS keywords "
                "LIMIT 50"
            )
            triples = []
            with self.driver.session() as session:
                result = session.run(cypher, entities=[name.lower() for name in entity_names])
                for record in result:
                    triples.append({
                        "source": record["source"],
                        "relation": record["relation"],
                        "target": record["target"],
                        "keywords": record["keywords"],
                    })
            return triples

        matches = []
        for relation in self.graph_relations:
            source = (relation.get("source") or "").lower()
            target = (relation.get("target") or "").lower()
            if source in entity_names or target in entity_names:
                matches.append(relation)
        return matches[:50]
