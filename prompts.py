GRAPH_EXTRACTION_PROMPT = """
You are a high-precision Knowledge Graph extraction engine.
Given the following text chunk, identify key entities, relationships between entities, and high-level query keywords.

Text Chunk:
"{text_chunk}"

Output MUST be strictly valid JSON matching this schema:
{{
  "entities": [
    {{"name": "EntityName", "type": "CATEGORY/CONCEPT/ORGANIZATION/TECHNOLOGY", "description": "Brief summary"}}
  ],
  "relationships": [
    {{"source": "EntityA", "target": "EntityB", "relation": "relationship_type", "keywords": ["key1", "key2"]}}
  ],
  "high_level_keywords": ["GlobalTheme1", "GlobalTheme2"]
}}

Return ONLY JSON. Do not include markdown framing or intro text.
"""

SYNTHESIS_PROMPT = """
You are an enterprise AI assistant answering a user query using dual-level retrieval context.

Context from Knowledge Graph Subgraph (Structured Triples):
{graph_context}

Context from Vector Search Chunks (Unstructured Snippets):
{vector_context}

User Query: "{query}"

Provide a comprehensive, factual response directly grounded in the provided graph and document context.
"""
