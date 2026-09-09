from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import settings
from retriever import AsyncGraphExtractor, HybridGraphRetriever
from storage import Neo4jFAISSStore

storage_instance: Optional[Neo4jFAISSStore] = None
extractor_instance: Optional[AsyncGraphExtractor] = None
retriever_instance: Optional[HybridGraphRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global storage_instance, extractor_instance, retriever_instance
    storage_instance = Neo4jFAISSStore()
    extractor_instance = AsyncGraphExtractor(semaphore_limit=settings.MAX_ASYNC_LLM)
    retriever_instance = HybridGraphRetriever(storage_instance, extractor_instance)
    yield
    if extractor_instance is not None:
        await extractor_instance.close()
    if storage_instance is not None:
        storage_instance.close()


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)


class IngestSchema(BaseModel):
    chunks: List[str]


class SearchSchema(BaseModel):
    query: str
    mode: str = settings.DEFAULT_MODE


@app.post("/api/v1/ingest")
async def ingest_documents(payload: IngestSchema):
    if not payload.chunks:
        raise HTTPException(status_code=400, detail="Chunks list cannot be empty.")
    if retriever_instance is None:
        raise HTTPException(status_code=503, detail="Retrieval service is not initialized.")
    return await retriever_instance.ingest_chunks(payload.chunks)


@app.post("/api/v1/search")
async def search_query(payload: SearchSchema):
    if payload.mode not in ["local", "global", "hybrid"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Select 'local', 'global', or 'hybrid'.")
    if retriever_instance is None:
        raise HTTPException(status_code=503, detail="Retrieval service is not initialized.")
    return await retriever_instance.search(payload.query, payload.mode)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": settings.APP_NAME}


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
