import asyncio

from main import app
from retriever import HybridGraphRetriever
from storage import Neo4jFAISSStore


def test_app_can_be_imported():
    assert app is not None
    assert hasattr(app, "router")


def test_hybrid_retriever_has_ingest_and_search():
    async def _run():
        store = Neo4jFAISSStore()
        try:
            retriever = HybridGraphRetriever(store)
            data = await retriever.ingest_chunks([
                "Alpha builds products for teams.",
                "Beta uses a graph store for retrieval.",
            ])
            assert data["ingested"] == 2
            result = await retriever.search("What does Alpha build for?", mode="hybrid")
            assert result["mode"] == "hybrid"
            assert "answer" in result
        finally:
            store.close()

    asyncio.run(_run())
