import asyncio
import time

import httpx
import numpy as np

API_URL = "http://localhost:8000/api/v1"

TEST_CHUNKS = [
    "This retrieval system combines vector similarity with graph traversal.",
    "A local model service can extract entity and relation pairs from text chunks.",
    "Graph relationships help answer multi-hop questions more accurately.",
    "Dense vectors provide fast nearest-neighbor lookup for relevant passages."
] * 10


async def run_benchmark():
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("--- Step 1: Testing Ingestion Throughput ---")
        start = time.time()
        response = await client.post(f"{API_URL}/ingest", json={"chunks": TEST_CHUNKS})
        elapsed = time.time() - start
        print(f"Ingest Result: {response.json()} | elapsed: {elapsed:.2f}s")

        print("\n--- Step 2: Benchmarking Search Latencies ---")
        modes = ["local", "global", "hybrid"]

        for mode in modes:
            latencies = []
            for _ in range(30):
                request_start = time.time()
                await client.post(
                    f"{API_URL}/search",
                    json={"query": "How does the graph help with retrieval?", "mode": mode},
                )
                latencies.append((time.time() - request_start) * 1000)

            avg = sum(latencies) / len(latencies)
            p95 = float(np.percentile(latencies, 95))
            p99 = float(np.percentile(latencies, 99))
            print(f"Mode: [{mode.upper():<6}] | Avg: {avg:.2f}ms | P95: {p95:.2f}ms | P99: {p99:.2f}ms")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
