from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Graph Retrieval Service"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    MAX_ASYNC_LLM: int = 8

    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password123"

    EMBEDDING_DIM: int = 768
    VECTOR_INDEX_PATH: str = "./data/faiss.index"
    DEFAULT_MODE: str = "hybrid"


settings = Settings()
