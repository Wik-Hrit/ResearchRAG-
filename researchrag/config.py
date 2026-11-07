"""Central configuration. Every value can be overridden with an environment variable."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # Paths
    papers_dir: str = field(default_factory=lambda: os.getenv("PAPERS_DIR", "papers"))
    index_dir: str = field(default_factory=lambda: os.getenv("INDEX_DIR", "faiss_index"))

    # Models
    # "fastembed" (light, no PyTorch, default) or "huggingface" (sentence-transformers)
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "fastembed"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    # Check https://console.groq.com/docs/models for currently available models.
    llm_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
    temperature: float = 0.0

    # Chunking (characters, not tokens: 1000 chars is roughly 200 to 250 tokens)
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 4
    # Minimum relevance score (0 to 1) a chunk needs to be passed on to the agent.
    similarity_threshold: float = 0.25

    # Agent loop limits
    max_rewrites: int = 2
    max_generations: int = 2
