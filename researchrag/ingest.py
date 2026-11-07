"""PDF loading, chunking and FAISS index management.

The index is rebuilt automatically when the PDFs, the chunking settings or the
embedding model change, so a stale index is never used silently.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings

META_FILE = "index_meta.json"


class NormalizedEmbeddings(Embeddings):
    """Wraps any embedding model and scales vectors to unit length (inner product = cosine)."""

    def __init__(self, base: Embeddings):
        self.base = base

    @staticmethod
    def _norm(vec: list[float]) -> list[float]:
        n = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / n for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._norm(list(v)) for v in self.base.embed_documents(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._norm(list(self.base.embed_query(text)))


def get_embeddings(settings: Settings) -> Embeddings:
    """Load the embedding model. Both backends run locally on CPU.

    - fastembed (default): ONNX runtime, small install, no PyTorch. Good for free cloud hosting.
    - huggingface: sentence-transformers + PyTorch.
    """
    if settings.embedding_backend == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        base = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    else:
        from langchain_community.embeddings import FastEmbedEmbeddings

        base = FastEmbedEmbeddings(model_name=settings.embedding_model)
    return NormalizedEmbeddings(base)


def load_pdfs(papers_dir: str) -> list[Document]:
    """Load every PDF page as a Document with `source` and `page` metadata."""
    from langchain_community.document_loaders import PyPDFLoader

    docs: list[Document] = []
    for pdf in sorted(Path(papers_dir).glob("**/*.pdf")):
        docs.extend(PyPDFLoader(str(pdf)).load())
    return docs


def chunk_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split pages into overlapping chunks and give each a stable chunk_id."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(list(documents))
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source_name"] = os.path.basename(chunk.metadata.get("source", "unknown"))
    return chunks


def fingerprint(papers_dir: str, settings: Settings) -> str:
    """Hash of the PDFs plus the settings that affect the index."""
    h = hashlib.sha256()
    for pdf in sorted(Path(papers_dir).glob("**/*.pdf")):
        stat = pdf.stat()
        h.update(f"{pdf.name}:{stat.st_size}:{int(stat.st_mtime)}".encode())
    h.update(
        f"{settings.chunk_size}:{settings.chunk_overlap}:"
        f"{settings.embedding_backend}:{settings.embedding_model}".encode()
    )
    return h.hexdigest()


def build_index(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    # Inner product on normalised vectors = cosine similarity, so scores are easy to read.
    return FAISS.from_documents(
        chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )


def search_with_scores(store: FAISS, query: str, k: int) -> list[tuple[Document, float]]:
    """Top-k chunks with their cosine similarity (higher is more similar)."""
    return [(doc, float(score)) for doc, score in store.similarity_search_with_score(query, k=k)]


def load_or_build_index(
    settings: Settings, embeddings: Embeddings | None = None, force: bool = False
) -> FAISS | None:
    """Return a FAISS index for the current PDFs, rebuilding only when needed.

    Returns None when there are no PDFs to index.
    """
    os.makedirs(settings.papers_dir, exist_ok=True)
    embeddings = embeddings or get_embeddings(settings)
    current = fingerprint(settings.papers_dir, settings)
    meta_path = Path(settings.index_dir) / META_FILE

    if not force and meta_path.exists():
        saved = json.loads(meta_path.read_text())
        if saved.get("fingerprint") == current:
            # The index stores its docstore with pickle. This is safe only because we
            # load an index this app created itself; never load untrusted index files.
            return FAISS.load_local(
                settings.index_dir,
                embeddings,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )

    pages = load_pdfs(settings.papers_dir)
    if not pages:
        return None
    chunks = chunk_documents(pages, settings.chunk_size, settings.chunk_overlap)
    store = build_index(chunks, embeddings)
    store.save_local(settings.index_dir)
    meta_path.write_text(
        json.dumps({"fingerprint": current, "pages": len(pages), "chunks": len(chunks)}, indent=2)
    )
    return store
