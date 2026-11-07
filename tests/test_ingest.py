import os
import time

from langchain_core.documents import Document

from researchrag.config import Settings
from researchrag.ingest import chunk_documents, load_or_build_index, search_with_scores


def _settings(tmp_path):
    s = Settings()
    s.papers_dir = str(tmp_path / "papers")
    s.index_dir = str(tmp_path / "index")
    s.chunk_size = 200
    s.chunk_overlap = 20
    return s


def test_chunks_get_ids_and_source_names():
    pages = [Document(page_content="word " * 200, metadata={"source": "/x/paper.pdf", "page": 0})]
    chunks = chunk_documents(pages, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["source_name"] == "paper.pdf" for c in chunks)
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_index_search_returns_cosine_scores(tmp_path, embeddings, make_pdf):
    s = _settings(tmp_path)
    make_pdf("rc.pdf", [
        "Echo state networks keep reservoir weights fixed and train only the readout layer.",
        "Liquid state machines use spiking neurons for temporal signal processing.",
    ])
    store = load_or_build_index(s, embeddings)
    hits = search_with_scores(store, "echo state networks readout layer", k=2)

    assert hits[0][0].metadata["page"] == 0
    assert hits[0][1] > hits[1][1]
    assert all(-1.0 <= score <= 1.0001 for _, score in hits)


def test_index_is_reused_then_rebuilt_when_papers_change(tmp_path, embeddings, make_pdf):
    s = _settings(tmp_path)
    make_pdf("a.pdf", ["first paper about reservoirs"])
    load_or_build_index(s, embeddings)
    meta = os.path.join(s.index_dir, "index_meta.json")
    first_mtime = os.path.getmtime(meta)

    time.sleep(1.1)
    load_or_build_index(s, embeddings)  # nothing changed -> reuse
    assert os.path.getmtime(meta) == first_mtime

    make_pdf("b.pdf", ["second paper about transformers"])
    store = load_or_build_index(s, embeddings)  # new PDF -> rebuild
    assert os.path.getmtime(meta) > first_mtime
    sources = {d.metadata["source_name"] for d in store.docstore._dict.values()}
    assert sources == {"a.pdf", "b.pdf"}


def test_no_pdfs_returns_none(tmp_path, embeddings):
    assert load_or_build_index(_settings(tmp_path), embeddings) is None


def test_normalized_embeddings_have_unit_length():
    import math

    from langchain_core.embeddings import Embeddings

    from researchrag.ingest import NormalizedEmbeddings

    class Raw(Embeddings):
        def embed_documents(self, texts):
            return [[3.0, 4.0] for _ in texts]

        def embed_query(self, text):
            return [0.0, 2.0]

    emb = NormalizedEmbeddings(Raw())
    assert emb.embed_documents(["a"]) == [[0.6, 0.8]]
    assert math.isclose(sum(v * v for v in emb.embed_query("q")), 1.0)
