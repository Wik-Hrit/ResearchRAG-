import hashlib
import math
import re
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class HashingEmbeddings(Embeddings):
    """Tiny normalised bag-of-words embedding: shared words give higher cosine similarity.

    Lets tests exercise real FAISS search without downloading a model.
    """

    dim = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z]+", text.lower()):
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def embeddings():
    return HashingEmbeddings()


@pytest.fixture
def make_pdf(tmp_path):
    """Write a small PDF with one paragraph per page."""
    canvas_mod = pytest.importorskip("reportlab.pdfgen.canvas")

    def _make(name, pages, folder=None):
        folder = Path(folder or tmp_path / "papers")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        c = canvas_mod.Canvas(str(path))
        for text in pages:
            y = 800
            for line in [text[i : i + 90] for i in range(0, len(text), 90)]:
                c.drawString(40, y, line)
                y -= 14
            c.showPage()
        c.save()
        return path

    return _make
