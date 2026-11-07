from pathlib import Path

import pytest

from evaluate import evaluate
from researchrag.ingest import load_pdfs

ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_reports_hit_rate_and_mrr(tmp_path, embeddings, make_pdf):
    make_pdf("rc.pdf", [
        "Echo state networks keep reservoir weights fixed and train only the readout layer.",
        "Backpropagation through time has vanishing gradients on long sequences.",
    ])
    pages = load_pdfs(str(tmp_path / "papers"))
    questions = [
        {"question": "which layer do echo state networks train", "source": "rc.pdf", "page": 1},
        {"question": "vanishing gradients backpropagation through time", "source": "rc.pdf", "page": 2},
    ]
    row = evaluate(pages, questions, embeddings, chunk_size=300, overlap_ratio=0.1, k=1)

    assert row["hit_at_k"] == 1.0
    assert row["mrr"] == 1.0
    assert row["overlap"] == 30


class AlwaysGroundedLLM:
    def invoke(self, prompt):
        if "grading whether" in prompt or "checking an answer" in prompt:
            return "yes"
        return "Echo state networks train only the readout layer [1]."


def test_streamlit_app_answers_end_to_end(tmp_path, monkeypatch, embeddings, make_pdf):
    testing = pytest.importorskip("streamlit.testing.v1")
    import researchrag.agent as agent_mod
    import researchrag.ingest as ingest_mod

    make_pdf("rc.pdf", ["Echo state networks train only the readout layer."], folder=tmp_path / "papers")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(agent_mod, "make_llm", lambda s: AlwaysGroundedLLM())
    monkeypatch.setattr(ingest_mod, "get_embeddings", lambda s: embeddings)

    at = testing.AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("How are echo state networks trained?").run()

    assert not at.exception
    assert any("readout layer [1]" in m.value for m in at.markdown)
    assert any(e.label.startswith("Sources") for e in at.expander)
    assert any(e.label == "Agent trace" for e in at.expander)


def _patched_app(monkeypatch, embeddings, tmp_path, **env):
    testing = pytest.importorskip("streamlit.testing.v1")
    import researchrag.agent as agent_mod
    import researchrag.ingest as ingest_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "test")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(agent_mod, "make_llm", lambda s: AlwaysGroundedLLM())
    monkeypatch.setattr(ingest_mod, "get_embeddings", lambda s: embeddings)
    return testing.AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60)


def test_password_gate_blocks_until_correct(tmp_path, monkeypatch, embeddings):
    at = _patched_app(monkeypatch, embeddings, tmp_path, APP_PASSWORD="s3cret")
    at.run()
    assert len(at.chat_input) == 0  # app content hidden

    at.text_input[0].set_value("wrong").run()
    assert any("Wrong password" in e.value for e in at.error)
    assert len(at.chat_input) == 0

    at.text_input[0].set_value("s3cret").run()
    assert at.session_state["authed"] is True


def test_public_mode_isolates_papers_and_limits_questions(tmp_path, monkeypatch, embeddings, make_pdf):
    # A shared papers/ folder exists, but a public visitor must not see it.
    make_pdf("shared.pdf", ["Shared secret document about reservoirs."], folder=tmp_path / "papers")
    at = _patched_app(monkeypatch, embeddings, tmp_path, PUBLIC_MODE="1", MAX_QUESTIONS_PER_SESSION="1")
    at.run()
    assert any("Upload at least one PDF" in i.value for i in at.info)

    # Visitor's own upload (simulated by writing into their private folder)
    workdir = at.session_state["workdir"]
    make_pdf("mine.pdf", ["Echo state networks train only the readout layer."], folder=f"{workdir}/papers")
    at.run()
    at.chat_input[0].set_value("How are echo state networks trained?").run()
    assert any("readout layer [1]" in m.value for m in at.markdown)
    assert any("Demo limit of 1 questions" in w.value for w in at.warning)
    assert at.chat_input[0].disabled
