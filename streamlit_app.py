"""Streamlit UI for the agentic ResearchRAG.

Run:  streamlit run streamlit_app.py
"""

import hmac
import os
import tempfile
from pathlib import Path

import streamlit as st

from researchrag.agent import ResearchAgent, make_llm
from researchrag.config import Settings
from researchrag.ingest import fingerprint, get_embeddings, load_or_build_index, search_with_scores

st.set_page_config(page_title="ResearchRAG", page_icon="📚", layout="wide")
st.title("📚 ResearchRAG")
st.caption("Agentic RAG over research papers: grades what it retrieves, rewrites weak queries, "
           "checks its answer is grounded, and refuses when the papers don't support an answer.")

# ---------- public deployment safeguards ----------
# On Streamlit Cloud, set these under App settings > Secrets (they become env vars).
PUBLIC_MODE = os.getenv("PUBLIC_MODE") == "1"          # per-visitor private uploads + limits
APP_PASSWORD = os.getenv("APP_PASSWORD")                # optional password gate
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "15"))
MAX_FILES, MAX_FILE_MB = 3, 10

if APP_PASSWORD and not st.session_state.get("authed"):
    entered = st.text_input("Password", type="password")
    if entered and hmac.compare_digest(entered, APP_PASSWORD):
        st.session_state.authed = True
        st.rerun()
    elif entered:
        st.error("Wrong password")
    st.stop()

settings = Settings()
if PUBLIC_MODE:
    # Each visitor gets their own temporary folder, so nobody sees anyone else's PDFs.
    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.mkdtemp(prefix="researchrag_")
    settings.papers_dir = os.path.join(st.session_state.workdir, "papers")
    settings.index_dir = os.path.join(st.session_state.workdir, "faiss_index")
    os.makedirs(settings.papers_dir, exist_ok=True)

# ---------- sidebar: settings and papers ----------
with st.sidebar:
    st.header("Retrieval settings")
    settings.chunk_size = st.number_input("Chunk size (characters)", 200, 4000, settings.chunk_size, 100)
    settings.chunk_overlap = st.number_input("Chunk overlap", 0, 1000, settings.chunk_overlap, 25)
    settings.top_k = st.slider("Top-k chunks", 1, 10, settings.top_k)
    settings.similarity_threshold = st.slider(
        "Similarity threshold", 0.0, 1.0, settings.similarity_threshold, 0.05,
        help="Chunks below this cosine similarity are dropped before the agent sees them.",
    )
    settings.max_rewrites = st.slider("Max query rewrites", 0, 3, settings.max_rewrites)

    st.header("Papers")
    uploads = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
    if uploads:
        os.makedirs(settings.papers_dir, exist_ok=True)
        saved = 0
        if PUBLIC_MODE:
            too_big = [f.name for f in uploads if f.size > MAX_FILE_MB * 1024 * 1024]
            if too_big:
                st.error(f"Skipped files over {MAX_FILE_MB} MB: {', '.join(too_big)}")
            uploads = [f for f in uploads if f.size <= MAX_FILE_MB * 1024 * 1024][:MAX_FILES]
        for f in uploads:
            target = Path(settings.papers_dir) / f.name
            # Streamlit reruns this script on every interaction; only write new or changed
            # files, otherwise the file timestamps change and the index rebuilds each time.
            if not target.exists() or target.stat().st_size != f.size:
                target.write_bytes(f.getbuffer())
                saved += 1
        if saved:
            st.success(f"Saved {saved} new file(s)")
    papers = sorted(p.name for p in Path(settings.papers_dir).glob("*.pdf")) if Path(settings.papers_dir).exists() else []
    st.write(f"{len(papers)} paper(s) indexed:" if papers else "No papers yet.")
    for name in papers:
        st.caption(f"• {name}")
    if st.button("Clear chat"):
        st.session_state.messages = []

if not os.getenv("GROQ_API_KEY"):
    st.error("Set GROQ_API_KEY in a .env file to use the assistant.")
    st.stop()


@st.cache_resource(show_spinner=False)
def _embeddings(model_name: str):
    s = Settings()
    s.embedding_model = model_name
    return get_embeddings(s)


@st.cache_resource(show_spinner="Indexing papers...", max_entries=20, ttl=3600)
def _index(fp: str, chunk_size: int, chunk_overlap: int, papers_dir: str, index_dir: str):
    # `fp` is part of the cache key, so the index is rebuilt whenever papers or settings change.
    s = Settings()
    s.chunk_size, s.chunk_overlap = chunk_size, chunk_overlap
    s.papers_dir, s.index_dir = papers_dir, index_dir
    return load_or_build_index(s, _embeddings(s.embedding_model))


store = _index(
    fingerprint(settings.papers_dir, settings),
    settings.chunk_size,
    settings.chunk_overlap,
    settings.papers_dir,
    settings.index_dir,
)
if store is None:
    st.info("Upload at least one PDF in the sidebar to start.")
    st.stop()

agent = ResearchAgent(
    llm=make_llm(settings),
    retrieve_fn=lambda q, k: search_with_scores(store, q, k),
    settings=settings,
)


def render_result(result: dict):
    """Answer, status, cited sources and the agent's decision trace."""
    if result["status"] == "refused":
        st.warning(result["answer"])
    else:
        st.markdown(result["answer"])
    if result.get("citations"):
        with st.expander(f"Sources ({len(result['citations'])})"):
            for i in result["citations"]:
                doc = result["relevant"][i - 1]
                st.markdown(f"**[{i}] {doc.metadata.get('source_name')}, page "
                            f"{int(doc.metadata.get('page', 0)) + 1}**")
                st.caption(doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))
    with st.expander("Agent trace"):
        for step in result["trace"]:
            st.write(f"→ {step}")


# ---------- chat with session history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(msg["result"])

st.session_state.setdefault("questions_asked", 0)
limit_reached = PUBLIC_MODE and st.session_state.questions_asked >= MAX_QUESTIONS
if limit_reached:
    st.warning(f"Demo limit of {MAX_QUESTIONS} questions reached for this session. "
               "Clone the repo from GitHub to run it without limits.")

if question := st.chat_input("Ask a question about your papers", disabled=limit_reached):
    st.session_state.questions_asked += 1
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.ask(question)
        render_result(result)
    st.session_state.messages.append({"role": "assistant", "result": result})
    if PUBLIC_MODE and st.session_state.questions_asked >= MAX_QUESTIONS:
        st.rerun()  # show the limit message and disable the input straight away
