"""Agentic RAG with LangGraph (corrective RAG pattern).

Instead of always doing retrieve -> generate, the agent decides its next step:

    retrieve -> grade_documents --relevant-------> generate -> check_grounding --grounded--> END
                     |                                 ^              |
                     | none relevant                   +--regenerate--+ (not grounded)
                     v                                                |
                rewrite_query -> retrieve                     limit reached
                     |                                                |
                     +-- rewrite limit reached --> refuse <-----------+

Every decision is written to `trace`, so the UI can show why the agent answered or refused.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from .config import Settings

# (query, k) -> list of (chunk, similarity score)
RetrieveFn = Callable[[str, int], list[tuple[Document, float]]]

REFUSAL = (
    "I couldn't find a well-supported answer to this in the provided papers. "
    "Try rephrasing the question or adding a relevant paper."
)
NOT_FOUND = "NOT_FOUND"

GRADE_PROMPT = """You are grading whether a retrieved passage is useful for answering a question.

Question: {question}

Passage:
{passage}

Does the passage contain any information relevant to answering the question, even partially?
Reply with exactly one word: yes or no."""

REWRITE_PROMPT = """The question below did not retrieve useful passages from a collection of research papers.
Rewrite it as a short keyword-style search query (5 to 12 words) built from specific technical
terms likely to appear in the paper text itself. Do not use words like "retrieve", "find" or "paper".
If the question asks about "this paper" in general, use terms such as abstract, we propose,
contribution, method, problem.

Original question: {question}
Previous query: {query}

Reply with only the rewritten query."""

GENERATE_PROMPT = """You are a research assistant. Answer the question using ONLY the numbered passages.

Rules:
- Every factual sentence must end with the passage number(s) it comes from, like [1] or [2][3].
- Do not use outside knowledge.
- If the passages do not contain the answer, reply with exactly: {not_found}
{feedback}
Passages:
{context}

Question: {question}

Answer:"""

GROUNDING_PROMPT = """You are checking an answer for hallucinations.

Passages:
{context}

Answer:
{answer}

Is every claim in the answer supported by the passages above?
Reply with exactly one word: yes or no."""


class AgentState(TypedDict, total=False):
    question: str
    query: str
    retrieved: list[tuple[Document, float]]
    relevant: list[Document]
    answer: str
    citations: list[int]
    rewrites: int
    generations: int
    grounded: bool
    status: Literal["answered", "refused"]
    trace: list[str]


def _text(response: Any) -> str:
    """Chat models return a message object; plain callables may return a string.

    Reasoning models (e.g. Qwen3) may wrap their thinking in <think> tags; strip it so
    yes/no decisions and answers only contain the final output.
    """
    text = getattr(response, "content", response) or ""
    if isinstance(text, list):  # some providers return content blocks
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _is_yes(text: str) -> bool:
    """True if the first word is 'yes', ignoring markdown or punctuation like **Yes.**"""
    match = re.search(r"[a-z]+", text.lower())
    return bool(match) and match.group() == "yes"


def _numbered_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source_name") or doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] ({src}, page {page})\n{doc.page_content}")
    return "\n\n".join(parts)


def parse_citations(answer: str, n_docs: int) -> list[int]:
    """Extract valid passage numbers like [1] or [2][3] from the answer."""
    found = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    return sorted(i for i in found if 1 <= i <= n_docs)


class ResearchAgent:
    def __init__(self, llm: Any, retrieve_fn: RetrieveFn, settings: Settings | None = None):
        self.llm = llm
        self.retrieve_fn = retrieve_fn
        self.settings = settings or Settings()
        self.graph = self._build_graph()

    # ---------- nodes ----------

    def retrieve(self, state: AgentState) -> AgentState:
        query = state.get("query") or state["question"]
        results = self.retrieve_fn(query, self.settings.top_k)
        kept = [(d, s) for d, s in results if s >= self.settings.similarity_threshold]
        dropped = len(results) - len(kept)
        msg = f"Retrieved {len(results)} chunks for '{query}'"
        if dropped:
            msg += f", dropped {dropped} below similarity {self.settings.similarity_threshold:.2f}"
        return {"query": query, "retrieved": kept, "trace": state.get("trace", []) + [msg]}

    def grade_documents(self, state: AgentState) -> AgentState:
        relevant = []
        for doc, _score in state.get("retrieved", []):
            verdict = _text(
                self.llm.invoke(
                    GRADE_PROMPT.format(question=state["question"], passage=doc.page_content)
                )
            )
            if _is_yes(verdict):
                relevant.append(doc)
        total = len(state.get("retrieved", []))
        msg = f"Graded chunks: {len(relevant)}/{total} relevant"
        return {"relevant": relevant, "trace": state["trace"] + [msg]}

    def rewrite_query(self, state: AgentState) -> AgentState:
        new_query = _text(
            self.llm.invoke(REWRITE_PROMPT.format(question=state["question"], query=state["query"]))
        ).strip('"')
        rewrites = state.get("rewrites", 0) + 1
        msg = f"Rewrote query (attempt {rewrites}): '{new_query}'"
        return {"query": new_query, "rewrites": rewrites, "trace": state["trace"] + [msg]}

    def generate(self, state: AgentState) -> AgentState:
        docs = state["relevant"]
        feedback = ""
        if state.get("generations", 0) > 0:
            feedback = (
                "\nYour previous answer contained claims not supported by the passages. "
                "Only state what the passages say.\n"
            )
        answer = _text(
            self.llm.invoke(
                GENERATE_PROMPT.format(
                    not_found=NOT_FOUND,
                    feedback=feedback,
                    context=_numbered_context(docs),
                    question=state["question"],
                )
            )
        )
        generations = state.get("generations", 0) + 1
        citations = parse_citations(answer, len(docs))
        msg = f"Generated answer (attempt {generations}) citing passages {citations or 'none'}"
        return {
            "answer": answer,
            "citations": citations,
            "generations": generations,
            "trace": state["trace"] + [msg],
        }

    def check_grounding(self, state: AgentState) -> AgentState:
        answer = state["answer"]
        if NOT_FOUND in answer:
            return {"grounded": False, "trace": state["trace"] + ["Model said the passages lack the answer"]}
        if not state.get("citations"):
            return {"grounded": False, "trace": state["trace"] + ["Grounding failed: answer has no citations"]}
        cited = [state["relevant"][i - 1] for i in state["citations"]]
        verdict = _text(
            self.llm.invoke(GROUNDING_PROMPT.format(context=_numbered_context(cited), answer=answer))
        )
        grounded = _is_yes(verdict)
        msg = "Grounding check passed" if grounded else "Grounding check failed: unsupported claims"
        return {"grounded": grounded, "trace": state["trace"] + [msg]}

    def finish(self, state: AgentState) -> AgentState:
        return {"status": "answered", "trace": state["trace"] + ["Returned grounded answer"]}

    def refuse(self, state: AgentState) -> AgentState:
        return {
            "status": "refused",
            "answer": REFUSAL,
            "citations": [],
            "trace": state["trace"] + ["Refused: no well-supported answer within retry limits"],
        }

    # ---------- routing ----------

    def route_after_grading(self, state: AgentState) -> str:
        if state.get("relevant"):
            return "generate"
        if state.get("rewrites", 0) < self.settings.max_rewrites:
            return "rewrite_query"
        return "refuse"

    def route_after_check(self, state: AgentState) -> str:
        if state.get("grounded"):
            return "finish"
        if NOT_FOUND in state.get("answer", ""):
            return "refuse"
        if state.get("generations", 0) < self.settings.max_generations:
            return "generate"
        return "refuse"

    # ---------- graph ----------

    def _build_graph(self):
        g = StateGraph(AgentState)
        for name in (
            "retrieve",
            "grade_documents",
            "rewrite_query",
            "generate",
            "check_grounding",
            "finish",
            "refuse",
        ):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade_documents")
        g.add_conditional_edges(
            "grade_documents",
            self.route_after_grading,
            {"generate": "generate", "rewrite_query": "rewrite_query", "refuse": "refuse"},
        )
        g.add_edge("rewrite_query", "retrieve")
        g.add_edge("generate", "check_grounding")
        g.add_conditional_edges(
            "check_grounding",
            self.route_after_check,
            {"finish": "finish", "generate": "generate", "refuse": "refuse"},
        )
        g.add_edge("finish", END)
        g.add_edge("refuse", END)
        return g.compile()

    def ask(self, question: str) -> AgentState:
        return self.graph.invoke(
            {"question": question, "rewrites": 0, "generations": 0, "trace": []}
        )


def make_llm(settings: Settings):
    """Groq-hosted LLM. Kept separate so the agent can run with any chat model."""
    from langchain_groq import ChatGroq

    return ChatGroq(model=settings.llm_model, temperature=settings.temperature)
