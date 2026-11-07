"""Agent routing tests with a scripted fake LLM and fake retriever (no API key or model download)."""

from collections import deque

from langchain_core.documents import Document

from researchrag.agent import NOT_FOUND, REFUSAL, ResearchAgent, parse_citations
from researchrag.config import Settings

ESN_DOC = Document(
    page_content="Echo state networks train only the readout layer; the reservoir weights stay fixed.",
    metadata={"source_name": "reservoir.pdf", "page": 3},
)
BPTT_DOC = Document(
    page_content="Backpropagation through time suffers from vanishing gradients on long sequences.",
    metadata={"source_name": "rnn.pdf", "page": 7},
)
NOISE_DOC = Document(
    page_content="The conference was held in Vienna in July.",
    metadata={"source_name": "misc.pdf", "page": 1},
)


class ScriptedLLM:
    """Returns queued replies per prompt type, so each test controls every decision."""

    def __init__(self, grade=(), rewrite=(), generate=(), ground=()):
        self.queues = {
            "grade": deque(grade),
            "rewrite": deque(rewrite),
            "generate": deque(generate),
            "ground": deque(ground),
        }
        self.calls = []

    def invoke(self, prompt: str) -> str:
        if "grading whether a retrieved passage" in prompt:
            kind = "grade"
        elif "keyword-style search query" in prompt:
            kind = "rewrite"
        elif "checking an answer for hallucinations" in prompt:
            kind = "ground"
        else:
            kind = "generate"
        self.calls.append(kind)
        return self.queues[kind].popleft()


def retriever(results_by_query):
    def fn(query, k):
        return results_by_query.get(query, results_by_query.get("*", []))[:k]

    return fn


def settings(**overrides):
    s = Settings()
    s.top_k = 4
    s.similarity_threshold = 0.3
    s.max_rewrites = 2
    s.max_generations = 2
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def test_happy_path_answers_with_citations():
    llm = ScriptedLLM(
        grade=["yes", "no"],
        generate=["ESNs train only the readout layer [1]."],
        ground=["yes"],
    )
    agent = ResearchAgent(llm, retriever({"*": [(ESN_DOC, 0.8), (NOISE_DOC, 0.5)]}), settings())
    out = agent.ask("How are echo state networks trained?")

    assert out["status"] == "answered"
    assert out["citations"] == [1]
    assert out["relevant"] == [ESN_DOC]
    assert "rewrite" not in llm.calls


def test_threshold_drops_weak_chunks_before_grading():
    llm = ScriptedLLM(grade=["yes"], generate=["Answer [1]."], ground=["yes"])
    agent = ResearchAgent(llm, retriever({"*": [(ESN_DOC, 0.8), (NOISE_DOC, 0.1)]}), settings())
    out = agent.ask("q")

    assert llm.calls.count("grade") == 1  # the 0.1 chunk never reached the grader
    assert any("dropped 1 below similarity" in t for t in out["trace"])


def test_rewrites_query_when_nothing_relevant_then_answers():
    llm = ScriptedLLM(
        grade=["no", "yes"],
        rewrite=["backpropagation through time vanishing gradient"],
        generate=["BPTT has vanishing gradients [1]."],
        ground=["yes"],
    )
    results = {
        "Why do RNNs struggle?": [(NOISE_DOC, 0.4)],
        "backpropagation through time vanishing gradient": [(BPTT_DOC, 0.9)],
    }
    out = ResearchAgent(llm, retriever(results), settings()).ask("Why do RNNs struggle?")

    assert out["status"] == "answered"
    assert out["rewrites"] == 1
    assert out["query"] == "backpropagation through time vanishing gradient"


def test_refuses_after_max_rewrites():
    llm = ScriptedLLM(grade=["no"] * 3, rewrite=["q2", "q3"])
    out = ResearchAgent(llm, retriever({"*": [(NOISE_DOC, 0.5)]}), settings()).ask("q1")

    assert out["status"] == "refused"
    assert out["answer"] == REFUSAL
    assert out["rewrites"] == 2
    assert "generate" not in llm.calls


def test_regenerates_once_when_not_grounded_then_passes():
    llm = ScriptedLLM(
        grade=["yes"],
        generate=["ESNs were invented in 1850 [1].", "ESNs train only the readout [1]."],
        ground=["no", "yes"],
    )
    out = ResearchAgent(llm, retriever({"*": [(ESN_DOC, 0.9)]}), settings()).ask("q")

    assert out["status"] == "answered"
    assert out["generations"] == 2
    assert llm.calls.count("ground") == 2


def test_refuses_when_never_grounded():
    llm = ScriptedLLM(grade=["yes"], generate=["Made up [1].", "Still made up [1]."], ground=["no", "no"])
    out = ResearchAgent(llm, retriever({"*": [(ESN_DOC, 0.9)]}), settings()).ask("q")

    assert out["status"] == "refused"
    assert out["generations"] == 2


def test_answer_without_citations_is_not_accepted():
    llm = ScriptedLLM(grade=["yes"], generate=["No citations here.", "Cited now [1]."], ground=["yes"])
    out = ResearchAgent(llm, retriever({"*": [(ESN_DOC, 0.9)]}), settings()).ask("q")

    assert out["status"] == "answered"
    assert llm.calls.count("ground") == 1  # uncited answer was rejected without an LLM call


def test_model_saying_not_found_leads_to_refusal():
    llm = ScriptedLLM(grade=["yes"], generate=[NOT_FOUND])
    out = ResearchAgent(llm, retriever({"*": [(ESN_DOC, 0.9)]}), settings()).ask("q")

    assert out["status"] == "refused"
    assert out["generations"] == 1


def test_no_retrieval_results_goes_straight_to_rewrite():
    llm = ScriptedLLM(rewrite=["q2", "q3"])
    out = ResearchAgent(llm, retriever({}), settings()).ask("q1")

    assert out["status"] == "refused"
    assert "grade" not in llm.calls


def test_parse_citations_ignores_out_of_range_and_duplicates():
    assert parse_citations("A [1]. B [2][1]. C [9].", n_docs=2) == [1, 2]
    assert parse_citations("nothing cited", n_docs=3) == []


def test_reasoning_tags_are_stripped_from_decisions():
    llm = ScriptedLLM(
        grade=["<think>The passage mentions readout training.</think>\nyes"],
        generate=["<think>draft</think>ESNs train only the readout [1]."],
        ground=["<think>check</think> yes"],
    )
    out = ResearchAgent(llm, retriever({"*": [(ESN_DOC, 0.9)]}), settings()).ask("q")

    assert out["status"] == "answered"
    assert "<think>" not in out["answer"]


def test_yes_parsing_handles_markdown_and_case():
    from researchrag.agent import _is_yes

    assert _is_yes("**Yes.**")
    assert _is_yes("  yes")
    assert not _is_yes("No, it does not.")
    assert not _is_yes("")
