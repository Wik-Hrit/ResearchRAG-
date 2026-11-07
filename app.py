"""Command-line interface for ResearchRAG.

Run:  python app.py            (interactive)
      python app.py "question" (single question)
"""

import os
import sys

from researchrag.agent import ResearchAgent, make_llm
from researchrag.config import Settings
from researchrag.ingest import load_or_build_index, search_with_scores


def print_result(result: dict) -> None:
    print("\n--- Answer ---")
    print(result["answer"])
    if result.get("citations"):
        print("\n--- Sources ---")
        for i in result["citations"]:
            doc = result["relevant"][i - 1]
            page = int(doc.metadata.get("page", 0)) + 1
            print(f"  [{i}] {doc.metadata.get('source_name')} | page {page}")
    print("\n--- Agent trace ---")
    for step in result["trace"]:
        print(f"  -> {step}")
    print()


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Set GROQ_API_KEY in a .env file first.")

    settings = Settings()
    store = load_or_build_index(settings)
    if store is None:
        sys.exit(f"No PDFs found in '{settings.papers_dir}/'. Add research papers and rerun.")

    agent = ResearchAgent(
        llm=make_llm(settings),
        retrieve_fn=lambda q, k: search_with_scores(store, q, k),
        settings=settings,
    )

    if len(sys.argv) > 1:
        print_result(agent.ask(" ".join(sys.argv[1:])))
        return

    print("\nResearchRAG ready. Type your question (or 'exit' to quit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if question:
            print_result(agent.ask(question))


if __name__ == "__main__":
    main()
