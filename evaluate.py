"""Retrieval evaluation across chunk sizes.

For each chunk size, builds a fresh index and checks whether the chunk containing the
expected answer (source PDF + page) appears in the top-k results.

Metrics:
  hit@k  share of questions where a correct chunk is in the top k
  MRR    mean reciprocal rank of the first correct chunk (1.0 = always ranked first)

Run:  python evaluate.py --questions eval/questions.json --sizes 500 1000 1500
"""

import argparse
import json
import os
from statistics import mean

from researchrag.config import Settings
from researchrag.ingest import (
    build_index,
    chunk_documents,
    get_embeddings,
    load_pdfs,
    search_with_scores,
)


def evaluate(pages, questions, embeddings, chunk_size, overlap_ratio, k):
    overlap = int(chunk_size * overlap_ratio)
    chunks = chunk_documents(pages, chunk_size, overlap)
    store = build_index(chunks, embeddings)
    hits, rranks = [], []
    for q in questions:
        results = search_with_scores(store, q["question"], k)
        rank = next(
            (
                i
                for i, (doc, _) in enumerate(results, start=1)
                if os.path.basename(doc.metadata.get("source", "")) == q["source"]
                # questions.json uses human page numbers (1-based); PyPDF pages are 0-based
                and int(doc.metadata.get("page", -1)) + 1 == int(q["page"])
            ),
            None,
        )
        hits.append(rank is not None)
        rranks.append(1 / rank if rank else 0.0)
    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": len(chunks),
        "hit_at_k": mean(hits),
        "mrr": mean(rranks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 1500])
    parser.add_argument("--overlap-ratio", type=float, default=0.15)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    settings = Settings()
    questions = json.load(open(args.questions))
    pages = load_pdfs(settings.papers_dir)
    if not pages:
        raise SystemExit(f"No PDFs in '{settings.papers_dir}/'.")
    embeddings = get_embeddings(settings)

    rows = [evaluate(pages, questions, embeddings, s, args.overlap_ratio, args.k) for s in args.sizes]

    print(f"\n{len(questions)} questions, k={args.k}, embedding={settings.embedding_model}\n")
    print(f"| Chunk size | Overlap | Chunks | Hit@{args.k} | MRR |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['chunk_size']} | {r['overlap']} | {r['chunks']} | {r['hit_at_k']:.2f} | {r['mrr']:.2f} |")
    best = max(rows, key=lambda r: (r["hit_at_k"], r["mrr"]))
    print(f"\nBest: chunk size {best['chunk_size']} (hit@{args.k} {best['hit_at_k']:.2f}, MRR {best['mrr']:.2f})")


if __name__ == "__main__":
    main()
