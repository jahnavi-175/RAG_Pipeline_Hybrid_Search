"""
Day 7: Benchmark hybrid retrieval vs vector-only retrieval.

This produces the "measurable insight" you should put in your README / resume
instead of just a screenshot - e.g. "hybrid retrieval improved top-5 hit rate
by X% over vector-only search on a 20-question test set."

Run:
    python eval.py
"""
import config
import json
from hybrid_search import HybridSearcher

# A hand-labeled test set: question -> the expected Wikipedia article title(s).
# Expands coverage across different query styles (semantic, keyword-dense, acronyms).
TEST_SET = [
    {"question": "Who created ELIZA?", "expected_title": "ELIZA"},
    {"question": "What is backpropagation used for?", "expected_title": "Backpropagation"},
    {"question": "What game did AlphaGo play?", "expected_title": "AlphaGo"},
    {"question": "What is a transformer architecture used for?",
     "expected_title": "Transformer (deep learning)"},
    {"question": "What does BERT stand for and what is it used for?",
     "expected_title": "BERT (language model)"},
    {"question": "What is the purpose of a vector database?", "expected_title": "Vector database"},
    {"question": "What is retrieval-augmented generation?",
     "expected_title": "Retrieval-augmented generation"},
    {"question": "What is overfitting in machine learning?", "expected_title": "Overfitting"},
    {"question": "How does k-means clustering work?", "expected_title": "K-means clustering"},
    {"question": "What is a decision tree?", "expected_title": "Decision tree learning"},
    {"question": "What caused the AI winter?", "expected_title": "AI winter"},
    {"question": "How does gradient descent optimize neural networks?", "expected_title": "Gradient descent"},
    {"question": "What is the Turing test?", "expected_title": "Turing test"},
    {"question": "How do generative adversarial networks work?", "expected_title": "Generative adversarial network"},
    {"question": "What is prompt engineering?", "expected_title": "Prompt engineering"},
]

def hit_rate(searcher, mode="hybrid", k=5):
    """Calculate top-k hit rate for 'vector', 'bm25', or 'hybrid' search."""
    hits = 0
    for item in TEST_SET:
        expected = item["expected_title"]
        if mode == "vector":
            ids = searcher._vector_search(item["question"], k=k)
            titles = [searcher.id_to_chunk[i]["title"] for i in ids]
        elif mode == "bm25":
            ids = searcher._bm25_search(item["question"], k=k)
            titles = [searcher.id_to_chunk[i]["title"] for i in ids]
        else:
            chunks = searcher.search(item["question"], k=k)
            titles = [c["title"] for c in chunks]

        if expected in titles:
            hits += 1
    return hits / len(TEST_SET)

def main():
    searcher = HybridSearcher()

    vector_only_rate = hit_rate(searcher, mode="vector")
    bm25_only_rate = hit_rate(searcher, mode="bm25")
    hybrid_rate = hit_rate(searcher, mode="hybrid")

    print(f"BM25-only top-5 hit rate:    {bm25_only_rate:.1%}")
    print(f"Vector-only top-5 hit rate:  {vector_only_rate:.1%}")
    print(f"Hybrid (RRF) top-5 hit rate: {hybrid_rate:.1%}")
    print(f"Improvement over BM25:       {(hybrid_rate - bm25_only_rate):.1%}")
    print(f"Improvement over Vector:     {(hybrid_rate - vector_only_rate):.1%}")

    with open("eval_results.json", "w") as f:
        json.dump({
            "bm25_only_hit_rate": bm25_only_rate,
            "vector_only_hit_rate": vector_only_rate,
            "hybrid_hit_rate": hybrid_rate,
            "test_set_size": len(TEST_SET),
        }, f, indent=2)
    print("\nSaved results to eval_results.json - use these numbers in your README/resume.")

if __name__ == "__main__":
    main()