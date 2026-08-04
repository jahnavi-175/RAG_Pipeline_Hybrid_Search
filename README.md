# RAG Pipeline with Hybrid Search

A retrieval-augmented generation system over a Wikipedia AI/ML corpus, combining
**vector similarity search** (semantic) with **BM25 keyword search** (lexical),
fused via Reciprocal Rank Fusion (RRF), and answered with Gemini.

## Why hybrid search?

Vector search alone misses exact keyword matches (e.g. acronyms, proper nouns
like "BERT" or "AlphaGo") when the embedding doesn't capture them well. BM25
alone misses semantically related content phrased differently. Combining both
catches more relevant chunks than either alone — see `eval_results.json` for
the measured improvement on this project's test set.

## Architecture

```
Wikipedia articles
      |
   chunking (300-word chunks, 50-word overlap)
      |
   Gemini embeddings ---------> ChromaDB (vector store)
      |
   BM25 index (keyword, built from same chunks)
      |
   query --> [vector search + BM25 search] --> Reciprocal Rank Fusion --> top-5 chunks
      |
   Gemini (gemini-1.5-flash) generates answer grounded in retrieved chunks
```

## Setup

1. Get a free Gemini API key: https://aistudio.google.com/apikey
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set your API key:
   ```
   export GEMINI_API_KEY="your-key-here"
   ```
   (or create a `.env` file with `GEMINI_API_KEY=your-key-here`)

## Usage

```
python ingest.py     # Day 1-3: fetch Wikipedia articles, chunk, embed, store
python app.py         # Day 6-7: interactive Q&A demo
python eval.py        # Day 7: benchmark hybrid vs vector-only retrieval
```

## 7-Day Build Plan

| Day | Task |
|-----|------|
| 1-2 | Run `ingest.py` fetching logic; verify articles pull cleanly, tune chunk size |
| 2-3 | Finish embedding + Chroma storage; confirm vector search returns sensible results |
| 4   | Build `hybrid_search.py` — BM25 index + RRF fusion; sanity-check ranked results |
| 5-6 | Build `rag.py` — prompt template, grounded generation, source citation |
| 6-7 | Build `app.py` CLI demo; run `eval.py` to get hit-rate numbers |
| 7   | Write this README, add architecture diagram, record a short demo GIF, push to GitHub |

## What to say about this project on your resume/LinkedIn

Don't just describe what it does — cite the eval numbers. Example:

> "Built a RAG pipeline with hybrid (vector + BM25) retrieval over a Wikipedia
> corpus, using Reciprocal Rank Fusion. Hybrid retrieval improved top-5 hit
> rate by X percentage points over vector-only search on a 10-question
> evaluation set (see eval_results.json)."

Fill in the actual X from your `eval_results.json` once you run `eval.py`.
BM25-only top-5 hit rate:    80.0%
Vector-only top-5 hit rate:  100.0%
Hybrid (RRF) top-5 hit rate: 100.0%
Improvement over BM25:       20.0%
Improvement over Vector:     0.0%

## Extending this for the internship application

If you want to go further to match "high volumes of data" / "relational DB"
from the eligibility criteria, a natural extension (not required for the 7-day
version) is to also index structured metadata in a small SQLite table
(e.g. article length, category, last-updated) and let queries filter on it
before retrieval — showing you can combine relational and vector data sources.