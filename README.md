# Hybrid-Search RAG Pipeline with Relational Metadata Filtering
An enterprise-grade Retrieval-Augmented Generation (RAG) system built over a curated Wikipedia AI/ML corpus. This pipeline combines dense semantic vector search with sparse lexical BM25 keyword search via Reciprocal Rank Fusion (RRF), backed by a SQLite relational database for structured metadata pre-filtering before context is passed to Google Gemini.

### Live Interactive Demo: https://ragpipelinehybridsearch-y5aacuyanzkyjsynruyk7e.streamlit.app/

## System Architecture & Highlights
Standard vector search often misses exact keyword hits (such as domain acronyms or technical identifiers like BERT or ELIZA), while pure lexical search fails on conceptual queries. This pipeline resolves both limitations by fusing vector and keyword methods with structured SQL metadata constraints.

## Core Capabilities
Hybrid Retrieval (Dense + Sparse): Integrates Gemini vector embeddings stored in ChromaDB with an in-memory BM25 index.

Reciprocal Rank Fusion (RRF): Merges vector and keyword search ranks without requiring score normalization across disparate scales.

Relational Metadata Pre-Filtering: Uses an embedded SQLite database (metadata.db) to filter candidate articles on structured properties (e.g., minimum article length) prior to hybrid retrieval.

Production-Grade Ingestion Pipeline: Built with batch embedding, exponential backoff, automatic retry mechanisms, and state persistence to handle strict external API rate limits gracefully.

Dual Interface Options: Features both a web UI built with Streamlit and a terminal CLI for testing.

## Pipeline Architecture
```text
                    ┌────────────────────────────┐
                    │  Wikipedia Corpus Fetcher  │
                    └─────────────┬──────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌──────────────────────┐                       ┌──────────────────────┐
│ SQLite Relational DB │                       │ Document Chunker     │
│  (Article Metadata)  │                       │ (300 words / overlap)│
└────────┬─────────────┘                       └──────────┬───────────┘
         │                                                │
         │                        ┌───────────────────────┴───────────────────────┐
         │                        ▼                                               ▼
         │             ┌──────────────────────┐                        ┌──────────────────────┐
         │             │  Chroma Vector Store │                        │  BM25 Lexical Index  │
         │             │ (gemini-embedding-2) │                        │   (In-Memory Index)  │
         │             └──────────┬───────────┘                        └──────────┬───────────┘
         │                        │                                               │
         └──────────────────────┐ │ ┌─────────────────────────────────────────────┘
                                ▼ ▼ ▼
                     ┌────────────────────────┐
                     │ User Query + SQL Filter│
                     └──────────┬─────────────┘
                                │
                                ▼
                     ┌────────────────────────┐
                     │ Reciprocal Rank Fusion │
                     └──────────┬─────────────┘
                                │ (Top-K Chunks)
                                ▼
                     ┌────────────────────────┐
                     │   Gemini Generation    │
                     └────────────────────────┘
```

## Evaluation & Benchmark Results
The pipeline includes a automated evaluation suite (eval.py) testing top-5 retrieval accuracy across semantic, keyword, and acronym-focused queries.

### Search Methodology	Top-5 Hit Rate
BM25 Keyword Search	80.0%
Vector Similarity Search	100.0%
Hybrid Search (RRF)	100.0%
Key Takeaway: Combining BM25 keyword matching with semantic vector search improved top-5 retrieval accuracy by +20.0% over traditional lexical keyword search alone.

## Getting Started
### Prerequisites
1. Python 3.10+
2. A free Gemini API key from Google AI Studio

### Installation
Clone the repository:
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
Install dependencies:
pip install -r requirements.txt
Configure your API Key:
Create a .env file in the root directory:
GEMINI_API_KEY="actual-api-key"

## Running the Application
### 1. Data Ingestion & Indexing
Fetch articles, build the SQLite metadata database, and populate ChromaDB and BM25 indices:
python ingest.py

### 2. Streamlit Web Interface (Recommended)
Launch the interactive web UI with real-time relational filtering controls:
streamlit run streamlit_app.py

### 3. Command Line Interface
Run the simple CLI demo directly in your terminal:
python app.py

### 4. Run Retrieval Evaluation Benchmarks
Evaluate and generate benchmark numbers across vector, BM25, and hybrid retrieval:
python eval.py

## Repository Structure
```text
├── data/                  # Storage directory for chunks and SQLite metadata
│   ├── corpus.json        # Raw article storage
│   ├── chunks.json        # Pre-processed text chunks
│   └── metadata.db        # SQLite relational database
├── chroma_store/          # ChromaDB persistent vector database
├── config.py              # Central pipeline configuration and environment management
├── ingest.py              # MediaWiki ingestion, SQLite creation, and batch embedding
├── hybrid_search.py       # Hybrid search engine (BM25 + Vector + SQLite + RRF)
├── rag.py                 # Grounded context generation using Gemini
├── app.py                 # Interactive Terminal CLI interface
├── streamlit_app.py       # Interactive Streamlit Web UI application
├── eval.py                # Retrieval accuracy benchmarking suite
└── requirements.txt       # Project dependencies
```