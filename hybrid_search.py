import sqlite3
import config

import json
import os

import time

import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from rank_bm25 import BM25Okapi

genai.configure(api_key=config.GEMINI_API_KEY)


class HybridSearcher:
    def __init__(self):
        # Vector store
        client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_or_create_collection("wiki_chunks", embedding_function=None)

        # BM25 keyword index, built from the same chunks
        with open("data/chunks.json") as f:
            self.chunks = json.load(f)
        self.id_to_chunk = {c["id"]: c for c in self.chunks}
        tokenized_corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _embed_query(self, query):
        for attempt in range(5):
            try:
                result = genai.embed_content(
                    model=config.EMBEDDING_MODEL,
                    content=query,
                    task_type="retrieval_query",
                    output_dimensionality=config.EMBEDDING_DIMENSIONS,
                )
                return result["embedding"]
            except Exception as e:
                msg = str(e)
                if "429" in msg or "quota" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
                    wait = 60 if attempt > 0 else 5
                    print(f"  Embedding rate limit hit, waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to embed query after retries: {query[:50]}")

    def _get_allowed_titles(self, min_words=0):
        """Query SQLite to get titles that match our relational filter."""
        with sqlite3.connect("data/metadata.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT title FROM articles WHERE word_count >= ?", (min_words,))
            return [row[0] for row in cursor.fetchall()]
        
    def _vector_search(self, query, k=config.TOP_K_VECTOR, allowed_titles=None):
        query_embedding = self._embed_query(query)
        
        # Build the metadata filter for ChromaDB
        where_filter = None
        if allowed_titles is not None:
            if len(allowed_titles) == 0:
                return [] # No articles met the relational criteria
            elif len(allowed_titles) == 1:
                where_filter = {"title": allowed_titles[0]}
            else:
                where_filter = {"title": {"$in": allowed_titles}}

        results = self.collection.query(
            query_embeddings=[query_embedding], 
            n_results=k,
            where=where_filter
        )
        return results["ids"][0] if results["ids"] else []

    def _bm25_search(self, query, k=config.TOP_K_BM25, allowed_titles=None):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Zero out scores for chunks that don't belong to allowed titles
        if allowed_titles is not None:
            for i, chunk in enumerate(self.chunks):
                if chunk["title"] not in allowed_titles:
                    scores[i] = 0.0

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        # Only return IDs that actually have a score > 0
        return [self.chunks[i]["id"] for i in ranked_indices if scores[i] > 0]

    def search(self, query, k=config.TOP_K_FINAL, rrf_k=60, min_words=0):
        allowed_titles = self._get_allowed_titles(min_words=min_words)
        
        candidate_k = max(k * 2, config.TOP_K_VECTOR)
        vector_ids = self._vector_search(query, k=candidate_k, allowed_titles=allowed_titles)
        bm25_ids = self._bm25_search(query, k=candidate_k, allowed_titles=allowed_titles)

        fused_scores = {}
        for rank, doc_id in enumerate(vector_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rrf_k + rank)
        for rank, doc_id in enumerate(bm25_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rrf_k + rank)

        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:k]
        return [self.id_to_chunk[doc_id] for doc_id in ranked_ids]

if __name__ == "__main__":
    searcher = HybridSearcher()
    query = "How does attention work in transformers?"
    results = searcher.search(query)
    print(f"Query: {query}\n")
    for r in results:
        print(f"- [{r['title']}] {r['text'][:120]}...")