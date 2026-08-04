import sqlite3
import config

import json
import os
import time

import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import requests
from tqdm import tqdm

genai.configure(api_key=config.GEMINI_API_KEY)

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "RAG-Hybrid-Search-Project/1.0 (student project)"}


def fetch_one_article(title):
    """
    Fetch plain-text article content directly from the MediaWiki API.
    This avoids the third-party `wikipedia` PyPI package, which has known
    reliability issues (silent failures, disambiguation-handling bugs) on
    recent Python/BeautifulSoup versions.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "redirects": 1,
    }
    resp = requests.get(WIKI_API_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)
    if page is None or "missing" in page:
        return None

    content = page.get("extract", "")
    if not content or len(content.split()) < 30:
        return None  # too short to be useful (likely a stub or redirect issue)

    resolved_title = page["title"]
    url = f"https://en.wikipedia.org/wiki/{resolved_title.replace(' ', '_')}"
    return {"title": resolved_title, "content": content, "url": url}


def fetch_articles(topics):
    """Pull plain-text Wikipedia article content for each topic, with basic retry."""
    articles = []
    for topic in tqdm(topics, desc="Fetching Wikipedia articles"):
        article = None
        for attempt in range(4):
            try:
                article = fetch_one_article(topic)
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    # Respect Retry-After if Wikipedia sends one, else back off harder each attempt
                    wait = int(e.response.headers.get("Retry-After", 5 * (attempt + 1)))
                    print(f"  rate limited on '{topic}', waiting {wait}s (attempt {attempt+1}/4)")
                    time.sleep(wait)
                else:
                    print(f"  retry {attempt+1}/4 for '{topic}': {e}")
                    time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                print(f"  retry {attempt+1}/4 for '{topic}': {e}")
                time.sleep(2 ** attempt)
        if article:
            articles.append(article)
        else:
            print(f"  skipped (no content found): {topic}")
        time.sleep(0.5)  # small pause between every request so we don't trip the rate limiter at all
    return articles


def chunk_text(text, chunk_size=config.CHUNK_SIZE_WORDS, overlap=config.CHUNK_OVERLAP_WORDS):
    """Simple word-based sliding-window chunking with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def embed_batch(texts):
    """
    Embed a batch of texts in a single Gemini API call. Batching is the real fix
    for free-tier rate limits: 870 chunks as 870 separate requests will blow
    through your RPM/RPD quota fast, but as ~35 batched requests of 25 chunks
    each, you stay comfortably under it.
    """
    for attempt in range(6):
        try:
            result = genai.embed_content(
                model=config.EMBEDDING_MODEL,
                content=texts,  # a list -> batch embedding, one API call for all of them
                task_type="retrieval_document",
                output_dimensionality=config.EMBEDDING_DIMENSIONS,
            )
            return result["embedding"]
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
                wait = 60  # RPM/RPD quota errors reliably clear within a minute or at day-reset
                print(f"  quota hit, waiting {wait}s before retry (attempt {attempt+1}/6)")
            else:
                wait = 2 ** attempt
                print(f"  embed retry {attempt+1}/6: {e}")
            time.sleep(wait)
    raise RuntimeError(f"Failed to embed batch after 6 attempts (first item: {texts[0][:60]}...)")


def main():
    os.makedirs("data", exist_ok=True)

    # Resume support: if we already fetched articles in a previous run, reuse them
    # instead of re-hitting the Wikipedia API from scratch.
    if os.path.exists(config.CORPUS_PATH):
        print("Found existing corpus.json, reusing it (delete it if you want a fresh fetch).")
        with open(config.CORPUS_PATH) as f:
            articles = json.load(f)
    else:
        print(f"Fetching {len(config.WIKI_TOPICS)} Wikipedia articles...")
        articles = fetch_articles(config.WIKI_TOPICS)
        print(f"Fetched {len(articles)} articles.")
        with open(config.CORPUS_PATH, "w") as f:
            json.dump(articles, f)
            
    #Relational DB SQLite for structured metadata storage
    print("Indexing structured metadata in SQLite...")
    conn = sqlite3.connect("data/metadata.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            title TEXT PRIMARY KEY,
            url TEXT,
            word_count INTEGER
        )
    ''')
    for article in articles:
        word_count = len(article["content"].split())
        cursor.execute('''
            INSERT OR REPLACE INTO articles (title, url, word_count)
            VALUES (?, ?, ?)
        ''', (article["title"], article["url"], word_count))
    
    conn.commit()
    conn.close()
    print("Metadata saved to data/metadata.db")

    # Build chunk records
    all_chunks = []
    for article in articles:
        for i, chunk in enumerate(chunk_text(article["content"])):
            all_chunks.append({
                "id": f"{article['title']}::chunk{i}",
                "text": chunk,
                "title": article["title"],
                "url": article["url"],
            })
    print(f"Created {len(all_chunks)} chunks.")

    # Save chunks.json immediately (not at the end) so hybrid_search.py's BM25
    # index has something to load even if embedding gets interrupted below.
    with open("data/chunks.json", "w") as f:
        json.dump(all_chunks, f)

    # Set up Chroma
    client = chromadb.PersistentClient(
        path=config.CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection("wiki_chunks", embedding_function=None)

    # Resume support: skip chunks that are already embedded and stored from a
    # previous (possibly interrupted) run.
    existing_ids = set(collection.get()["ids"])
    remaining_chunks = [c for c in all_chunks if c["id"] not in existing_ids]
    if existing_ids:
        print(f"{len(existing_ids)} chunks already embedded from a previous run, skipping those.")
    print(f"Embedding {len(remaining_chunks)} remaining chunks in batches of {config.EMBED_BATCH_SIZE}...")

    for i in tqdm(range(0, len(remaining_chunks), config.EMBED_BATCH_SIZE), desc="Embedding batches"):
        batch = remaining_chunks[i:i + config.EMBED_BATCH_SIZE]
        embeddings = embed_batch([c["text"] for c in batch])
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[{"title": c["title"], "url": c["url"]} for c in batch],
        )
        time.sleep(2)  # small pause between batches to stay comfortably under RPM limits

    print("Done. Vector store saved to", config.CHROMA_DB_PATH)
    print("Chunk data saved to data/chunks.json")


if __name__ == "__main__":
    main()