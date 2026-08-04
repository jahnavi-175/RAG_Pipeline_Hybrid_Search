"""
Central config for the RAG pipeline.
Set your Gemini API key as an environment variable before running anything:

    export GEMINI_API_KEY="your-key-here"      # Mac/Linux
    setx GEMINI_API_KEY "your-key-here"        # Windows (restart terminal after)

Get a free key at: https://aistudio.google.com/apikey
"""
import os
import sys
import logging
import warnings

# Disable ChromaDB telemetry before chromadb or posthog gets imported anywhere
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Silence posthog and chromadb telemetry logging errors
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

# Monkeypatch ChromaDB posthog telemetry to prevent argument count error on Windows
try:
    import chromadb.telemetry.product.posthog as chroma_posthog
    chroma_posthog.Posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

# Suppress deprecation and telemetry warnings across project scripts
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# If not in os.environ, check Streamlit's secrets manager
if not GEMINI_API_KEY:
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"].strip()
            os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    except Exception:
        pass

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey "
        "and set it as an environment variable, in Streamlit secrets, or in a .env file."
    )

# Models (both on Gemini's free tier)
EMBEDDING_MODEL = "models/gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768  # gemini-embedding-2 supports output_dimensionality=768
GENERATION_MODEL = "gemini-flash-latest"

# Chunking
CHUNK_SIZE_WORDS = 300      # ~400 tokens
CHUNK_OVERLAP_WORDS = 50

# Embedding batching - batching many chunks per API call keeps you well under
# free-tier rate limits (870 chunks as 1 request each will hit quota fast;
# as ~35 batched requests, it won't).
EMBED_BATCH_SIZE = 25

# Retrieval
TOP_K_VECTOR = 8
TOP_K_BM25 = 8
TOP_K_FINAL = 5            # after fusion, how many chunks go to the LLM

# Storage paths
CHROMA_DB_PATH = "./chroma_store"
CORPUS_PATH = "./data/corpus.json"

# Wikipedia topics to build the demo corpus from.
# Swap this list for any topic cluster you want the RAG system to know about.
WIKI_TOPICS = [
    "Artificial intelligence", "Machine learning", "Large language model",
    "Neural network", "Deep learning", "Natural language processing",
    "Transformer (deep learning architecture)", "Reinforcement learning",
    "Computer vision", "Generative adversarial network",
    "Convolutional neural network", "Recurrent neural network",
    "Attention (machine learning)", "GPT-3", "BERT (language model)",
    "AlphaGo", "Turing test", "Expert system", "Knowledge graph",
    "Vector database", "Retrieval-augmented generation",
    "Prompt engineering", "Fine-tuning (machine learning)",
    "Overfitting", "Gradient descent", "Backpropagation",
    "Support vector machine", "Random forest", "Decision tree learning",
    "K-means clustering", "Bayesian network", "Markov decision process",
    "Speech recognition", "Autonomous robot", "Self-driving car",
    "Chatbot", "ELIZA", "Turing Award", "History of artificial intelligence",
    "AI winter",
]