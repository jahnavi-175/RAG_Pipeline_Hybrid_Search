import streamlit as st
import config
from rag import RAGPipeline

# Page config
st.set_page_config(page_title="Wikipedia RAG AI", page_icon="📚", layout="centered")

st.title("Wikipedia Hybrid Search RAG")
st.markdown("Ask a question! The AI will answer using a hybrid search (semantic + keyword) across our Wikipedia corpus.")

# Initialize the pipeline once and store it in session state
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

with st.spinner("Loading RAG pipeline..."):
    rag = load_pipeline()

# Sidebar for filters
with st.sidebar:
    st.header("Search Filters")
    min_words = st.slider(
        "Minimum Article Length (Words)", 
        min_value=0, max_value=5000, step=100, value=0,
        help="Use SQLite relational filtering to only search long, detailed articles."
    )

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("View Sources"):
                for source in msg["sources"]:
                    st.write(f"- [{source['title']}]({source['url']})")

# User input
if prompt := st.chat_input("E.g., What is retrieval-augmented generation?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            # Pass the min_words filter to the RAG pipeline
            answer, chunks = rag.answer(prompt, return_chunks=True, min_words=min_words)
            
            st.markdown(answer)
            if chunks:
                with st.expander("View Sources"):
                    for chunk in chunks:
                        st.write(f"- [{chunk['title']}]({chunk['url']})")
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": chunks
        })