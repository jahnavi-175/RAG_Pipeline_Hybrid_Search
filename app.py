"""
Day 6-7: Minimal interactive CLI demo.

Run:
    python app.py
"""
import config
from rag import RAGPipeline

def main():
    print("Loading RAG pipeline (hybrid search over Wikipedia AI/ML corpus)...")
    rag = RAGPipeline()
    print("Ready. Ask a question (or type 'exit' to quit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer, chunks = rag.answer(question, return_chunks=True)
        print(f"\nAssistant: {answer}\n")
        print("Sources used:")
        for c in chunks:
            print(f"  - {c['title']} ({c['url']})")
        print()

if __name__ == "__main__":
    main()