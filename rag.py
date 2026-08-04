import config
import google.generativeai as genai
from hybrid_search import HybridSearcher

genai.configure(api_key=config.GEMINI_API_KEY)

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below.
If the context doesn't contain the answer, say "I don't have enough information to answer that."
Always cite which source article(s) you used.

Context:
{context}

Question: {question}

Answer (cite sources as [Article Title]):"""

class RAGPipeline:
    def __init__(self):
        self.searcher = HybridSearcher()
        self.model = genai.GenerativeModel(config.GENERATION_MODEL)

    def answer(self, question, k=config.TOP_K_FINAL, return_chunks=False, min_words=0):
        chunks = self.searcher.search(question, k=k, min_words=min_words)
        if not chunks:
            return "No documents found matching the filter criteria.", [] if return_chunks else "No documents found matching the filter criteria."

        context = "\n\n".join(
            f"[{c['title']}]\n{c['text']}" for c in chunks
        )
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        response = self.model.generate_content(prompt)
        try:
            answer_text = response.text
        except ValueError:
            answer_text = "I couldn't generate a response based on the provided context."

        if return_chunks:
            return answer_text, chunks
        return answer_text

if __name__ == "__main__":
    rag = RAGPipeline()
    question = "What is the difference between supervised and reinforcement learning?"
    answer = rag.answer(question)
    print(f"Q: {question}\n\nA: {answer}")