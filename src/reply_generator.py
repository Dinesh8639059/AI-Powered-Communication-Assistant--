# src/reply_generator.py

from src.rag import RAG

# Initialize RAG once
rag = RAG()

def generate_reply(subject: str, body: str, sentiment: str, priority: str) -> str:
    """
    Generate a context-aware reply using RAG + Groq LLM.
    """
    try:
        reply = rag.generate_reply(subject, body, sentiment, priority)
        return reply
    except Exception as e:
        return f"(Reply generation failed: {e})"
