# src/rag.py
import os
import streamlit as st
import openai
import numpy as np

# ---------------------------
# Setup Groq API
# ---------------------------
class RAG:
    def __init__(self):
        # Load API key from secrets.toml
        try:
            self.api_key = st.secrets["GROQ_API_KEY"]["api_key"]
        except Exception:
            self.api_key = st.secrets.get("GROQ_API_KEY", "")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        # Load KB
        self.kb_docs = self._load_kb("data/faq.txt")
        self.embed_model = self._try_get_embeddings_model()
        self.kb_embs = None
        if self.embed_model and self.kb_docs:
            self.kb_embs = self.embed_model.encode(self.kb_docs, convert_to_numpy=True)

    # ---------------------------
    # Knowledge Base
    # ---------------------------
    def _load_kb(self, kb_path: str):
        if not os.path.exists(kb_path):
            return []
        with open(kb_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def _try_get_embeddings_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None

    def retrieve_context(self, query: str, top_k: int = 2):
        """Return top-k relevant docs from KB"""
        if not self.kb_docs:
            return []

        if self.embed_model and self.kb_embs is not None:
            q_emb = self.embed_model.encode([query])[0]
            sims = np.dot(self.kb_embs, q_emb)
            idx = np.argsort(sims)[-top_k:][::-1]
            return [self.kb_docs[i] for i in idx]
        else:
            # Fallback: keyword overlap
            scores = []
            q = query.lower()
            for d in self.kb_docs:
                score = sum(1 for token in q.split() if token in d.lower())
                scores.append(score)
            idx = np.argsort(scores)[-top_k:][::-1]
            return [self.kb_docs[i] for i in idx if scores[i] > 0]

    # ---------------------------
    # Prompt builder
    # ---------------------------
    def _build_prompt(self, subject: str, body: str, sentiment: str, priority: str, context_chunks: list) -> str:
        context_text = "\n".join(context_chunks) if context_chunks else "No KB context available."
        empathy = ""
        if sentiment.lower() == "negative":
            empathy = "The customer appears frustrated — acknowledge their frustration politely.\n"
        urgency_note = ""
        if priority.lower() == "urgent":
            urgency_note = "This is URGENT — provide immediate steps.\n"

        prompt = f"""
You are a professional AI support assistant.

Guidelines:
- Maintain professional, empathetic, and concise tone.
- {empathy}{urgency_note}
- Use knowledge base context if helpful.
- Reference any products mentioned.
- Keep reply 5–8 sentences, end with polite sign-off.

Context:
{context_text}

Email Subject: {subject}
Email Body: {body}
Sentiment: {sentiment}
Priority: {priority}

Provide only the reply body (no analysis).
"""
        return prompt

    # ---------------------------
    # LLM Call
    # ---------------------------
    def generate_reply(self, subject: str, body: str, sentiment: str, priority: str) -> str:
        """Retrieve KB + build prompt + call Groq"""
        try:
            context = self.retrieve_context(body, top_k=2)
            prompt = self._build_prompt(subject, body, sentiment, priority, context)

            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"(Reply generation failed: {e})"
