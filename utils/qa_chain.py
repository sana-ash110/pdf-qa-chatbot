from __future__ import annotations

import os
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a PDF document.

You will be given:
- CONTEXT: relevant excerpts retrieved from the document
- QUESTION: the user's question

Rules:
- Answer ONLY from the provided context. Do not use outside knowledge.
- If the context does not contain enough information, say "I couldn't find enough information in the document to answer that."
- Be concise and clear.
- Cite page numbers inline like (Page 3) when you use information from a specific excerpt.
- If multiple excerpts support your answer, cite all relevant pages.
"""

CONTEXT_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}
"""


# ── QAChain ───────────────────────────────────────────────────────────────────

class QAChain:
    """
    RAG chain answering questions about a PDF using Groq + VectorStore.

    Args:
        store:     VectorStore instance (from vector_store.py).
        model:     Groq model name.
        top_k:     Number of text chunks to retrieve per question.
        min_score: Minimum cosine similarity to include a chunk.
    """

    def __init__(
        self,
        store,
        model:     str   = "llama-3.3-70b-versatile",
        top_k:     int   = 5,
        min_score: float = 0.3,
    ) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Add it to your .env file:\n"
                "GROQ_API_KEY=gsk_your_key_here"
            )

        self.store     = store
        self.client    = Groq(api_key=api_key)
        self.model     = model
        self.top_k     = top_k
        self.min_score = min_score
        self.history: list[dict] = []

    def ask(self, question: str) -> dict:
        """
        Answers a question using retrieved context.

        Returns:
            {
                "answer":  str,
                "sources": list[dict],  # page_number, source, score, excerpt
                "images":  list[dict],  # relevant images (may be empty)
            }
        """
        # ── Retrieve text chunks ──────────────────────────────────────────────
        text_results = self.store.search(
            question,
            k           = self.top_k,
            filter_type = "text",
            min_score   = self.min_score,
        )

        # ── Retrieve images ───────────────────────────────────────────────────
        image_results = self.store.search(
            question,
            k           = 3,
            filter_type = "image",
            min_score   = self.min_score,
        )

        # ── No context found ──────────────────────────────────────────────────
        if not text_results:
            return {
                "answer":  "I couldn't find relevant information in the document to answer that question. I can only answer based on the content of the PDF.",
                "sources": [],
                "images":  image_results,
            }

        # ── Build context string ──────────────────────────────────────────────
        context_parts = []
        for r in text_results:
            page_label = f"[Page {r['page_number']}]" if r["page_number"] else ""
            context_parts.append(f"{page_label}\n{r['text'].strip()}")

        context = "\n\n---\n\n".join(context_parts)

        # ── Build Groq messages ───────────────────────────────────────────────
        user_message = CONTEXT_TEMPLATE.format(
            context  = context,
            question = question,
        )

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + self.history
            + [{"role": "user", "content": user_message}]
        )

        # ── Call Groq ─────────────────────────────────────────────────────────
        response = self.client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            temperature = 0.2,
            max_tokens  = 1024,
        )

        answer = response.choices[0].message.content.strip()

        # ── Update history (question only, not full context) ──────────────────
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant", "content": answer})

        # Cap at 10 turns to avoid token overflow
        if len(self.history) > 20:
            self.history = self.history[-20:]

        # ── Build sources ─────────────────────────────────────────────────────
        sources = [
            {
                "page_number": r["page_number"],
                "source":      r["source"],
                "score":       round(r["score"], 3),
                "excerpt":     r["text"][:200],
            }
            for r in text_results
        ]

        return {
            "answer":  answer,
            "sources": sources,
            "images":  image_results,
        }

    def clear_history(self) -> None:
        """Resets conversation. Call when a new PDF is loaded."""
        self.history = []

    def set_top_k(self, k: int) -> None:
        self.top_k = k

    def set_model(self, model: str) -> None:
        self.model = model