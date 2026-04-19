from __future__ import annotations

import os
import pickle

import faiss
import numpy as np

from embeddings import (
    build_index,
    embed_texts,
    embed_texts_clip,
)


# ── VectorStore class ─────────────────────────────────────────────────────────

class VectorStore:
 

    def __init__(
        self,
        embedding_index: dict,
    ) -> None:
        self.index    = embedding_index
        self.metadata = embedding_index["metadata"]
        self.use_clip = embedding_index.get("use_clip_for_text", False)

        embeddings: np.ndarray = embedding_index["embeddings"]

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Embedding index is empty or has wrong shape.")

        self.dim = embeddings.shape[1]

        # IndexFlatIP = exact inner-product search.
        # Because our vectors are L2-normalised, inner product == cosine similarity.
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index.add(embeddings.astype("float32"))

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:       str,
        k:           int  = 5,
        filter_type: str  = "all",   # "all" | "text" | "image"
        min_score:   float = 0.0,
    ) -> list[dict]:
 
        # Embed the query with whichever encoder was used for the index
        if self.use_clip:
            query_vec = embed_texts_clip([query])[0].astype("float32")
        else:
            query_vec = embed_texts([query])[0].astype("float32")

        query_dim = query_vec.shape[0]

        # If query dim doesn't match the index (mixed-mode index), fall back
        # to numpy search on matching entries only
        if query_dim != self.dim:
            return self._fallback_search(query_vec, k, filter_type, min_score)

        # FAISS search — returns distances (scores) and row indices
        scores, indices = self.faiss_index.search(
            query_vec.reshape(1, -1), k * 3  # over-fetch to allow filtering
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:                        # FAISS returns -1 for empty slots
                continue
            if score < min_score:
                continue

            meta = self.metadata[idx]

            if filter_type != "all" and meta["type"] != filter_type:
                continue

            results.append({
                "text":        meta.get("content", ""),
                "page_number": meta.get("page_number"),
                "source":      meta.get("source", ""),
                "score":       float(score),
                "type":        meta.get("type", "text"),
                "chunk_id":    meta.get("chunk_id"),
                # image-specific fields (None for text chunks)
                "image_data":  meta.get("data"),
                "image_ext":   meta.get("ext"),
            })

            if len(results) >= k:
                break

        return results

    def _fallback_search(
        self,
        query_vec:   np.ndarray,
        k:           int,
        filter_type: str,
        min_score:   float,
    ) -> list[dict]:
        """
        Numpy dot-product search used when query dim != index dim.
        This handles the mixed MiniLM (384) + CLIP (512) case.
        """
        embeddings = self.index["embeddings"]
        scores_all = []

        for i, (emb, meta) in enumerate(zip(embeddings, self.metadata)):
            if filter_type != "all" and meta["type"] != filter_type:
                continue
            if emb.shape[0] != query_vec.shape[0]:
                continue
            score = float(np.dot(emb.astype("float32"), query_vec))
            if score >= min_score:
                scores_all.append((score, i))

        scores_all.sort(reverse=True)
        results = []
        for score, idx in scores_all[:k]:
            meta = self.metadata[idx]
            results.append({
                "text":        meta.get("content", ""),
                "page_number": meta.get("page_number"),
                "source":      meta.get("source", ""),
                "score":       score,
                "type":        meta.get("type", "text"),
                "chunk_id":    meta.get("chunk_id"),
                "image_data":  meta.get("data"),
                "image_ext":   meta.get("ext"),
            })
        return results

    # ── LangChain-compatible retriever ────────────────────────────────────────

    def as_retriever(self, k: int = 5, filter_type: str = "text"):
   
        from langchain_core.documents import Document

        def retrieve(query: str) -> list[Document]:
            results = self.search(query, k=k, filter_type=filter_type)
            docs = []
            for r in results:
                docs.append(Document(
                    page_content=r["text"],
                    metadata={
                        "page_number": r["page_number"],
                        "source":      r["source"],
                        "score":       r["score"],
                        "chunk_id":    r["chunk_id"],
                    }
                ))
            return docs

        return retrieve

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str = "vector_store") -> None:

        os.makedirs(directory, exist_ok=True)

        faiss.write_index(
            self.faiss_index,
            os.path.join(directory, "faiss.index")
        )
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "metadata":   self.metadata,
                "dim":        self.dim,
                "use_clip":   self.use_clip,
            }, f)

        print(f"Vector store saved to '{directory}/'")

    @classmethod
    def load(cls, directory: str = "vector_store") -> "VectorStore":
    
        faiss_path    = os.path.join(directory, "faiss.index")
        metadata_path = os.path.join(directory, "metadata.pkl")

        if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Vector store files not found in '{directory}/'. "
                "Run build_vector_store() first."
            )

        faiss_index = faiss.read_index(faiss_path)

        with open(metadata_path, "rb") as f:
            saved = pickle.load(f)

        # Reconstruct a minimal index dict so __init__ can read use_clip
        dummy_index = {
            "embeddings":        np.zeros((1, saved["dim"]), dtype="float32"),
            "metadata":          saved["metadata"],
            "use_clip_for_text": saved["use_clip"],
        }

        instance = cls.__new__(cls)
        instance.index       = dummy_index
        instance.metadata    = saved["metadata"]
        instance.use_clip    = saved["use_clip"]
        instance.dim         = saved["dim"]
        instance.faiss_index = faiss_index

        print(f"Vector store loaded: {faiss_index.ntotal} entries from '{directory}/'")
        return instance

    # ── Info ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_text  = sum(1 for m in self.metadata if m["type"] == "text")
        n_image = sum(1 for m in self.metadata if m["type"] == "image")
        return (
            f"VectorStore(entries={self.faiss_index.ntotal}, "
            f"text={n_text}, images={n_image}, "
            f"dim={self.dim}, clip={self.use_clip})"
        )


# ── Convenience builder ───────────────────────────────────────────────────────

def build_vector_store(
    chunks:            list[dict],
    pages:             list[dict],
    use_clip_for_text: bool = True,
) -> VectorStore:
    """
    One-shot function: embeds chunks + images and returns a VectorStore.

    This is the function you call from app.py after loading + chunking the PDF.

    Args:
        chunks:            Output of chunking.chunk_pages().
        pages:             Output of pdf_loader.load_pdf().
        use_clip_for_text: True  → CLIP for both text and images (unified space,
                                   cross-modal retrieval works).
                           False → MiniLM for text, CLIP for images (faster text
                                   embedding, but cross-modal search won't work).

    Returns:
        A ready-to-search VectorStore.
    """
    index = build_index(chunks, pages, use_clip_for_text=use_clip_for_text)
    return VectorStore(index)

