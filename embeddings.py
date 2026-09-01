"""
embeddings.py — Embedding model wrapper.

Uses ChromaDB's built-in default embedding function, which runs
the all-MiniLM-L6-v2 model via ONNX runtime. This avoids the heavy
sentence-transformers / keras / tensorflow dependency chain entirely
while providing the exact same embedding model.

The same embedding function is used for both document chunks and
user queries so that the vector space is consistent.
"""

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# ─── Lazy-loaded singleton ───────────────────────────────────────────
# The embedding function is created on first use, not at import time.
# This avoids slowing down Streamlit reruns.
_embed_fn = None


def get_embedding_function():
    """
    Return the ChromaDB embedding function instance (singleton).
    Created once on first call, then reused.
    """
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = DefaultEmbeddingFunction()
    return _embed_fn


def get_embedding(text: str) -> list[float]:
    """
    Convert a single piece of text into an embedding vector.
    Used for embedding a user's question at query time.
    """
    fn = get_embedding_function()
    return fn([text])[0]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of texts into embedding vectors (batch mode).
    Used for embedding document chunks during ingestion.
    """
    fn = get_embedding_function()
    return fn(texts)
